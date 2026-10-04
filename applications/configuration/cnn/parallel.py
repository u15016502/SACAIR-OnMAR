"""
Parallel evaluation of candidate designs.

Scoring the GA's population is the largest single cost in a run - on a
24-timestep CIFAR-10 run it was 48% of all batches processed, more than the
timestep training itself - and every candidate is independent of the others.
So it is also the easiest thing to parallelise, and the place where extra CPUs
or GPUs buy the most.

How work is placed
------------------
A pool of worker processes is started once and reused for the whole run. Each
worker takes a rank on startup and pins itself to one device:

  * multiple GPUs - worker ``r`` uses ``cuda:r % num_gpus``, so a node with
    four GPUs scores four candidates at once. This beats using the GPUs
    together on one candidate (DataParallel), because candidate scoring is
    embarrassingly parallel while a single short probe run is latency-bound.
  * one GPU - workers share it; a small probe model rarely saturates a modern
    GPU on its own, so a few workers still overlap usefully.
  * CPU only - each worker is held to a slice of the available threads.
    Without that, N workers each spawning N BLAS threads oversubscribe the
    node and run slower than one worker would.

Determinism
-----------
Each candidate is seeded from (run seed, timestep, its design) rather than from
the ambient RNG state. That removes the dependence on *evaluation order*, which
is exactly what parallelism changes: a candidate scores the same whether it was
the first or the fifth of its generation, and whether it ran in the parent or a
worker.

It does not make fitness bit-identical across different machine
configurations. CPU reductions are not associative, so the number of BLAS
threads affects rounding, and a few hundred training steps amplify that into
visible differences - measured at up to ~0.06 probe accuracy on a short probe
between a 10-thread serial run and a 2-thread worker. Set
``candidate_eval_threads`` to fix the thread count per evaluation if a run has
to be reproducible across machines with different core counts; it costs some
speed on the serial path.
"""

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from applications.configuration.cnn.cnn_model import ConfigurableCNN, build_optimizer


# Populated inside each worker by _init_worker; never touched in the parent.
_WORKER_STATE: Dict[str, Any] = {}


def design_seed(base_seed: int, timestep: int, design: Dict[str, Any]) -> int:
    """Derive a stable seed for one candidate evaluation.

    Args:
        base_seed: The run's seed.
        timestep: Current timestep.
        design: The candidate design.

    Returns:
        A seed that depends only on these three things, so the same candidate
        always scores the same regardless of where it is evaluated.
    """
    payload = json.dumps(design, sort_keys=True, default=str).encode()
    digest = hashlib.sha1(payload).hexdigest()[:8]
    return (base_seed * 1_000_003 + timestep * 10_007 + int(digest, 16)) % (2 ** 31 - 1)


def _resolve_worker_device(rank: int, device_kind: str, num_gpus: int) -> torch.device:
    """Pick the device this worker should use."""
    if device_kind == 'cuda' and num_gpus > 0:
        return torch.device(f'cuda:{rank % num_gpus}')
    if device_kind == 'mps':
        return torch.device('mps')
    return torch.device('cpu')


def _init_worker(
    rank_counter,
    probe_train: List[Tuple[torch.Tensor, torch.Tensor]],
    probe_val: List[Tuple[torch.Tensor, torch.Tensor]],
    config: Dict[str, Any],
) -> None:
    """Set a worker up once: take a rank, pin a device, hold the probe data.

    The probe batches are sent here at pool startup rather than with every
    task. Shipping them per candidate would dominate the time saved, since
    they are tens of megabytes.
    """
    with rank_counter.get_lock():
        rank = rank_counter.value
        rank_counter.value += 1

    device = _resolve_worker_device(rank, config['device_kind'], config['num_gpus'])

    # Hold each worker to its share of the node's threads. Oversubscription
    # here is the classic way a "parallel" run ends up slower than a serial one.
    threads = max(1, config['threads_per_worker'])
    torch.set_num_threads(threads)
    for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[variable] = str(threads)

    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True

    _WORKER_STATE.update({
        'rank': rank,
        'device': device,
        'probe_train': probe_train,
        'probe_val': probe_val,
        'config': config,
    })


def score_candidate(
    design: Dict[str, Any],
    timestep: int,
    state_dict: Optional[Dict[str, torch.Tensor]] = None,
    state: Optional[Dict[str, Any]] = None,
) -> float:
    """Train a candidate design on the probe batches and return its accuracy.

    Used both by the worker processes and, with ``state`` supplied, by the
    parent process when running serially - so the two paths cannot diverge.

    Args:
        design: Candidate design.
        timestep: Current timestep.
        state_dict: Optional weights to warm-start compatible tensors from.
        state: Worker-style state dict; defaults to this process's own.

    Returns:
        Probe accuracy, or 0.0 for a design that cannot be trained or is over
        the configured budget.
    """
    state = state if state is not None else _WORKER_STATE
    config = state['config']
    device = state['device']
    probe_train = state['probe_train']
    probe_val = state['probe_val']

    # Pinning the thread count here, rather than per worker, is what makes a
    # fitness comparable across machines with different core counts.
    eval_threads = config.get('eval_threads')
    if eval_threads:
        torch.set_num_threads(max(1, int(eval_threads)))

    try:
        torch.manual_seed(design_seed(config['base_seed'], timestep, design))

        candidate = ConfigurableCNN(
            input_shape=tuple(config['input_shape']),
            num_classes=config['num_classes'],
            design=design,
            max_flat_features=config['max_flat_features'],
        ).to(device)

        if candidate.get_num_parameters() > config['max_candidate_parameters']:
            return 0.0
        if candidate.estimate_macs() > config['max_candidate_macs']:
            return 0.0

        if state_dict is not None:
            target = candidate.state_dict()
            merged = {
                key: (state_dict[key].to(device)
                      if key in state_dict and state_dict[key].shape == value.shape
                      else value)
                for key, value in target.items()
            }
            candidate.load_state_dict(merged)

        if config['channels_last'] and device.type == 'cuda':
            candidate = candidate.to(memory_format=torch.channels_last)

        optimizer = build_optimizer(candidate.parameters(), design)
        criterion = nn.CrossEntropyLoss()
        use_amp = config['amp'] and device.type == 'cuda'
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

        candidate.train()
        for _ in range(config['candidate_passes']):
            for inputs, targets in probe_train:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                if config['channels_last'] and device.type == 'cuda':
                    inputs = inputs.to(memory_format=torch.channels_last)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda', enabled=use_amp):
                    loss = criterion(candidate(inputs), targets)

                if not torch.isfinite(loss):
                    return 0.0  # diverged

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        candidate.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in probe_val:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                if config['channels_last'] and device.type == 'cuda':
                    inputs = inputs.to(memory_format=torch.channels_last)
                with torch.autocast(device_type='cuda', enabled=use_amp):
                    predictions = candidate(inputs).argmax(dim=1)
                correct += (predictions == targets).sum().item()
                total += targets.size(0)

        return correct / total if total else 0.0

    except (RuntimeError, ValueError) as error:
        # A degenerate or too-large design is a fitness-0 individual, not a
        # failed run.
        print(f"    Candidate rejected: {type(error).__name__}: {error}")
        return 0.0
    finally:
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def _score_task(args: Tuple[Dict[str, Any], int, Optional[Dict[str, torch.Tensor]]]) -> float:
    """Pool entry point."""
    design, timestep, state_dict = args
    return score_candidate(design, timestep, state_dict)


class CandidateEvaluator:
    """Scores populations of candidate designs, in parallel where it helps."""

    def __init__(
        self,
        config: Dict[str, Any],
        probe_train: List[Tuple[torch.Tensor, torch.Tensor]],
        probe_val: List[Tuple[torch.Tensor, torch.Tensor]],
        num_workers: int = 0,
    ):
        """
        Args:
            config: Everything a worker needs to build and score a model.
            probe_train: Fixed training batches for scoring.
            probe_val: Fixed validation batches for scoring.
            num_workers: Worker processes. 0 or 1 scores in this process,
                which keeps small runs and debugging simple.
        """
        self.config = config
        self.probe_train = probe_train
        self.probe_val = probe_val
        self.num_workers = max(0, num_workers)
        self._pool: Optional[ProcessPoolExecutor] = None

        # Serial fallback state, shaped like a worker's.
        self._local_state = {
            'rank': 0,
            'device': torch.device(config['device']),
            'probe_train': probe_train,
            'probe_val': probe_val,
            'config': config,
        }

    # ------------------------------------------------------------------

    def _ensure_pool(self) -> Optional[ProcessPoolExecutor]:
        """Start the worker pool on first use."""
        if self.num_workers <= 1:
            return None
        if self._pool is not None:
            return self._pool

        import multiprocessing as mp

        # CUDA cannot be initialised in a forked process, and fork is the
        # default on Linux, so the context is set explicitly.
        context = mp.get_context('spawn')
        rank_counter = context.Value('i', 0)

        try:
            self._pool = ProcessPoolExecutor(
                max_workers=self.num_workers,
                mp_context=context,
                initializer=_init_worker,
                initargs=(rank_counter, self.probe_train, self.probe_val, self.config),
            )
        except Exception as error:
            # The usual cause is a caller whose module-level code runs on
            # import: 'spawn' re-imports the main module in each worker, so
            # the entry point has to sit behind an
            # `if __name__ == "__main__":` guard. Scoring still works, just
            # serially, so the run continues rather than failing.
            print(f"  Could not start the candidate pool ({type(error).__name__}: "
                  f"{error}).\n"
                  f"  Scoring candidates in this process instead. If this was "
                  f"unexpected, check that the script's entry point is behind "
                  f"an `if __name__ == \"__main__\":` guard - spawned workers "
                  f"re-import the main module.")
            self.num_workers = 0
            self._pool = None

        return self._pool

    def evaluate(
        self,
        designs: List[Dict[str, Any]],
        timestep: int,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> List[float]:
        """Score a list of designs.

        Args:
            designs: Candidate designs.
            timestep: Current timestep.
            state_dict: Optional weights to warm-start from.

        Returns:
            One fitness per design, in the same order.
        """
        if not designs:
            return []

        pool = self._ensure_pool()

        if pool is None:
            return [
                score_candidate(design, timestep, state_dict, self._local_state)
                for design in designs
            ]

        # Warm-start weights have to travel to each worker, so send them on
        # the CPU and only when they are actually in use.
        payload_state = None
        if state_dict is not None:
            payload_state = {key: value.detach().cpu() for key, value in state_dict.items()}

        tasks = [(design, timestep, payload_state) for design in designs]

        try:
            return list(pool.map(_score_task, tasks))
        except Exception as error:
            print(f"  Parallel scoring failed ({error}); falling back to this process")
            self.shutdown()
            self.num_workers = 0
            return [
                score_candidate(design, timestep, state_dict, self._local_state)
                for design in designs
            ]

    def shutdown(self) -> None:
        """Close the worker pool."""
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass


def plan_workers(
    requested: Optional[int],
    device_kind: str,
    num_gpus: int,
) -> Tuple[int, int]:
    """Decide how many workers to run and how many threads each may use.

    Args:
        requested: Workers asked for; None to decide automatically.
        device_kind: 'cuda', 'mps' or 'cpu'.
        num_gpus: Visible GPUs.

    Returns:
        ``(num_workers, threads_per_worker)``.
    """
    cpu_count = os.cpu_count() or 1

    # On a cluster the scheduler, not the machine, says what is allocated.
    for variable in ('PBS_NCPUS', 'SLURM_CPUS_PER_TASK', 'NSLOTS'):
        value = os.environ.get(variable)
        if value and value.isdigit():
            cpu_count = int(value)
            break

    if requested is None:
        if device_kind == 'cuda' and num_gpus > 1:
            # One worker per GPU: candidates are independent, so this scales
            # better than sharing the GPUs on a single candidate.
            workers = num_gpus
        elif device_kind == 'cuda':
            workers = 2  # a small probe model leaves a modern GPU idle
        elif device_kind == 'mps':
            workers = 1  # one Metal context per process; extra workers thrash
        else:
            workers = max(1, min(8, cpu_count // 2))
    else:
        workers = max(0, requested)

    if workers <= 1:
        return workers, max(1, cpu_count)

    threads = max(1, cpu_count // workers)
    return workers, threads
