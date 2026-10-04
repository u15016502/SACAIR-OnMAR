"""
CNN Configuration Application

Automated configuration of convolutional neural networks for image
classification, with the design space of Appendix A.3.

Timesteps and training state
----------------------------
Algorithm 9 separates two things that are easy to conflate:

    c <- design_algorithm(dataset, t)      choose a design for timestep t
    p <- aa.exec(c, dataset, t)            apply it for one timestep

``exec_design`` implements the second line. It trains the *same* network for
one more epoch, carrying weights and optimiser state across timesteps, so the
performance trace over a run is the trajectory of one training process under a
changing design. Rebuilding the network from scratch each timestep would make
every timestep an independent "epoch 1" and the trace pure noise.

When the design changes mid-run the architecture may change with it. Rather
than discarding what has been learned, every parameter tensor whose name and
shape survive the change is copied into the new network
(:meth:`_transfer_weights`), so a design change costs only the parameters it
actually invalidates.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from typing import Any, Dict, List, Optional, Tuple
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from applications.base_application import BaseApplication
from datasets.image_datasets import ImageDatasetLoader
from applications.configuration.cnn.cnn_model import ConfigurableCNN, build_optimizer
from applications.configuration.cnn.meta_features import (
    CNNMetaFeatureExtractor, get_feature_names,
)
from applications.configuration.cnn.pretrained_init import PretrainedInitialiser
from applications.configuration.cnn.parallel import (
    CandidateEvaluator, design_seed, plan_workers,
)


# Thesis optimiser numbering (Appendix A.3). Ftrl (8) is omitted: it has no
# PyTorch equivalent, and leaving it in the space meant one option in eight
# silently fell back to Adam.
OPTIMIZERS = {
    1: 'adam',
    2: 'adamax',
    3: 'rmsprop',
    4: 'adagrad',
    5: 'adadelta',
    6: 'sgd',
    7: 'nadam',
}

# Activations offered for hidden layers. The thesis lists nine; Sigmoid (5) and
# Softmax (6) are withheld here, because in a hidden layer they only hurt -
# sigmoid saturates and stalls gradients in a deep stack, and softmax
# normalises a hidden representation across the channel dimension, which
# destroys it. Both remain implemented in cnn_model, so a design that specifies
# them still builds.
HIDDEN_ACTIVATIONS = [1, 2, 3, 4, 7, 8, 9]  # ELU, GELU, ReLU, SELU, Softplus, Swish, Tanh


# ----------------------------------------------------------------------
# Design space
#
# Structured per-layer form, matching the variable-length chromosome of
# Appendix A.3: each convolutional and dense layer carries its own options, and
# the number of layers is itself under search.
#
# Where this narrows the thesis space, it is to remove options that cannot win,
# so the GA spends its budget on designs that might:
#   * conv filters stop at 512 rather than 2048 - a 2048-filter convolution
#     over a 28x28 map is orders of magnitude slower without being better
#   * dense nodes stop at 1024 rather than 4096
#   * at most 3 dense layers rather than 6
#   * dropout is capped at 0.5 rather than 1.0 - rates near 1.0 drop nearly
#     every activation, and 1.0 drops all of them
#   * learning rate is sampled log-uniformly, so the search covers 1e-4 to
#     1e-1 evenly in orders of magnitude instead of putting ~99% of its
#     samples above 1e-3
# ----------------------------------------------------------------------

DESIGN_SPACE_SPEC: Dict[str, Any] = {
    'conv_layers': {
        'type': 'block_list',
        'min_blocks': 1,
        'max_blocks': 6,
        'genes': {
            'filters': {'type': 'categorical', 'options': [8, 16, 32, 64, 128, 256, 512]},
            'batch_norm': {'type': 'categorical', 'options': [0, 1]},
            'activation': {'type': 'categorical', 'options': HIDDEN_ACTIVATIONS},
            'dropout': {
                'type': 'continuous',
                'min': 0.0,
                'max': 0.5,
                'disabled_value': -1.0,
                'disabled_prob': 0.4,
            },
            'max_pool': {'type': 'categorical', 'options': [0, 2, 4]},
        },
    },
    'dense_layers': {
        'type': 'block_list',
        'min_blocks': 1,
        'max_blocks': 3,
        'genes': {
            'nodes': {'type': 'categorical', 'options': [16, 32, 64, 128, 256, 512, 1024]},
            'batch_norm': {'type': 'categorical', 'options': [0, 1]},
            'activation': {'type': 'categorical', 'options': HIDDEN_ACTIVATIONS},
            'dropout': {
                'type': 'continuous',
                'min': 0.0,
                'max': 0.5,
                'disabled_value': -1.0,
                'disabled_prob': 0.4,
            },
        },
    },
    'optimizer': {'type': 'categorical', 'options': sorted(OPTIMIZERS)},
    'learning_rate': {
        'type': 'continuous',
        'min': 0.0001,
        'max': 0.1,
        'log': True,
    },
}


# A sane hand-picked design, used when a run needs a starting point.
DEFAULT_DESIGN: Dict[str, Any] = {
    'conv_layers': [
        {'filters': 32, 'batch_norm': 1, 'activation': 3, 'dropout': -1.0, 'max_pool': 2},
        {'filters': 64, 'batch_norm': 1, 'activation': 3, 'dropout': 0.25, 'max_pool': 2},
    ],
    'dense_layers': [
        {'nodes': 128, 'batch_norm': 1, 'activation': 3, 'dropout': 0.3},
    ],
    'optimizer': 1,
    'learning_rate': 0.001,
}


class CNNConfigurationApplication(BaseApplication):
    """
    CNN configuration application for automated hyperparameter optimization.
    """

    def __init__(
        self,
        dataset_name: str,
        random_seed: int = 42,
        device: Optional[str] = None,
        use_data_parallel: bool = True,
        batch_size: int = 128,
        num_workers: int = 4,
        max_train_batches_per_timestep: Optional[int] = None,
        candidate_train_batches: int = 24,
        candidate_val_batches: int = 8,
        candidate_passes: int = 1,
        warm_start_candidates: bool = True,
        max_candidate_parameters: int = 60_000_000,
        max_candidate_macs: int = 250_000_000,
        pretrained_init: bool = True,
        pretrained_donor: str = 'resnet18',
        candidate_workers: Optional[int] = None,
        candidate_eval_threads: Optional[int] = None,
        amp: bool = True,
        channels_last: bool = True,
        val_batches_per_timestep: Optional[int] = None,
    ):
        """
        Initialize the CNN configuration application.

        Args:
            dataset_name: Name of the dataset (mnist, fashion-mnist, cifar-10, cifar-100)
            random_seed: Random seed for reproducibility
            device: Device to use ('cuda', 'cpu', 'mps', or None to auto-detect)
            use_data_parallel: Use DataParallel for multi-GPU training
            batch_size: Batch size for the data loaders
            num_workers: Data loading workers
            max_train_batches_per_timestep: Cap on training batches per
                timestep. None trains a full epoch per timestep; a small value
                makes quick end-to-end runs feasible on a laptop.
            candidate_train_batches: Training batches used to score a GA
                candidate design.
            candidate_val_batches: Validation batches used to score a GA
                candidate design.
            candidate_passes: Passes over the candidate training probe.
            warm_start_candidates: Start candidate scoring from the live
                model's weights where they are shape-compatible, so fitness
                answers "how good would switching to this design be right
                now".

                On by default. A cold-started probe trains each candidate from
                scratch for a few dozen batches, and over so few steps a wide
                dense stack fits faster than a deeper convolutional one - so
                the proxy rewards whichever design starts fastest rather than
                whichever ends best. Left cold, the GA reliably picked a
                single unpooled convolution feeding three large dense layers:
                9.5M parameters scoring *below* a 422K hand-picked design.

                The trade-off: fitness now depends partly on how much a
                candidate resembles the incumbent, since only compatible
                tensors transfer, which biases the search towards the
                incumbent's own lineage. Set False to compare designs purely
                on their own merits, and raise candidate_train_batches well
                above the default if you do.
            max_candidate_parameters: Candidate designs larger than this score
                0 rather than risking an out-of-memory failure.
            max_candidate_macs: Compute budget per image for a candidate
                design, in multiply-accumulates. Candidates above it score 0.
                This is what keeps a GA generation's cost predictable: without
                it, a single design with several wide unpooled convolutions
                takes longer to score than every other candidate combined.
                Raise it on a GPU; the default suits CPU and admits models far
                larger than these datasets need.
            pretrained_init: Seed a design's convolution layers from a
                pretrained donor network when the design is first created,
                instead of starting from random initialisation. Applies only
                when there is no trained state to carry over - that is, at the
                start of a run. A later design change still transfers from the
                live model, which by then holds what the run has learned and
                is a better source than the donor.
            pretrained_donor: Donor architecture for pretrained_init (see
                pretrained_init.SUPPORTED_DONORS). Its weights are downloaded
                on first use; if that is not possible the design falls back to
                default initialisation and says so.
            candidate_workers: Worker processes for scoring GA candidates,
                which is the largest single cost in a run (48% of all batches
                on a measured CIFAR-10 run) and fully parallel. None sizes the
                pool automatically: one worker per GPU on a multi-GPU node,
                else half the allocated CPUs up to 8. 0 or 1 scores in this
                process.
            candidate_eval_threads: Fix the BLAS threads used per candidate
                evaluation. None lets each worker use its share, which is
                fastest but makes a fitness depend on the machine's core count
                (CPU reductions are not associative). Set it - 1 or 2 is
                typical - when a run must reproduce across different machines.
            amp: Use mixed precision on CUDA. Ignored on CPU and MPS, where it
                does not help.
            channels_last: Use channels-last memory layout on CUDA, which
                suits convolutions on tensor cores.
            val_batches_per_timestep: Cap on validation batches used for the
                per-timestep performance figure. None uses the whole
                validation set, which is the honest default; a cap trades a
                noisier signal for ~10% less work per timestep.
        """
        super().__init__(dataset_name, random_seed)

        # Set device: CUDA when present, otherwise CPU.
        #
        # MPS (Apple Silicon) is supported but deliberately not auto-selected.
        # Parts of the PyTorch MPS backend are still incomplete, and a gap
        # there surfaces as a RuntimeError inside candidate scoring, which the
        # GA would read as a fitness of 0 - silently corrupting the search
        # rather than failing loudly. Pass device='mps' to opt in for a local
        # speed-up, having checked the designs in use actually run on it.
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.amp = amp and self.device.type == 'cuda'
        self.channels_last = channels_last and self.device.type == 'cuda'
        self.val_batches_per_timestep = val_batches_per_timestep

        if self.device.type == 'cuda':
            # Fixed input shapes per design, so cuDNN's autotuner pays off.
            torch.backends.cudnn.benchmark = True
            # TF32 matmuls: substantially faster on Ampere and later, and the
            # precision loss is immaterial for this experiment.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self._scaler = torch.amp.GradScaler('cuda', enabled=self.amp)

        # Multi-GPU configuration
        self.use_data_parallel = use_data_parallel and torch.cuda.device_count() > 1
        if self.use_data_parallel:
            print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_train_batches_per_timestep = max_train_batches_per_timestep
        self.candidate_train_batches = candidate_train_batches
        self.candidate_val_batches = candidate_val_batches
        self.candidate_passes = candidate_passes
        self.warm_start_candidates = warm_start_candidates
        self.max_candidate_parameters = max_candidate_parameters
        self.max_candidate_macs = max_candidate_macs
        self.num_candidates_rejected = 0

        # Candidate scoring is the biggest cost in a run and is fully
        # parallel, so it gets its own worker pool.
        self.candidate_workers, self.threads_per_worker = plan_workers(
            candidate_workers,
            self.device.type,
            torch.cuda.device_count() if self.device.type == 'cuda' else 0,
        )
        self.candidate_eval_threads = candidate_eval_threads
        self._candidate_evaluator: Optional[CandidateEvaluator] = None

        # Pretrained initialisation for a design built with no prior state.
        self.pretrained_init = pretrained_init
        self.pretrained_initialiser = (
            PretrainedInitialiser(pretrained_donor) if pretrained_init else None
        )
        self.pretrained_init_stats: Dict[str, Any] = {}

        # Initialize dataset loader
        self.dataset_loader = ImageDatasetLoader(dataset_name, random_seed=random_seed)
        self.dataset_info = self.dataset_loader.get_dataset_info()

        # Data loaders (initialized in load_data)
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # Fixed probe batches for scoring GA candidates. Scoring every
        # candidate on the same data is what makes their fitnesses comparable.
        self._probe_train: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._probe_val: List[Tuple[torch.Tensor, torch.Tensor]] = []

        # Persistent training state, carried across timesteps.
        self.model: Optional[ConfigurableCNN] = None
        self._forward_model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self._optimizer_signature: Optional[Tuple] = None
        self.current_design: Optional[Dict[str, Any]] = None
        self.criterion = nn.CrossEntropyLoss()

        # Meta-feature extractor, created once per run so that features
        # comparing consecutive timesteps (weight distances) have a previous
        # timestep to compare against.
        self.meta_feature_extractor: Optional[CNNMetaFeatureExtractor] = None

        # Run statistics
        self.num_design_changes = 0
        self.num_timesteps_executed = 0
        self.last_transfer_fraction = 0.0

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def load_data(self) -> None:
        """Load and prepare the dataset."""
        if self.train_loader is not None:
            return  # already loaded; a run must not reload between timesteps

        self.train_loader, self.val_loader, self.test_loader = self.dataset_loader.load_dataset(
            val_split=0.1,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
        print(f"Loaded {self.dataset_name}: "
              f"Train={len(self.train_loader.dataset)}, "
              f"Val={len(self.val_loader.dataset)}, "
              f"Test={len(self.test_loader.dataset)}")

        self._build_probe_sets()

    def _build_probe_sets(self) -> None:
        """Materialise the fixed batches used to score GA candidates."""
        self._probe_train = []
        for index, batch in enumerate(self.train_loader):
            if index >= self.candidate_train_batches:
                break
            self._probe_train.append(batch)

        self._probe_val = []
        for index, batch in enumerate(self.val_loader):
            if index >= self.candidate_val_batches:
                break
            self._probe_val.append(batch)

    # ------------------------------------------------------------------
    # Design space
    # ------------------------------------------------------------------

    def get_design_space(self) -> Dict[str, Any]:
        """
        Get the design space for CNN configuration.

        Returns the structured per-layer space of Appendix A.3; see
        DESIGN_SPACE_SPEC for the layout and for where it narrows the thesis
        space.
        """
        return copy.deepcopy(DESIGN_SPACE_SPEC)

    def get_default_design(self) -> Dict[str, Any]:
        """Get a reasonable hand-picked design to seed the search from.

        Scaled to the dataset. Two convolutional blocks are enough for 28x28
        greyscale digits, but too shallow for 32x32 colour images: CIFAR needs
        three blocks with pooling before the receptive field covers enough of
        the image, and punishes a shallow design far more than MNIST does.

        This is a *seed*, not a constraint - the GA is free to evolve away
        from it, and does.
        """
        channels, height, _ = self.dataset_info['input_shape']

        if channels == 1 and height <= 28:
            return copy.deepcopy(DEFAULT_DESIGN)

        return {
            'conv_layers': [
                {'filters': 32, 'batch_norm': 1, 'activation': 3,
                 'dropout': -1.0, 'max_pool': 0},
                {'filters': 64, 'batch_norm': 1, 'activation': 3,
                 'dropout': 0.2, 'max_pool': 2},
                {'filters': 128, 'batch_norm': 1, 'activation': 3,
                 'dropout': 0.3, 'max_pool': 2},
            ],
            'dense_layers': [
                {'nodes': 256, 'batch_norm': 1, 'activation': 3, 'dropout': 0.4},
            ],
            'optimizer': 1,
            'learning_rate': 0.001,
        }

    def get_design_algorithm_config(self) -> Dict[str, Any]:
        """
        GA settings for this application.

        The population is deliberately small: every fitness evaluation trains
        a network, so a generation already costs a multiple of a timestep.
        """
        return {
            'population_size': 8,
            'tournament_size': 3,
            'crossover_rate': 0.75,
            # Almost every offspring is mutated, but only lightly. With a
            # population this small, gating mutation at the individual level
            # as well as the gene level leaves a given gene mutating a couple
            # of times across a whole run, and the search stalls.
            'mutation_rate': 0.9,
            'gene_mutation_rate': 0.12,
            'structure_mutation_rate': 0.2,
            'elite_size': 1,
            'generations_per_step': 1,
        }

    # ------------------------------------------------------------------
    # Model construction and state transfer
    # ------------------------------------------------------------------

    def _build_model(
        self,
        design: Dict[str, Any],
        seed_pretrained: bool = False,
        announce: Optional[bool] = None,
    ) -> ConfigurableCNN:
        """Build a network for a design (no state transfer).

        Args:
            design: Design to build.
            seed_pretrained: Seed the convolution layers from the pretrained
                donor. Only meaningful when there is no trained state to carry
                over; once a run is under way the live model is a better
                source than the donor.
            announce: Override the pretrained initialiser's verbosity.

        Returns:
            The built network, on the application's device.
        """
        model = ConfigurableCNN(
            input_shape=self.dataset_info['input_shape'],
            num_classes=self.dataset_info['num_classes'],
            design=design,
        ).to(self.device)

        if seed_pretrained and self.pretrained_initialiser is not None:
            stats = self.pretrained_initialiser.initialise(model, announce=announce)
            if announce is not False:
                self.pretrained_init_stats = stats

        return model

    def _transfer_weights(self, source: nn.Module, target: nn.Module) -> Tuple[float, float]:
        """Copy every compatible parameter from ``source`` into ``target``.

        A design change usually alters part of the network, not all of it: two
        designs differing only in learning rate share every tensor, and one
        that resizes the third convolution still shares the first two. Copying
        what matches means a design change does not throw away the whole
        training trajectory.

        Args:
            source: Network to copy from.
            target: Network to copy into (modified in place).

        Returns:
            ``(value_fraction, tensor_fraction)`` - the share of the target's
            individual values carried over, and the share of its tensors.
            Both are reported because they answer different questions: a
            change to one convolutional layer can leave most *tensors* intact
            while invalidating most *values*, since the first dense layer
            holds the bulk of the parameters.
        """
        source_state = source.state_dict()
        target_state = target.state_dict()

        merged = {}
        transferred_values = 0
        total_values = 0
        transferred_tensors = 0
        total_tensors = 0

        for key, value in target_state.items():
            total_values += value.numel()
            total_tensors += 1
            candidate = source_state.get(key)
            if candidate is not None and candidate.shape == value.shape:
                merged[key] = candidate.clone()
                transferred_values += value.numel()
                transferred_tensors += 1
            else:
                merged[key] = value

        target.load_state_dict(merged)
        return (
            transferred_values / total_values if total_values else 0.0,
            transferred_tensors / total_tensors if total_tensors else 0.0,
        )

    def _apply_design(self, design: Dict[str, Any]) -> Tuple[float, float]:
        """Make ``design`` the live design, preserving what state it allows.

        Args:
            design: Design to apply.

        Returns:
            ``(value_fraction, tensor_fraction)`` carried over from the
            previous design.
        """
        if not self.validate_design(design):
            raise ValueError(f"Invalid design: {design}")

        if self.model is None:
            # The first design of a run: nothing to carry over, so this is
            # where pretrained values are seeded if they are enabled.
            self.model = self._build_model(design, seed_pretrained=True)
            if self.channels_last:
                self.model = self.model.to(memory_format=torch.channels_last)
            self._warn_if_over_budget(self.model)
            self._forward_model = (
                nn.DataParallel(self.model) if self.use_data_parallel else self.model
            )
            self.current_design = copy.deepcopy(design)
            self.last_transfer_fraction = 0.0
            self._rebuild_optimizer(design)
            return 0.0, 0.0

        if design == self.current_design:
            self.last_transfer_fraction = 1.0
            return 1.0, 1.0  # nothing to do; state carries over untouched

        # Design changed mid-run: rebuild and keep every compatible tensor.
        # Layers the change invalidated are left at default initialisation
        # rather than reseeded from the donor - by this point the run's own
        # weights are the better source, and the donor's would be a step
        # backwards from what has already been learned here.
        new_model = self._build_model(design)
        self._warn_if_over_budget(new_model)
        value_fraction, tensor_fraction = self._transfer_weights(self.model, new_model)

        self.model = new_model
        self._forward_model = (
            nn.DataParallel(self.model) if self.use_data_parallel else self.model
        )
        self.current_design = copy.deepcopy(design)
        self.num_design_changes += 1
        self.last_transfer_fraction = value_fraction
        self._rebuild_optimizer(design)

        return value_fraction, tensor_fraction

    def _warn_if_over_budget(self, model: ConfigurableCNN) -> None:
        """Warn when the live design exceeds the candidate compute budget.

        A design chosen by one of the approaches is always executed - refusing
        it would misreport the run. But a design this expensive makes a
        timestep take many times longer than the rest, so say so rather than
        leaving an unexplained stall.
        """
        macs = model.estimate_macs()
        if macs > self.max_candidate_macs:
            print(
                f"    Note: this design needs {macs/1e6:.0f}M MACs/image, above the "
                f"{self.max_candidate_macs/1e6:.0f}M budget; this timestep will be slow"
            )

    def _rebuild_optimizer(self, design: Dict[str, Any]) -> None:
        """Create the optimiser, reusing its state when the design allows.

        Optimiser state (Adam's moment estimates, SGD's momentum buffers) is
        worth keeping: discarding it every timestep would reset the adaptive
        step sizes just as they become useful. It is only valid, though, while
        the optimiser type, learning rate and parameter tensors are unchanged.
        """
        signature = (
            int(design.get('optimizer', 1)),
            float(design.get('learning_rate', 0.001)),
            id(self.model),
        )

        if self.optimizer is not None and signature == self._optimizer_signature:
            return

        self.optimizer = self._create_optimizer(design)
        self._optimizer_signature = signature

    def _create_optimizer(self, design: Dict[str, Any]) -> optim.Optimizer:
        """Create the optimiser named by a design."""
        return build_optimizer(self.model.parameters(), design)

    # ------------------------------------------------------------------
    # Timestep execution
    # ------------------------------------------------------------------

    def exec_design(self, design: Dict[str, Any], timestep: int) -> Dict[str, float]:
        """Apply a design for one timestep and measure it.

        This is ``aa.exec(c, dataset, t)``: it advances the persistent network
        by one epoch under ``design``, it does not start a new training run.

        Args:
            design: Design to apply.
            timestep: Current timestep.

        Returns:
            Metrics for this timestep, including 'performance' (validation
            accuracy), 'train_loss' and 'weights_transferred'.
        """
        if self.train_loader is None:
            self.load_data()

        start_time = time.time()
        transferred_values, transferred_tensors = self._apply_design(design)

        # One epoch of training on the live model.
        self._forward_model.train()
        total_loss = 0.0
        num_batches = 0

        for inputs, targets in self.train_loader:
            if (
                self.max_train_batches_per_timestep is not None
                and num_batches >= self.max_train_batches_per_timestep
            ):
                break

            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            if self.channels_last:
                inputs = inputs.to(memory_format=torch.channels_last)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', enabled=self.amp):
                outputs = self._forward_model(inputs)
                loss = self.criterion(outputs, targets)

            if not torch.isfinite(loss):
                # A diverged design (typically a large learning rate) would
                # otherwise poison the weights with NaNs for every later
                # timestep. Skip the update and let the design be judged on
                # the accuracy it produces.
                continue

            self._scaler.scale(loss).backward()
            self._scaler.step(self.optimizer)
            self._scaler.update()

            total_loss += loss.item()
            num_batches += 1

        val_accuracy = self._evaluate_loader(
            self.val_loader, max_batches=self.val_batches_per_timestep
        )
        self.is_trained = True
        self.num_timesteps_executed += 1

        return {
            'performance': val_accuracy,
            'val_accuracy': val_accuracy,
            'train_loss': total_loss / num_batches if num_batches else float('nan'),
            'weights_transferred': transferred_values,
            'tensors_transferred': transferred_tensors,
            'num_parameters': float(self.model.get_num_parameters()),
            'timestep_time': time.time() - start_time,
        }

    def _evaluator_config(self) -> Dict[str, Any]:
        """Everything a worker needs to build and score a candidate."""
        return {
            'input_shape': list(self.dataset_info['input_shape']),
            'num_classes': self.dataset_info['num_classes'],
            'max_flat_features': 8192,
            'candidate_passes': self.candidate_passes,
            'max_candidate_parameters': self.max_candidate_parameters,
            'max_candidate_macs': self.max_candidate_macs,
            'base_seed': self.random_seed,
            'device': str(self.device),
            'device_kind': self.device.type,
            'num_gpus': torch.cuda.device_count() if self.device.type == 'cuda' else 0,
            'threads_per_worker': self.threads_per_worker,
            'eval_threads': self.candidate_eval_threads,
            'amp': self.amp,
            'channels_last': self.channels_last,
        }

    def _ensure_evaluator(self) -> CandidateEvaluator:
        """Build the candidate evaluator, once the probe data exists."""
        if self._candidate_evaluator is None:
            if self.train_loader is None:
                self.load_data()
            self._candidate_evaluator = CandidateEvaluator(
                config=self._evaluator_config(),
                probe_train=self._probe_train,
                probe_val=self._probe_val,
                num_workers=self.candidate_workers,
            )
            if self.candidate_workers > 1:
                print(f"  Candidate scoring: {self.candidate_workers} worker(s), "
                      f"{self.threads_per_worker} thread(s) each, device "
                      f"{self.device.type}"
                      + (f" x{torch.cuda.device_count()}" if self.device.type == 'cuda' else ""))
        return self._candidate_evaluator

    def evaluate_population(
        self,
        designs: List[Dict[str, Any]],
        timestep: int,
    ) -> List[float]:
        """Score a whole population of candidate designs at once.

        This is the parallel entry point. Scoring a population is the largest
        cost in a run and each candidate is independent, so the work is spread
        over worker processes - one per GPU on a multi-GPU node, or a share of
        the allocated CPUs otherwise.

        Args:
            designs: Candidate designs.
            timestep: Current timestep.

        Returns:
            One fitness per design, in order.
        """
        if not designs:
            return []

        evaluator = self._ensure_evaluator()
        state_dict = None
        if self.warm_start_candidates and self.model is not None:
            state_dict = self.model.state_dict()

        fitnesses = evaluator.evaluate(designs, timestep, state_dict)

        for design, fitness in zip(designs, fitnesses):
            if fitness == 0.0:
                self.num_candidates_rejected += 1

        return fitnesses

    def evaluate_candidate(self, design: Dict[str, Any], timestep: int) -> float:
        """Score one candidate design for the GA.

        Routed through :meth:`evaluate_population` so the single- and
        many-candidate paths cannot diverge. Seeding is derived from the run
        seed, the timestep and the design itself, so a candidate scores
        identically however many workers are in use.

        Args:
            design: Candidate design.
            timestep: Current timestep.

        Returns:
            Probe accuracy, or 0.0 for a design that cannot be trained.
        """
        return self.evaluate_population([design], timestep)[0]


    def _create_optimizer_for(self, model: nn.Module, design: Dict[str, Any]) -> optim.Optimizer:
        """Create an optimiser for an arbitrary model (used for candidates)."""
        saved_model = self.model
        self.model = model
        try:
            return self._create_optimizer(design)
        finally:
            self.model = saved_model

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, design: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Evaluate on the test set.

        Args:
            design: Optional design to apply before evaluating. Applying it
                preserves the trained state where compatible; this does *not*
                start a fresh training run, so the number reported is the one
                the run actually earned.

        Returns:
            Test metrics, including 'test_performance' and 'test_accuracy'.
        """
        if self.train_loader is None:
            self.load_data()

        if design is not None and design != self.current_design:
            self._apply_design(design)

        if self.model is None:
            raise RuntimeError("No model to evaluate; run at least one timestep first")

        test_accuracy = self._evaluate_loader(self.test_loader)
        val_accuracy = self._evaluate_loader(self.val_loader)

        return {
            'test_performance': test_accuracy,
            'test_accuracy': test_accuracy,
            'val_accuracy': val_accuracy,
            'num_parameters': float(self.model.get_num_parameters()),
        }

    def _evaluate_loader(self, data_loader, max_batches: Optional[int] = None) -> float:
        """Accuracy of the live model on a data loader.

        Args:
            data_loader: Loader to evaluate on.
            max_batches: Stop after this many batches; None uses all of them.

        Returns:
            Accuracy.
        """
        self._forward_model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for index, (inputs, targets) in enumerate(data_loader):
                if max_batches is not None and index >= max_batches:
                    break
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                if self.channels_last:
                    inputs = inputs.to(memory_format=torch.channels_last)
                with torch.autocast(device_type='cuda', enabled=self.amp):
                    predicted = self._forward_model(inputs).argmax(dim=1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        return correct / total if total else 0.0

    # ------------------------------------------------------------------
    # Full training run (baselines and standalone use)
    # ------------------------------------------------------------------

    def train(self, design: Dict[str, Any], timesteps: Optional[int] = None) -> Dict[str, float]:
        """
        Train a single fixed design for a number of timesteps.

        This is the static-design path, used by the baselines and by the test
        script. It resets the training state first, so it is a self-contained
        training run rather than a continuation; the meta-learning approaches
        drive :meth:`exec_design` instead, one timestep at a time.

        Args:
            design: Design to train
            timesteps: Number of epochs (default: 50)

        Returns:
            Dictionary of performance metrics
        """
        if timesteps is None:
            timesteps = 50

        self.reset_run_state()
        if self.train_loader is None:
            self.load_data()

        start_time = time.time()
        best_val_accuracy = 0.0
        val_accuracies = []
        train_losses = []

        for epoch in range(timesteps):
            metrics = self.exec_design(design, epoch)
            val_accuracies.append(metrics['performance'])
            train_losses.append(metrics['train_loss'])
            best_val_accuracy = max(best_val_accuracy, metrics['performance'])

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{timesteps}], "
                      f"Loss: {metrics['train_loss']:.4f}, "
                      f"Val Acc: {metrics['performance']:.4f}")

        return {
            'performance': best_val_accuracy,
            'val_accuracy': best_val_accuracy,
            'final_val_accuracy': val_accuracies[-1] if val_accuracies else 0.0,
            'final_train_loss': train_losses[-1] if train_losses else float('nan'),
            'best_design': copy.deepcopy(design),
            'training_time': time.time() - start_time,
        }

    # ------------------------------------------------------------------
    # Meta-features
    # ------------------------------------------------------------------

    def get_meta_features(self) -> Dict[str, float]:
        """Extract static dataset-level meta-features."""
        return {
            'num_train_samples': len(self.train_loader.dataset) if self.train_loader else 0,
            'num_val_samples': len(self.val_loader.dataset) if self.val_loader else 0,
            'num_test_samples': len(self.test_loader.dataset) if self.test_loader else 0,
            'num_classes': self.dataset_info['num_classes'],
            'input_channels': self.dataset_info['channels'],
            'input_height': self.dataset_info['height'],
            'input_width': self.dataset_info['width'],
            'total_pixels': (
                self.dataset_info['height']
                * self.dataset_info['width']
                * self.dataset_info['channels']
            ),
        }

    def get_meta_feature_names(self) -> List[str]:
        """The fixed meta-feature schema (see meta_features.META_FEATURE_NAMES)."""
        return get_feature_names()

    def extract_meta_features(self, timestep: int = 0) -> Dict[str, Any]:
        """
        Extract meta-features describing the current state of the run.

        The returned dictionary always carries the same keys in the same order,
        whatever is computable at this timestep, so the vectors stored in the
        knowledge repository are all the same length.

        Args:
            timestep: Current training timestep (epoch)

        Returns:
            Dictionary of meta-features
        """
        if self.train_loader is None:
            self.load_data()

        if self.meta_feature_extractor is None:
            self.meta_feature_extractor = CNNMetaFeatureExtractor(self.model, self.device)
        else:
            # Point the extractor at the live model: a design change replaces
            # the model object, but the extractor must keep its history.
            self.meta_feature_extractor.model = self.model

        return self.meta_feature_extractor.extract_meta_features(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            dataset_info=self.dataset_info,
            timestep=timestep,
        )

    # ------------------------------------------------------------------
    # Lifecycle and metadata
    # ------------------------------------------------------------------

    def reset_run_state(self) -> None:
        """Discard the trained state so a new run starts clean."""
        # Worker processes hold the previous run's probe data; drop the pool so
        # the next run does not score candidates against stale batches.
        if self._candidate_evaluator is not None:
            self._candidate_evaluator.shutdown()
            self._candidate_evaluator = None

        self.model = None
        self._forward_model = None
        self.optimizer = None
        self._optimizer_signature = None
        self.current_design = None
        self.meta_feature_extractor = None
        self.is_trained = False
        self.num_design_changes = 0
        self.num_timesteps_executed = 0
        self.last_transfer_fraction = 0.0
        self.num_candidates_rejected = 0
        self.pretrained_init_stats = {}

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        self._scaler = torch.amp.GradScaler('cuda', enabled=self.amp)

    def get_application_type(self) -> str:
        """Get the type of application."""
        return 'configuration'

    def supports_dynamic_designs(self) -> bool:
        """The design may change between timesteps; training state carries over."""
        return True

    def has_builtin_design_algorithm(self) -> bool:
        """This application applies the design it is given; the GA searches."""
        return False

    def get_num_timesteps(self) -> int:
        """Default number of timesteps for this application (thesis: 80)."""
        return 80

    def get_run_statistics(self) -> Dict[str, Any]:
        """Summary of what happened during the current run."""
        return {
            'num_timesteps_executed': self.num_timesteps_executed,
            'num_design_changes': self.num_design_changes,
            'last_transfer_fraction': self.last_transfer_fraction,
            'num_parameters': self.model.get_num_parameters() if self.model else 0,
            'num_candidates_rejected': self.num_candidates_rejected,
            'pretrained_init': self.pretrained_init_stats,
            'architecture': self.model.describe() if self.model else None,
        }
