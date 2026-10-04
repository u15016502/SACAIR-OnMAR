"""
Video Classification Configuration Application

Automated configuration of a Temporal Segment Network for video
classification, with the seven-option design space of the thesis's video
chromosome (``reference/thesis_chromosomes.py``).

Timesteps and training state
----------------------------
Algorithm 9 separates choosing a design from applying it:

    c <- design_algorithm(dataset, t)      choose a design for timestep t
    p <- aa.exec(c, dataset, t)            apply it for one timestep

:meth:`VideoConfigurationApplication.exec_design` implements the second line.
It trains the *same* network for one more epoch, carrying weights and
optimiser state across timesteps, so a run's performance trace is the
trajectory of one training process under a changing design - as in the CNN
application, and for the same reason: rebuilding the network each timestep
would make every timestep an independent "epoch 1".

When the design changes, every parameter tensor whose name and shape survive
the change is copied into the new network (:meth:`_transfer_weights`), so a
design change costs only the parameters it actually invalidates. Changing the
number of segments or the consensus function keeps nearly everything; changing
the base architecture keeps nearly nothing.

Datasets
--------
The datasets the original command line named (UCF101, HMDB51, LMTD) are tens
of gigabytes and need their frames extracted before any of this code runs,
and no loader for them exists in this repository. Two paths are therefore
supported:

``dataset_name='synthetic'``
    A generated frame-folder dataset (:mod:`video_configuration.synthetic`),
    in the same on-disk layout as UCF101. It exists so the pipeline can be
    verified end to end without a download; its classes are separated by
    motion direction, so accuracy on it demonstrates that the implementation
    works, not that it is competitive.

``dataset_name='custom'`` (also 'ucf101', 'hmdb51')
    A real frame-extracted dataset, via ``root_path``, ``train_list``,
    ``val_list`` and ``num_classes``. The list files are the standard
    ``<frame directory> <number of frames> <class index>`` format.

Design-space availability
-------------------------
Base architectures 8 and 9 (BN-Inception and InceptionV3) are defined in a
``tf_model_zoo`` package that is not bundled. They are dropped from the design
space when it is not importable, and reported in :attr:`excluded_options`;
pass ``strict_design_space=True`` to demand them and fail instead.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from typing import Any, Dict, List, Optional, Tuple
import copy
import importlib.util
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision

from applications.base_application import BaseApplication
from applications.configuration.video.meta_features import (
    VideoMetaFeatureExtractor, get_feature_names,
)
from video_configuration.models import TSN
from video_configuration.dataset import TSNDataSet
from video_configuration.transforms import (
    GroupScale, GroupCenterCrop, GroupNormalize, Stack, ToTorchFormatTensor,
)
from video_configuration import synthetic


# Thesis base-architecture numbering. 1-7 come from torchvision; 8 and 9 need
# the tf_model_zoo directory that is not bundled.
BASE_ARCHITECTURES: Dict[int, str] = {
    1: 'resnet18',
    2: 'resnet34',
    3: 'resnet50',
    4: 'resnet101',
    5: 'resnet152',
    6: 'vgg16',
    7: 'vgg19',
    8: 'BNInception',
    9: 'InceptionV3',
}

# Which architectures need the unbundled model zoo.
MODEL_ZOO_ARCHITECTURES = (8, 9)

# Thesis consensus numbering. See video_configuration.ops.basic_ops for why
# the fifth is 'weighted' rather than the original CLI's 'rnn'.
CONSENSUS_FUNCTIONS: Dict[int, str] = {
    1: 'avg',
    2: 'max',
    3: 'topk',
    4: 'identity',
    5: 'weighted',
}


# ----------------------------------------------------------------------
# Design space
#
# The seven options of the thesis video chromosome, with its ranges:
#   keyframe extraction      1-5
#   number of segments       2-10
#   base architecture        1-9
#   consensus function       1-5
#   learning rate            0.0009 - 0.01
#   dropout                  0.4 - 0.65
#   gradient norm clipping   1.0 - 20.0
#
# The learning rate is sampled uniformly rather than log-uniformly, unlike the
# CNN application's: the range spans one order of magnitude, so uniform
# sampling already covers it evenly, and this keeps the thesis's declaration.
# ----------------------------------------------------------------------

DESIGN_SPACE_SPEC: Dict[str, Any] = {
    'keyframe_extraction': {'type': 'categorical', 'options': [1, 2, 3, 4, 5]},
    'num_segments': {'type': 'categorical', 'options': [2, 3, 4, 5, 6, 7, 8, 9, 10]},
    'base_architecture': {'type': 'categorical', 'options': sorted(BASE_ARCHITECTURES)},
    'consensus_function': {'type': 'categorical', 'options': sorted(CONSENSUS_FUNCTIONS)},
    'learning_rate': {'type': 'continuous', 'min': 0.0009, 'max': 0.01},
    'dropout': {'type': 'continuous', 'min': 0.4, 'max': 0.65},
    'gradient_norm_clipping': {'type': 'continuous', 'min': 1.0, 'max': 20.0},
}


# A hand-picked design to seed the search from: the TSN paper's defaults,
# scaled down to the smallest base architecture. ResNet18 rather than
# ResNet101 because the design algorithm gets only theta_t calls in a run, and
# a seed whose every evaluation costs minutes spends the whole budget on one
# generation. The genetic algorithm can evolve towards the larger models.
DEFAULT_DESIGN: Dict[str, Any] = {
    'keyframe_extraction': 1,    # random frame per segment (TSN's sampler)
    'num_segments': 3,           # the TSN paper's default
    'base_architecture': 1,      # resnet18
    'consensus_function': 1,     # average
    'learning_rate': 0.001,
    'dropout': 0.5,
    'gradient_norm_clipping': 20.0,
}


def model_zoo_available() -> bool:
    """Whether the ``tf_model_zoo`` package is importable.

    Returns:
        True if BN-Inception and InceptionV3 base models can be built.
    """
    try:
        return importlib.util.find_spec('tf_model_zoo') is not None
    except (ImportError, ValueError):
        return False


class VideoConfigurationApplication(BaseApplication):
    """
    Video classification configuration application.
    """

    def __init__(
        self,
        dataset_name: str = 'synthetic',
        random_seed: int = 42,
        device: Optional[str] = None,
        modality: str = 'RGB',
        batch_size: int = 4,
        num_workers: int = 0,
        max_train_batches_per_timestep: Optional[int] = None,
        val_batches_per_timestep: Optional[int] = None,
        candidate_train_batches: int = 4,
        candidate_val_batches: int = 2,
        max_candidate_parameters: int = 150_000_000,
        warm_start_candidates: bool = True,
        partial_bn: bool = True,
        strict_design_space: bool = False,
        # Real-dataset configuration
        root_path: str = '',
        train_list: Optional[str] = None,
        val_list: Optional[str] = None,
        num_classes: Optional[int] = None,
        image_tmpl: str = 'img_{:05d}.jpg',
        # Synthetic-dataset configuration
        synthetic_root: str = './data/synthetic_video',
        synthetic_classes: int = 4,
        synthetic_videos_per_class: int = 8,
        synthetic_frames: int = 16,
    ):
        """
        Initialize the video configuration application.

        Args:
            dataset_name: 'synthetic' to generate a dataset, or 'ucf101',
                'hmdb51' or 'custom' to use a frame-extracted one via
                ``train_list`` / ``val_list``.
            random_seed: Random seed for reproducibility
            device: 'cuda', 'mps', 'cpu', or None to auto-detect
            modality: 'RGB', 'Flow' or 'RGBDiff'
            batch_size: Videos per batch. Small by default: a batch holds
                ``batch_size * num_segments`` images, and num_segments is
                itself under search and may be as high as 10.
            num_workers: Data loading workers
            max_train_batches_per_timestep: Cap on training batches per
                timestep. None trains a full epoch.
            val_batches_per_timestep: Cap on validation batches per timestep.
            candidate_train_batches: Training batches used to score a GA
                candidate design.
            candidate_val_batches: Validation batches used to score a
                candidate.
            max_candidate_parameters: Candidates larger than this are
                rejected rather than evaluated. Without a cap, one ResNet152
                candidate costs more than every other candidate in its
                generation combined.
            warm_start_candidates: Start candidate scoring from the live
                model's weights where they are shape-compatible, so fitness
                answers "how good would switching to this design be now".
            partial_bn: Freeze all but the first BatchNorm layer, as the TSN
                paper does when fine-tuning.
            strict_design_space: Demand the full design space, raising if the
                model zoo is missing, rather than dropping those options.
            root_path: Prefix for the paths in the list files.
            train_list: Training list file.
            val_list: Validation list file.
            num_classes: Number of classes in a real dataset.
            image_tmpl: Frame filename template.
            synthetic_root: Where to generate the synthetic dataset.
            synthetic_classes: Motion classes to generate.
            synthetic_videos_per_class: Videos per class to generate.
            synthetic_frames: Frames per generated video.
        """
        super().__init__(dataset_name, random_seed)

        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = torch.device(device)

        self.modality = modality
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_train_batches_per_timestep = max_train_batches_per_timestep
        self.val_batches_per_timestep = val_batches_per_timestep
        self.candidate_train_batches = candidate_train_batches
        self.candidate_val_batches = candidate_val_batches
        self.max_candidate_parameters = max_candidate_parameters
        self.warm_start_candidates = warm_start_candidates
        self.partial_bn = partial_bn
        self.strict_design_space = strict_design_space

        self.root_path = root_path
        self.train_list = train_list
        self.val_list = val_list
        self.image_tmpl = image_tmpl
        self._num_classes = num_classes

        self.synthetic_root = synthetic_root
        self.synthetic_classes = synthetic_classes
        self.synthetic_videos_per_class = synthetic_videos_per_class
        self.synthetic_frames = synthetic_frames

        self.dataset_info: Dict[str, Any] = {}
        self.excluded_options: Dict[str, List[int]] = {}

        # Loaders are rebuilt when a design changes the genes they depend on.
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self._loader_signature: Optional[Tuple] = None

        # Persistent training state, carried across timesteps.
        self.model: Optional[TSN] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self._optimizer_signature: Optional[Tuple] = None
        self.current_design: Optional[Dict[str, Any]] = None
        self.criterion = nn.CrossEntropyLoss()
        self.meta_feature_extractor: Optional[VideoMetaFeatureExtractor] = None
        self._last_metrics: Dict[str, float] = {}

        # Run statistics
        self.num_design_changes = 0
        self.num_timesteps_executed = 0
        self.num_candidates_rejected = 0
        self.last_transfer_fraction = 0.0

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def load_data(self) -> None:
        """Prepare the dataset, generating it if it is the synthetic one."""
        if self.dataset_info:
            return  # already prepared

        if self.dataset_name == 'synthetic':
            manifest = synthetic.generate(
                root=self.synthetic_root,
                num_classes=self.synthetic_classes,
                videos_per_class=self.synthetic_videos_per_class,
                num_frames=self.synthetic_frames,
                random_seed=self.random_seed,
                image_tmpl=self.image_tmpl,
            )
            self.train_list = manifest['train_list']
            self.val_list = manifest['val_list']
            self.root_path = ''
            self.dataset_info = {
                'name': 'synthetic',
                'num_classes': manifest['num_classes'],
                'num_train_videos': manifest['num_train'],
                'num_val_videos': manifest['num_val'],
                'frames_per_video': manifest['num_frames'],
                'frame_size': manifest['frame_size'],
                'class_names': manifest['class_names'],
            }
        else:
            if not self.train_list or not self.val_list:
                raise ValueError(
                    f"Dataset '{self.dataset_name}' needs frame-extracted "
                    f"video and list files: pass train_list and val_list "
                    f"(lines of '<frame directory> <num frames> <class "
                    f"index>'), plus num_classes. No loader for UCF101, "
                    f"HMDB51 or LMTD exists in this repository - extract "
                    f"frames and build the list files first, or use "
                    f"dataset_name='synthetic' to exercise the pipeline."
                )

            num_train = sum(1 for line in open(self.train_list) if line.strip())
            num_val = sum(1 for line in open(self.val_list) if line.strip())

            if self._num_classes is None:
                # Infer from the list files rather than guessing.
                labels = set()
                for list_file in (self.train_list, self.val_list):
                    for line in open(list_file):
                        if line.strip():
                            labels.add(int(line.strip().split(' ')[2]))
                self._num_classes = max(labels) + 1

            self.dataset_info = {
                'name': self.dataset_name,
                'num_classes': self._num_classes,
                'num_train_videos': num_train,
                'num_val_videos': num_val,
                'frames_per_video': 0,   # varies per video
                'frame_size': 0,         # unknown until read
                'class_names': [],
            }

        print(f"Loaded {self.dataset_name}: "
              f"Train={self.dataset_info['num_train_videos']} videos, "
              f"Val={self.dataset_info['num_val_videos']} videos, "
              f"Classes={self.dataset_info['num_classes']}")

    def _build_loaders(self, design: Dict[str, Any], model: TSN) -> None:
        """Build the data loaders for a design, if they are not current.

        The loaders depend on three design genes - the segment count and the
        keyframe strategy decide which frames are read, and the base
        architecture decides the crop size and normalisation - so they are
        rebuilt whenever one of those changes, and reused otherwise.

        Args:
            design: The design in force.
            model: The built network, for its crop size and normalisation.
        """
        signature = (
            int(design['num_segments']),
            int(design['keyframe_extraction']),
            int(design['base_architecture']),
        )
        if self._loader_signature == signature and self.train_loader is not None:
            return

        crop_size = model.crop_size
        scale_size = model.scale_size
        normalize = GroupNormalize(model.input_mean, model.input_std)

        train_transform = torchvision.transforms.Compose([
            model.get_augmentation(),
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])
        val_transform = torchvision.transforms.Compose([
            GroupScale(int(scale_size)),
            GroupCenterCrop(crop_size),
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])

        common = {
            'root_path': self.root_path,
            'num_segments': int(design['num_segments']),
            'modality': self.modality,
            'image_tmpl': self.image_tmpl,
            'keyframe_strategy': int(design['keyframe_extraction']),
            'random_seed': self.random_seed,
        }

        train_dataset = TSNDataSet(
            list_file=self.train_list, transform=train_transform,
            random_shift=True, test_mode=False, **common
        )
        val_dataset = TSNDataSet(
            list_file=self.val_list, transform=val_transform,
            random_shift=False, test_mode=True, **common
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, drop_last=False,
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers,
        )
        self._loader_signature = signature

    # ------------------------------------------------------------------
    # Design space
    # ------------------------------------------------------------------

    def get_design_space(self) -> Dict[str, Any]:
        """
        Get the design space for video configuration.

        Returns the seven options of the thesis video chromosome. Base
        architectures needing the unbundled ``tf_model_zoo`` are dropped when
        it is not importable (see :attr:`excluded_options`).
        """
        spec = copy.deepcopy(DESIGN_SPACE_SPEC)
        self.excluded_options = {}

        if not model_zoo_available():
            if self.strict_design_space:
                raise ImportError(
                    f"Base architectures "
                    f"{[BASE_ARCHITECTURES[a] for a in MODEL_ZOO_ARCHITECTURES]} "
                    f"need the tf_model_zoo package, which is not bundled "
                    f"(442 files, 51.8 MB). Recover it with 'git checkout "
                    f"20e8a896 -- applications/video_configuration/"
                    f"tf_model_zoo', or pass strict_design_space=False to "
                    f"search the remaining architectures."
                )

            spec['base_architecture']['options'] = [
                a for a in sorted(BASE_ARCHITECTURES)
                if a not in MODEL_ZOO_ARCHITECTURES
            ]
            self.excluded_options['base_architecture'] = list(MODEL_ZOO_ARCHITECTURES)

            warnings.warn(
                f"Dropped base architectures "
                f"{[BASE_ARCHITECTURES[a] for a in MODEL_ZOO_ARCHITECTURES]} "
                f"from the design space: tf_model_zoo is not importable. "
                f"Recover it with 'git checkout 20e8a896 -- "
                f"applications/video_configuration/tf_model_zoo'.",
                RuntimeWarning,
                stacklevel=2,
            )

        return spec

    def get_default_design(self) -> Dict[str, Any]:
        """A hand-picked design to seed the search from."""
        space = self.get_structured_design_space()
        design = copy.deepcopy(DEFAULT_DESIGN)

        options = space.spec['base_architecture']['options']
        if design['base_architecture'] not in options:
            design['base_architecture'] = options[0]

        return design

    def get_design_algorithm_config(self) -> Dict[str, Any]:
        """
        GA settings for this application.

        The population is small and mutation light-but-frequent, as for the
        CNN: every fitness evaluation trains a network, so a generation
        already costs a multiple of a timestep. There is no structural
        mutation - all seven genes are scalar.
        """
        return {
            'population_size': 6,
            'tournament_size': 3,
            'crossover_rate': 0.75,
            'mutation_rate': 0.9,
            'gene_mutation_rate': 0.2,
            'structure_mutation_rate': 0.0,
            'elite_size': 1,
            'generations_per_step': 1,
        }

    # ------------------------------------------------------------------
    # Model construction and state transfer
    # ------------------------------------------------------------------

    def _build_model(self, design: Dict[str, Any]) -> TSN:
        """Build the TSN a design describes.

        Args:
            design: The design to build.

        Returns:
            The network, on the configured device.
        """
        model = TSN(
            num_class=self.dataset_info['num_classes'],
            num_segments=int(design['num_segments']),
            modality=self.modality,
            base_model=BASE_ARCHITECTURES[int(design['base_architecture'])],
            consensus_type=CONSENSUS_FUNCTIONS[int(design['consensus_function'])],
            dropout=float(design['dropout']),
            partial_bn=self.partial_bn,
        )
        return model.to(self.device)

    def _transfer_weights(self, source: nn.Module, target: nn.Module) -> float:
        """Copy every parameter of ``source`` that still fits ``target``.

        A design change need not invalidate the whole network. Changing the
        segment count or the consensus function leaves the base model
        untouched; only a change of base architecture replaces it.

        Args:
            source: The network trained so far.
            target: The newly built network.

        Returns:
            The fraction of the target's parameter values that were filled
            from the source.
        """
        source_state = source.state_dict()
        target_state = target.state_dict()

        transferred = 0
        total = 0
        new_state = {}

        for name, tensor in target_state.items():
            total += tensor.numel()
            candidate = source_state.get(name)
            if candidate is not None and candidate.shape == tensor.shape:
                new_state[name] = candidate.clone()
                transferred += tensor.numel()
            else:
                new_state[name] = tensor

        target.load_state_dict(new_state)
        return transferred / total if total else 0.0

    def _apply_design(self, design: Dict[str, Any]) -> float:
        """Make ``design`` the live design, preserving what it does not change.

        Args:
            design: The design to apply.

        Returns:
            The fraction of parameter values carried over.
        """
        if self.model is not None and design == self.current_design:
            self._build_loaders(design, self.model)
            return 1.0

        new_model = self._build_model(design)

        transferred = 0.0
        if self.model is not None:
            transferred = self._transfer_weights(self.model, new_model)
            self.num_design_changes += 1

        self.model = new_model
        self.current_design = copy.deepcopy(design)
        self.last_transfer_fraction = transferred
        self._rebuild_optimizer(design)
        self._build_loaders(design, self.model)

        return transferred

    def _rebuild_optimizer(self, design: Dict[str, Any]) -> None:
        """Create the optimiser, keeping it across timesteps where possible.

        The optimiser holds momentum buffers keyed to parameter tensors, so it
        is rebuilt only when the parameters or the learning rate actually
        change; otherwise momentum would be discarded every timestep.
        """
        signature = (
            int(design['base_architecture']),
            int(design['num_segments']),
            int(design['consensus_function']),
            float(design['learning_rate']),
            float(design['dropout']),
        )
        if self._optimizer_signature == signature and self.optimizer is not None:
            return

        # TSN assigns per-group learning-rate and weight-decay multipliers.
        policies = self.model.get_optim_policies()
        self.optimizer = optim.SGD(
            [
                {
                    'params': group['params'],
                    'lr': float(design['learning_rate']) * group['lr_mult'],
                    'weight_decay': 5e-4 * group['decay_mult'],
                }
                for group in policies
            ],
            lr=float(design['learning_rate']),
            momentum=0.9,
        )
        self._optimizer_signature = signature

    # ------------------------------------------------------------------
    # Loss and prediction
    #
    # The 'identity' consensus returns per-segment scores rather than one
    # score per video, so loss and prediction both have to reduce over the
    # segment axis. Supervising each segment separately (mean of per-segment
    # cross-entropies) is not the same objective as the cross-entropy of the
    # averaged scores, which is what the 'avg' consensus gives - so the two
    # options stay genuinely distinct rather than 'identity' collapsing into
    # a duplicate of 'avg'.
    # ------------------------------------------------------------------

    def _loss_and_logits(
        self,
        output: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the loss and the video-level scores.

        Args:
            output: Model output, (batch, classes) or
                (batch, segments, classes).
            targets: Class indices, (batch,).

        Returns:
            Tuple of (loss, video-level scores).
        """
        if output.dim() == 3:
            batch, segments, classes = output.shape
            per_segment_targets = targets.unsqueeze(1).expand(batch, segments)
            loss = self.criterion(
                output.reshape(batch * segments, classes),
                per_segment_targets.reshape(batch * segments),
            )
            # Prediction averages the per-segment probabilities.
            logits = torch.softmax(output, dim=2).mean(dim=1)
            return loss, logits

        return self.criterion(output, targets), output

    # ------------------------------------------------------------------
    # Timestep execution
    # ------------------------------------------------------------------

    def exec_design(self, design: Dict[str, Any], timestep: int) -> Dict[str, float]:
        """Apply a design for one timestep and measure it.

        This is ``aa.exec(c, dataset, t)``: it advances the persistent network
        by one epoch under ``design``.

        Args:
            design: Design to apply.
            timestep: Current timestep.

        Returns:
            Metrics for this timestep, including 'performance' (validation
            accuracy), 'train_loss', the gradient-norm statistics, and the
            fraction of weights carried over.
        """
        if not self.dataset_info:
            self.load_data()

        start_time = time.time()
        transferred = self._apply_design(design)

        clip_gradient = float(design['gradient_norm_clipping'])

        self.model.train()
        total_loss = 0.0
        num_batches = 0
        grad_norms: List[float] = []
        num_clipped = 0

        for inputs, targets in self.train_loader:
            if (
                self.max_train_batches_per_timestep is not None
                and num_batches >= self.max_train_batches_per_timestep
            ):
                break

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(inputs)
            loss, _ = self._loss_and_logits(output, targets)

            if not torch.isfinite(loss):
                # A diverged design would otherwise poison the weights with
                # NaNs for every later timestep. Skip the update and let the
                # design be judged on the accuracy it produces.
                continue

            loss.backward()

            # clip_grad_norm_ returns the norm *before* clipping, which is
            # what says whether the threshold actually bound.
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), clip_gradient
            )
            total_norm = float(total_norm)
            grad_norms.append(total_norm)
            if total_norm > clip_gradient:
                num_clipped += 1

            self.optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1

        val_accuracy = self._evaluate_loader(
            self.val_loader, max_batches=self.val_batches_per_timestep
        )
        self.is_trained = True
        self.num_timesteps_executed += 1

        metrics = {
            'performance': val_accuracy,
            'val_accuracy': val_accuracy,
            'train_loss': total_loss / num_batches if num_batches else float('nan'),
            'mean_grad_norm': float(np.mean(grad_norms)) if grad_norms else 0.0,
            'max_grad_norm': float(np.max(grad_norms)) if grad_norms else 0.0,
            'fraction_batches_clipped': (
                num_clipped / len(grad_norms) if grad_norms else 0.0
            ),
            'weights_transferred': float(transferred),
            'num_parameters': float(sum(p.numel() for p in self.model.parameters())),
            'num_batches': float(num_batches),
            'timestep_time': time.time() - start_time,
        }
        self._last_metrics = metrics
        return metrics

    # ------------------------------------------------------------------
    # Candidate scoring
    # ------------------------------------------------------------------

    def evaluate_candidate(self, design: Dict[str, Any], timestep: int) -> float:
        """Score one candidate design for the genetic algorithm.

        A candidate is built, warm-started from the live model where the
        shapes allow, trained on a few batches and scored on a few more. The
        live model and optimiser are untouched, so scoring a population does
        not advance the run.

        Args:
            design: Candidate design.
            timestep: Current timestep.

        Returns:
            Probe accuracy, or 0.0 for a design that is too large or cannot
            be trained.
        """
        if not self.dataset_info:
            self.load_data()

        try:
            candidate = self._build_model(design)
        except Exception as exc:
            warnings.warn(
                f"Candidate design {design} could not be built "
                f"({type(exc).__name__}: {exc}); scored as 0.0.",
                RuntimeWarning,
                stacklevel=2,
            )
            return 0.0

        num_parameters = sum(p.numel() for p in candidate.parameters())
        if num_parameters > self.max_candidate_parameters:
            self.num_candidates_rejected += 1
            return 0.0

        if self.warm_start_candidates and self.model is not None:
            self._transfer_weights(self.model, candidate)

        # The candidate needs loaders matching its own segment count and
        # keyframe strategy; building them here leaves the live ones alone.
        saved = (self.train_loader, self.val_loader, self._loader_signature)
        self.train_loader = self.val_loader = None
        self._loader_signature = None

        try:
            self._build_loaders(design, candidate)
            accuracy = self._score_probe(candidate, design)
        except Exception as exc:
            warnings.warn(
                f"Candidate design {design} failed to evaluate "
                f"({type(exc).__name__}: {exc}); scored as 0.0.",
                RuntimeWarning,
                stacklevel=2,
            )
            accuracy = 0.0
        finally:
            self.train_loader, self.val_loader, self._loader_signature = saved

        return accuracy

    def _score_probe(self, candidate: TSN, design: Dict[str, Any]) -> float:
        """Train ``candidate`` on a few batches and score it on a few more."""
        policies = candidate.get_optim_policies()
        optimizer = optim.SGD(
            [
                {
                    'params': group['params'],
                    'lr': float(design['learning_rate']) * group['lr_mult'],
                    'weight_decay': 5e-4 * group['decay_mult'],
                }
                for group in policies
            ],
            lr=float(design['learning_rate']),
            momentum=0.9,
        )
        clip_gradient = float(design['gradient_norm_clipping'])

        candidate.train()
        for index, (inputs, targets) in enumerate(self.train_loader):
            if index >= self.candidate_train_batches:
                break

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            optimizer.zero_grad(set_to_none=True)
            loss, _ = self._loss_and_logits(candidate(inputs), targets)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(candidate.parameters(), clip_gradient)
            optimizer.step()

        return self._evaluate_loader(
            self.val_loader, max_batches=self.candidate_val_batches, model=candidate
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate_loader(
        self,
        data_loader: Optional[DataLoader],
        max_batches: Optional[int] = None,
        model: Optional[nn.Module] = None,
    ) -> float:
        """Accuracy of a model on a data loader.

        Args:
            data_loader: The loader to evaluate on.
            max_batches: Cap on batches, or None for all of them.
            model: The model to use, or None for the live one.

        Returns:
            Accuracy in [0, 1].
        """
        model = model if model is not None else self.model
        if model is None or data_loader is None:
            return 0.0

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for index, (inputs, targets) in enumerate(data_loader):
                if max_batches is not None and index >= max_batches:
                    break

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                _, logits = self._loss_and_logits(model(inputs), targets)
                correct += int((logits.argmax(dim=1) == targets).sum().item())
                total += int(targets.numel())

        return correct / total if total else 0.0

    def evaluate(self, design: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Evaluate on the held-out videos.

        The validation list is also the test list: the datasets this
        application is pointed at have a single held-out split, and inventing
        a third would mean reporting a number measured on fewer videos than
        the run already uses. The reported figure is therefore the full
        held-out accuracy of the network the run produced, with no retraining.

        Args:
            design: Optional design to apply before evaluating.

        Returns:
            Test metrics, including 'test_performance' and 'test_accuracy'.
        """
        if not self.dataset_info:
            self.load_data()

        if design is not None and design != self.current_design:
            self._apply_design(design)

        if self.model is None:
            raise RuntimeError("No model to evaluate; run at least one timestep first")

        test_accuracy = self._evaluate_loader(self.val_loader)

        return {
            'test_performance': test_accuracy,
            'test_accuracy': test_accuracy,
            'num_parameters': float(sum(p.numel() for p in self.model.parameters())),
            'num_val_videos': float(self.dataset_info['num_val_videos']),
        }

    # ------------------------------------------------------------------
    # Static-design path
    # ------------------------------------------------------------------

    def train(self, design: Dict[str, Any], timesteps: Optional[int] = None) -> Dict[str, float]:
        """
        Train a single fixed design for a number of timesteps.

        This is the static-design path, used by the baselines and the test
        script. It resets the training state first, so it is a self-contained
        training run rather than a continuation; the meta-learning approaches
        drive :meth:`exec_design` one timestep at a time instead.

        Args:
            design: Design to train
            timesteps: Number of epochs (default: 100, the thesis budget)

        Returns:
            Dictionary of performance metrics
        """
        if timesteps is None:
            timesteps = self.get_num_timesteps()

        self.reset_run_state()
        if not self.dataset_info:
            self.load_data()

        start_time = time.time()
        accuracies = []
        losses = []

        for epoch in range(timesteps):
            metrics = self.exec_design(design, epoch)
            accuracies.append(metrics['performance'])
            losses.append(metrics['train_loss'])

            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{timesteps}], "
                      f"Loss: {metrics['train_loss']:.4f}, "
                      f"Val Acc: {metrics['performance']:.4f}")

        return {
            'performance': max(accuracies) if accuracies else 0.0,
            'val_accuracy': max(accuracies) if accuracies else 0.0,
            'final_val_accuracy': accuracies[-1] if accuracies else 0.0,
            'final_train_loss': losses[-1] if losses else float('nan'),
            'best_design': copy.deepcopy(design),
            'training_time': time.time() - start_time,
        }

    # ------------------------------------------------------------------
    # Meta-features
    # ------------------------------------------------------------------

    def get_meta_feature_names(self) -> List[str]:
        """The fixed meta-feature schema (see meta_features.META_FEATURE_NAMES)."""
        return get_feature_names()

    def get_meta_features(self) -> Dict[str, float]:
        """Static dataset-level meta-features."""
        if not self.dataset_info:
            self.load_data()
        return {
            'num_train_videos': float(self.dataset_info['num_train_videos']),
            'num_val_videos': float(self.dataset_info['num_val_videos']),
            'num_classes': float(self.dataset_info['num_classes']),
            'frames_per_video': float(self.dataset_info['frames_per_video']),
            'frame_size': float(self.dataset_info['frame_size']),
        }

    def _model_stats(self) -> Dict[str, float]:
        """Parameter counts and feature width of the live network."""
        if self.model is None:
            return {}

        num_parameters = sum(p.numel() for p in self.model.parameters())
        num_trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        feature_dim = (
            float(self.model.new_fc.in_features)
            if getattr(self.model, 'new_fc', None) is not None else 0.0
        )

        return {
            'num_parameters': float(num_parameters),
            'num_trainable_parameters': float(num_trainable),
            'feature_dim': feature_dim,
        }

    def extract_meta_features(self, timestep: int = 0) -> Dict[str, Any]:
        """
        Extract meta-features describing the current state of the run.

        The returned dictionary always carries the same keys in the same order,
        whatever is computable at this timestep, so the vectors stored in the
        knowledge repository are all the same length.

        Args:
            timestep: Current timestep

        Returns:
            Dictionary of meta-features
        """
        if not self.dataset_info:
            self.load_data()

        if self.meta_feature_extractor is None:
            self.meta_feature_extractor = VideoMetaFeatureExtractor(self.dataset_info)

        features = self.meta_feature_extractor.extract_meta_features(
            design=self.current_design,
            model_stats=self._model_stats(),
            timestep_metrics=self._last_metrics,
            timestep=timestep,
        )

        # Record after extracting, so the delta features compare against the
        # previous timestep rather than against themselves.
        if self._last_metrics:
            self.meta_feature_extractor.observe(
                features['train_loss'], features['val_accuracy']
            )

        return features

    # ------------------------------------------------------------------
    # Lifecycle and metadata
    # ------------------------------------------------------------------

    def reset_run_state(self) -> None:
        """Discard the trained state so a new run starts clean."""
        self.model = None
        self.optimizer = None
        self._optimizer_signature = None
        self.current_design = None
        self.meta_feature_extractor = None
        self._last_metrics = {}
        self.is_trained = False
        self.num_design_changes = 0
        self.num_timesteps_executed = 0
        self.num_candidates_rejected = 0
        self.last_transfer_fraction = 0.0

        # Loaders depend on the design, so they are dropped with it.
        self.train_loader = None
        self.val_loader = None
        self._loader_signature = None

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

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
        """Default number of timesteps for this application (thesis: 100)."""
        return 100

    def get_run_statistics(self) -> Dict[str, Any]:
        """Summary of what happened during the current run."""
        return {
            'num_timesteps_executed': self.num_timesteps_executed,
            'num_design_changes': self.num_design_changes,
            'num_candidates_rejected': self.num_candidates_rejected,
            'last_transfer_fraction': self.last_transfer_fraction,
            'excluded_design_options': dict(self.excluded_options),
            'device': str(self.device),
            'architecture': (
                BASE_ARCHITECTURES[int(self.current_design['base_architecture'])]
                if self.current_design else None
            ),
            'consensus': (
                CONSENSUS_FUNCTIONS[int(self.current_design['consensus_function'])]
                if self.current_design else None
            ),
            **self._model_stats(),
        }
