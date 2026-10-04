"""
Meta-features for the video classification configuration application.

As for the other applications, the schema is declared up front as
:data:`META_FEATURE_NAMES` and
:meth:`VideoMetaFeatureExtractor.extract_meta_features` always returns exactly
those keys in that order. The meta-learners cannot be fitted on ragged input,
and the knowledge repository stores one vector per timestep.

Three groups, in order:

*Dataset* - fixed for a run: how many videos and classes, how long the videos
are, how large the frames are.

*Model* - the shape of the network the current design produced: parameter
count, feature width, segment count, dropout, and how much of the network is
frozen (TSN's partial batch-norm freezing means a sizeable fraction of a
run's parameters may not be training).

*Training trajectory* - loss and accuracy now, and how they are moving. The
gradient-norm features are the ones specific to this application: the video
design space includes a gradient-norm clipping threshold, and whether that
threshold is actually binding is not something any other feature reveals. A
design whose clipping never fires and one whose clipping fires on every batch
behave entirely differently while looking identical in their hyperparameters.
"""

from typing import Any, Dict, List, Optional
import numpy as np


META_FEATURE_NAMES: List[str] = [
    # Dataset
    'num_train_videos',
    'num_val_videos',
    'num_classes',
    'frames_per_video',
    'frame_size',
    # Model
    'num_parameters',
    'num_trainable_parameters',
    'trainable_fraction',
    'feature_dim',
    'num_segments',
    'dropout',
    'learning_rate',
    'clip_gradient',
    # Training trajectory
    'train_loss',
    'train_loss_delta',
    'val_accuracy',
    'val_accuracy_delta',
    'val_accuracy_best_so_far',
    'val_accuracy_mean',
    'mean_grad_norm',
    'max_grad_norm',
    'fraction_batches_clipped',
    'weights_transferred',
    'timestep',
]


def get_feature_names() -> List[str]:
    """The fixed meta-feature schema, in order.

    Returns:
        The meta-feature names.
    """
    return list(META_FEATURE_NAMES)


class VideoMetaFeatureExtractor:
    """Builds the fixed-length meta-feature vector for a video run.

    One extractor is created per run and kept, because the trajectory
    features compare the current timestep against the previous one.
    """

    def __init__(self, dataset_info: Dict[str, Any]):
        """
        Args:
            dataset_info: The dataset's static description, as
                :class:`applications.configuration.video.video_application`
                builds it.
        """
        self.dataset_info = dataset_info
        self.loss_history: List[float] = []
        self.accuracy_history: List[float] = []

    def reset(self) -> None:
        """Forget the run's history."""
        self.loss_history = []
        self.accuracy_history = []

    def observe(self, train_loss: float, val_accuracy: float) -> None:
        """Record a timestep's outcome, for the trajectory features.

        Args:
            train_loss: Mean training loss over the timestep.
            val_accuracy: Validation accuracy after the timestep.
        """
        self.loss_history.append(float(train_loss))
        self.accuracy_history.append(float(val_accuracy))

    def extract_meta_features(
        self,
        design: Optional[Dict[str, Any]],
        model_stats: Optional[Dict[str, float]],
        timestep_metrics: Optional[Dict[str, float]],
        timestep: int,
    ) -> Dict[str, float]:
        """Build the meta-feature dictionary for the current timestep.

        Args:
            design: The design in force, or None before the first timestep.
            model_stats: Parameter counts and feature width of the live
                network, or None before it is built.
            timestep_metrics: Metrics from the last ``exec_design``, or None
                before the first timestep.
            timestep: Current timestep.

        Returns:
            Exactly the keys of :data:`META_FEATURE_NAMES`, in that order.
        """
        design = design or {}
        model_stats = model_stats or {}
        metrics = timestep_metrics or {}

        num_parameters = float(model_stats.get('num_parameters', 0.0))
        num_trainable = float(model_stats.get('num_trainable_parameters', 0.0))

        train_loss = float(metrics.get('train_loss', 0.0))
        if not np.isfinite(train_loss):
            # A diverged timestep reports NaN; the repository must stay
            # numeric, and a large finite loss carries the same information.
            train_loss = 1e3
        val_accuracy = float(metrics.get('performance', 0.0))

        previous_loss = self.loss_history[-1] if self.loss_history else train_loss
        previous_accuracy = (
            self.accuracy_history[-1] if self.accuracy_history else val_accuracy
        )
        accuracy_history = self.accuracy_history + [val_accuracy]

        features: Dict[str, float] = {
            # Dataset
            'num_train_videos': float(self.dataset_info.get('num_train_videos', 0)),
            'num_val_videos': float(self.dataset_info.get('num_val_videos', 0)),
            'num_classes': float(self.dataset_info.get('num_classes', 0)),
            'frames_per_video': float(self.dataset_info.get('frames_per_video', 0)),
            'frame_size': float(self.dataset_info.get('frame_size', 0)),
            # Model
            'num_parameters': num_parameters,
            'num_trainable_parameters': num_trainable,
            'trainable_fraction': (
                num_trainable / num_parameters if num_parameters > 0 else 0.0
            ),
            'feature_dim': float(model_stats.get('feature_dim', 0.0)),
            'num_segments': float(design.get('num_segments', 0)),
            'dropout': float(design.get('dropout', 0.0)),
            'learning_rate': float(design.get('learning_rate', 0.0)),
            'clip_gradient': float(design.get('gradient_norm_clipping', 0.0)),
            # Trajectory
            'train_loss': train_loss,
            'train_loss_delta': train_loss - previous_loss,
            'val_accuracy': val_accuracy,
            'val_accuracy_delta': val_accuracy - previous_accuracy,
            'val_accuracy_best_so_far': float(max(accuracy_history)),
            'val_accuracy_mean': float(np.mean(accuracy_history)),
            'mean_grad_norm': float(metrics.get('mean_grad_norm', 0.0)),
            'max_grad_norm': float(metrics.get('max_grad_norm', 0.0)),
            'fraction_batches_clipped': float(metrics.get('fraction_batches_clipped', 0.0)),
            'weights_transferred': float(metrics.get('weights_transferred', 0.0)),
            'timestep': float(timestep),
        }

        return {name: float(features.get(name, 0.0)) for name in META_FEATURE_NAMES}
