"""
Meta-features for the clustering composition application.

The meta-learners take a meta-feature vector as input (accuracy prediction) or
condition on it (design prediction), and the knowledge repository stores one
per timestep. Both require a *fixed-length* vector: the schema below is
therefore declared up front as :data:`META_FEATURE_NAMES`, and
:meth:`ClusteringMetaFeatureExtractor.extract_meta_features` always returns
exactly those keys in that order, filling in a neutral value for anything not
computable at the current timestep.

Three groups of feature, in the order they appear:

*Dataset* - fixed for a run; they let a repository shared across datasets
distinguish them.

*Clustering state* - the shape of the current partition: how many clusters,
how uneven they are, how many instances ended up in none of them (possible,
since cluster removal can orphan instances) or in several (possible when the
membership-limit gene allows overlapping clusters). This is the composition
the design is actually producing.

*Quality and trajectory* - the internal and external indices for the current
partition, plus how performance and cluster count are moving. OnMAR's decision
is whether the design still works *now*, so the derivatives matter as much as
the levels: a partition that is collapsing cluster-by-cluster looks fine on
level and bad on trend.
"""

from typing import Any, Dict, List, Optional, Sequence
import numpy as np


META_FEATURE_NAMES: List[str] = [
    # Dataset
    'num_instances',
    'num_classes',
    'input_channels',
    'input_height',
    'input_width',
    'total_pixels',
    'feature_dim',
    # Clustering state
    'num_clusters',
    'num_clusters_per_class',
    'mean_cluster_size',
    'std_cluster_size',
    'min_cluster_size',
    'max_cluster_size',
    'cluster_size_imbalance',
    'fraction_unassigned',
    'fraction_multi_assigned',
    'centroid_variance',
    'mean_centroid_norm',
    # Quality and trajectory
    'silhouette',
    'db_index',
    'ch_index',
    'ari',
    'mi',
    'performance',
    'performance_delta',
    'performance_best_so_far',
    'performance_mean',
    'num_clusters_delta',
    'stopped',
    'timestep',
]


def get_feature_names() -> List[str]:
    """The fixed meta-feature schema, in order.

    Returns:
        The meta-feature names.
    """
    return list(META_FEATURE_NAMES)


class ClusteringMetaFeatureExtractor:
    """Builds the fixed-length meta-feature vector for a clustering run.

    One extractor is created per run and kept, because several features
    compare the current timestep against the previous one.
    """

    def __init__(self, dataset_info: Dict[str, Any]):
        """
        Args:
            dataset_info: The dataset's static description, as
                :class:`datasets.image_datasets.ImageDatasetLoader` reports it.
        """
        self.dataset_info = dataset_info
        self.performance_history: List[float] = []
        self.num_clusters_history: List[int] = []

    def reset(self) -> None:
        """Forget the run's history."""
        self.performance_history = []
        self.num_clusters_history = []

    def observe(self, performance: float, num_clusters: int) -> None:
        """Record a timestep's outcome, for the trajectory features.

        Args:
            performance: Clustering accuracy at this timestep.
            num_clusters: Number of clusters after this timestep.
        """
        self.performance_history.append(float(performance))
        self.num_clusters_history.append(int(num_clusters))

    # ------------------------------------------------------------------

    def _cluster_shape_features(
        self,
        clusters: Optional[Sequence[Any]],
        num_instances: int,
    ) -> Dict[str, float]:
        """Features describing the current partition's shape."""
        if not clusters:
            return {
                'num_clusters': 0.0,
                'mean_cluster_size': 0.0,
                'std_cluster_size': 0.0,
                'min_cluster_size': 0.0,
                'max_cluster_size': 0.0,
                'cluster_size_imbalance': 0.0,
                'fraction_unassigned': 1.0,
                'fraction_multi_assigned': 0.0,
                'centroid_variance': 0.0,
                'mean_centroid_norm': 0.0,
            }

        sizes = np.array([len(c.vector_indices) for c in clusters], dtype=float)

        # How many clusters each instance belongs to. Zero is possible after
        # cluster removal; more than one is possible when the membership-limit
        # gene permits overlap.
        membership_counts = np.zeros(max(num_instances, 1), dtype=float)
        for c in clusters:
            for index in c.vector_indices:
                if 0 <= int(index) < len(membership_counts):
                    membership_counts[int(index)] += 1.0

        centroids = [np.asarray(c.centroid, dtype=float).ravel() for c in clusters]
        widths = {centroid.size for centroid in centroids}
        if len(widths) == 1 and centroids[0].size > 0:
            stacked = np.vstack(centroids)
            centroid_variance = float(np.var(stacked))
            mean_centroid_norm = float(np.mean(np.linalg.norm(stacked, axis=1)))
        else:
            # Centroids of differing width cannot be stacked; this happens
            # only transiently, while a design change is being absorbed.
            centroid_variance = 0.0
            mean_centroid_norm = float(
                np.mean([np.linalg.norm(c) for c in centroids]) if centroids else 0.0
            )

        mean_size = float(np.mean(sizes))

        return {
            'num_clusters': float(len(clusters)),
            'mean_cluster_size': mean_size,
            'std_cluster_size': float(np.std(sizes)),
            'min_cluster_size': float(np.min(sizes)),
            'max_cluster_size': float(np.max(sizes)),
            # Coefficient of variation: scale-free, so it is comparable across
            # partitions with very different cluster counts.
            'cluster_size_imbalance': float(np.std(sizes) / mean_size) if mean_size > 0 else 0.0,
            'fraction_unassigned': float(np.mean(membership_counts == 0)),
            'fraction_multi_assigned': float(np.mean(membership_counts > 1)),
            'centroid_variance': centroid_variance,
            'mean_centroid_norm': mean_centroid_norm,
        }

    def _trajectory_features(self, performance: float, num_clusters: int) -> Dict[str, float]:
        """Features comparing this timestep with the run so far."""
        previous_performance = (
            self.performance_history[-1] if self.performance_history else performance
        )
        previous_clusters = (
            self.num_clusters_history[-1] if self.num_clusters_history else num_clusters
        )
        history = self.performance_history + [performance]

        return {
            'performance': float(performance),
            'performance_delta': float(performance - previous_performance),
            'performance_best_so_far': float(max(history)),
            'performance_mean': float(np.mean(history)),
            'num_clusters_delta': float(num_clusters - previous_clusters),
        }

    def extract_meta_features(
        self,
        clusters: Optional[Sequence[Any]],
        metrics: Optional[Dict[str, Any]],
        num_instances: int,
        feature_dim: int,
        timestep: int,
    ) -> Dict[str, float]:
        """Build the meta-feature dictionary for the current timestep.

        Args:
            clusters: The current clusters, or None before the first timestep.
            metrics: The metrics reported by
                :func:`clustering_composition.run.run`, or None before the
                first timestep.
            num_instances: Number of instances being clustered.
            feature_dim: Width of the extracted feature vectors.
            timestep: Current timestep.

        Returns:
            Exactly the keys of :data:`META_FEATURE_NAMES`, in that order.
        """
        metrics = metrics or {}
        channels, height, width = self.dataset_info['input_shape']

        features: Dict[str, float] = {
            'num_instances': float(num_instances),
            'num_classes': float(self.dataset_info['num_classes']),
            'input_channels': float(channels),
            'input_height': float(height),
            'input_width': float(width),
            'total_pixels': float(channels * height * width),
            'feature_dim': float(feature_dim),
        }

        features.update(self._cluster_shape_features(clusters, num_instances))

        performance = float(metrics.get('fitness', 0.0))
        num_clusters = int(features['num_clusters'])

        features.update({
            'silhouette': float(metrics.get('silhouette', 0.0)),
            'db_index': float(metrics.get('db_index', 0.0)),
            'ch_index': float(metrics.get('ch_index', 0.0)),
            'ari': float(metrics.get('ari', 0.0)),
            'mi': float(metrics.get('mi', 0.0)),
            'stopped': 1.0 if metrics.get('stopped') else 0.0,
            'timestep': float(timestep),
        })
        features.update(self._trajectory_features(performance, num_clusters))

        # Normalised cluster count: the interesting quantity is clusters
        # relative to classes, not the raw count, so one repository can hold
        # runs on datasets with 10 and 100 classes.
        num_classes = features['num_classes']
        features['num_clusters_per_class'] = (
            float(num_clusters) / num_classes if num_classes > 0 else 0.0
        )

        # Return in the declared order, so the vector layout is stable.
        return {name: float(features.get(name, 0.0)) for name in META_FEATURE_NAMES}
