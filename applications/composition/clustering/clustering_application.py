"""
Clustering Composition Application

Automated composition of a clustering algorithm from interchangeable
components, with the eleven-option design space of the thesis's clustering
chromosome (``reference/thesis_chromosomes.py``).

This is a *composition* application rather than a configuration one: a design
does not set the hyperparameters of one fixed algorithm, it selects which
components the algorithm is built from - how clusters are initialised, which
single composition operator runs this timestep, how distance is measured, how
features are extracted, when to stop.

Timesteps and clustering state
------------------------------
Algorithm 9 separates choosing a design from applying it:

    c <- design_algorithm(dataset, t)      choose a design for timestep t
    p <- aa.exec(c, dataset, t)            apply it for one timestep

:meth:`ClusteringCompositionApplication.exec_design` implements the second
line. It advances *one* clustering by a single composition step, carrying the
clusters across timesteps in the state list that
:func:`clustering_composition.run.run` maintains. Re-clustering from scratch
each timestep would make every timestep independent and the performance trace
meaningless - which is the same reason the CNN application carries its network
weights across timesteps.

Design changes and the feature space
------------------------------------
A design change mid-run can change the ``feature_extraction`` gene, and then
the clusters carried over from the previous timestep describe points in a
feature space that no longer exists: their centroids have the old
dimensionality and their stored vectors the old coordinates.

:meth:`_reproject_clusters` is the analogue of the CNN application's
``_transfer_weights``. What a cluster fundamentally *is* here is a set of
member instances, and membership is a property of the data, not of the
representation - so membership is kept and the vectors and centroids are
recomputed in the new feature space. A change of representation therefore
costs the run its centroid geometry but not the composition it has built up,
in the same way that a change of architecture costs the CNN only the
parameters whose shapes actually changed.

Design-space availability
-------------------------
Several options need libraries outside ``requirements-min.txt`` (see
:mod:`clustering_composition.optional_deps`). Rather than let the genetic
algorithm sample designs that raise, options whose dependencies are missing
are dropped from the design space at construction and reported in
:attr:`excluded_options`. Pass ``strict_design_space=True`` to demand the full
space and fail loudly instead.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from typing import Any, Dict, List, Optional, Sequence, Tuple
import copy
import time
import warnings
import numpy as np

from applications.base_application import BaseApplication
from datasets.image_datasets import ImageDatasetLoader
from applications.composition.clustering.meta_features import (
    ClusteringMetaFeatureExtractor, get_feature_names,
)
from clustering_composition import optional_deps
from clustering_composition.run import (
    run as run_clustering_timestep, as_matrix,
)
from clustering_composition.utils import (
    get_feature_extraction,
    convert_clusters_to_label_representations,
    infer_cluster_labels,
    infer_data_labels,
    get_distance_metric,
)
from clustering_composition.cluster import cluster as Cluster
from clustering_composition import label_utils


# The eleven genes of the clustering chromosome, in the order
# clustering_composition.run.run indexes them. This list is the single
# definition of that order; designs are dictionaries and are converted to a
# positional chromosome only at the boundary.
CHROMOSOME_ORDER: List[str] = [
    'distance_metric',           # 0
    'cluster_initialization',    # 1
    'cluster_creation',          # 2
    'cluster_addition',          # 3
    'cluster_removal',           # 4
    'cluster_merging',           # 5
    'cluster_splitting',         # 6
    'membership_limit',          # 7
    'stopping_criteria',         # 8
    'step',                      # 9
    'feature_extraction',        # 10
]


# The option values of the thesis chromosome. The recovered components
# implement more options than the thesis declares for three of these genes
# (cluster_creation has 8 implemented, cluster_splitting 16, feature
# extraction 10 including a raw-pixel option), and the extra ones are left out
# so this space is the thesis's. They remain reachable through
# ``extra_options``.
THESIS_OPTIONS: Dict[str, List[int]] = {
    'distance_metric': list(range(8)),
    'cluster_initialization': list(range(8)),
    'cluster_creation': list(range(7)),
    'cluster_addition': list(range(5)),
    'cluster_removal': list(range(6)),
    'cluster_merging': list(range(4)),
    'cluster_splitting': list(range(7)),
    'membership_limit': [0, 1],
    'stopping_criteria': [0, 1],
    'step': list(range(5)),
    'feature_extraction': list(range(9)),
}


# Which options need an optional library, and which one.
OPTION_REQUIREMENTS: Dict[Tuple[str, int], str] = {
    ('cluster_creation', 2): 'kneed',        # DBSCAN, knee-located eps
    ('cluster_creation', 4): 'kmedoids',     # k-medoids (fasterpam)
    ('cluster_addition', 4): 'markov',       # Markov clustering
    ('feature_extraction', 0): 'tensorflow',  # ResNet50
    ('feature_extraction', 1): 'tensorflow',  # InceptionV3
    ('feature_extraction', 2): 'tensorflow',  # DenseNet121
    ('feature_extraction', 3): 'tensorflow',  # Xception
    ('feature_extraction', 4): 'tensorflow',  # VGG16
    ('feature_extraction', 6): 'opencv',      # SIFT
    ('feature_extraction', 8): 'opencv_contrib',  # BRIEF (cv2.xfeatures2d)
}


# A hand-picked design to seed the search from: k-means++ initialisation with
# centroid assignment, mini-batch k-means creation as the timestep's operator,
# single membership, Euclidean distance, and PCA features.
#
# PCA is the seed's feature extractor rather than one of the pretrained CNNs
# because it needs no download and is roughly three orders of magnitude
# cheaper, and the design algorithm gets only theta_t calls in a run - a seed
# that takes seconds per evaluation spends the whole budget on one generation.
# The genetic algorithm is free to evolve towards the CNN extractors, and does.
DEFAULT_DESIGN: Dict[str, int] = {
    'distance_metric': 0,          # Euclidean
    'cluster_initialization': 2,   # k-means++ seeds, then centroid assignment
    'cluster_creation': 0,         # mini-batch k-means
    'cluster_addition': 0,         # assign by nearest centroid
    'cluster_removal': 0,          # none
    'cluster_merging': 0,          # none
    'cluster_splitting': 0,        # none
    'membership_limit': 0,         # one cluster per instance
    'stopping_criteria': 0,        # convergence of assignments
    'step': 0,                     # the timestep's operator is creation
    'feature_extraction': 5,       # PCA
}


class ClusteringCompositionApplication(BaseApplication):
    """
    Clustering composition application for automated algorithm composition.
    """

    def __init__(
        self,
        dataset_name: str,
        random_seed: int = 42,
        num_instances: int = 128,
        num_test_instances: int = 128,
        strict_design_space: bool = False,
        extra_options: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize the clustering composition application.

        Args:
            dataset_name: Name of the dataset (mnist, fashion-mnist,
                cifar-10, cifar-100)
            random_seed: Random seed for reproducibility
            num_instances: Number of instances to cluster. A clustering is
                built over one fixed sample and carried across timesteps, so
                this is the size of the problem, not a batch size. The cost of
                a timestep grows super-linearly in it for several components
                (spectral clustering builds an affinity matrix), so it is
                deliberately small by default.
            num_test_instances: Held-out instances used by :meth:`evaluate`.
            strict_design_space: Demand the full thesis design space, raising
                if an option's dependencies are missing, rather than dropping
                the unavailable options.
            extra_options: Include the options the recovered components
                implement beyond the thesis chromosome.
            verbose: Print a metric line per timestep, as the original
                ``run.py`` did.
        """
        super().__init__(dataset_name, random_seed)

        self.num_instances = num_instances
        self.num_test_instances = num_test_instances
        self.strict_design_space = strict_design_space
        self.extra_options = extra_options
        self.verbose = verbose

        self.dataset_loader = ImageDatasetLoader(dataset_name, random_seed=random_seed)
        self.dataset_info = self.dataset_loader.get_dataset_info()

        # The fixed instance sample (populated by load_data).
        self.train_images: Optional[np.ndarray] = None
        self.train_labels: Optional[np.ndarray] = None
        self.test_images: Optional[np.ndarray] = None
        self.test_labels: Optional[np.ndarray] = None

        # Per-run clustering state, carried across timesteps.
        self.state: Optional[List[Dict[str, Any]]] = None
        self.current_design: Optional[Dict[str, Any]] = None
        self.meta_feature_extractor: Optional[ClusteringMetaFeatureExtractor] = None

        # Feature extraction depends only on the extractor gene and the fixed
        # image sample, so it is computed once per extractor and reused by
        # every timestep and every candidate evaluation.
        self._feature_cache: Dict[int, np.ndarray] = {}
        self._test_feature_cache: Dict[int, np.ndarray] = {}

        # Which options were dropped for want of a dependency.
        self.excluded_options: Dict[str, List[int]] = {}

        # Run statistics
        self.num_design_changes = 0
        self.num_timesteps_executed = 0
        self.num_candidates_failed = 0
        self.num_reprojections = 0

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def load_data(self) -> None:
        """Load the fixed instance sample this run clusters."""
        if self.train_images is not None:
            return  # already loaded; a run must not resample between timesteps

        (
            self.train_images,
            self.train_labels,
            self.test_images,
            self.test_labels,
        ) = self.dataset_loader.load_arrays(
            num_train=self.num_instances,
            num_test=self.num_test_instances,
        )

        print(f"Loaded {self.dataset_name}: "
              f"clustering {len(self.train_images)} instances "
              f"({len(set(self.train_labels.tolist()))} classes present), "
              f"{len(self.test_images)} held out for test")

    def _as_dataloader(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """The instance sample in the ``(images, labels)`` form ``run`` wants."""
        return [(self.train_images, self.train_labels)]

    @staticmethod
    def _channels_last(images: np.ndarray) -> np.ndarray:
        """Lay images out as ``run`` does: (N, C, H, W) -> (N, W, H, C)."""
        return images.reshape(
            (images.shape[0], images.shape[-1], images.shape[-2], images.shape[-3])
        )

    def _ensure_features(self, extractor: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Features for the clustered and held-out instances, in one space.

        Both sets are feature-extracted *together* and then split, rather than
        extracted separately. Two of the extractors make this necessary.

        PCA and t-SNE are fitted on whatever batch they are handed, and the
        recovered implementations return only the transformed values - no
        fitted transform that could be applied to new data. Extracted
        separately, the clustered and held-out sets therefore land in
        different embeddings, and not even of the same width: PCA yields
        ``min(n_samples, n_features)`` components, so 48 clustered instances
        gave 48 columns and 64 held-out instances gave 64, and comparing a
        held-out instance against a cluster centroid failed outright with
        "XA and XB must have the same number of columns". Even had the widths
        matched, the two bases would have been unrelated.

        Extracting jointly puts every instance in one embedding. For the
        pretrained CNNs, SIFT and BRIEF this changes nothing, since those are
        per-image. For PCA and t-SNE it makes the embedding transductive: the
        basis is fitted over the held-out *inputs* as well as the clustered
        ones. That is a real property of the result and worth stating when
        reporting it - though t-SNE has no out-of-sample extension at all, so
        for that extractor transduction is the only thing available. No
        held-out *label* is used at any point.

        Args:
            extractor: The ``feature_extraction`` gene value.

        Returns:
            Tuple of (clustered features, held-out features). The second is
            None when no instances are held out.
        """
        if extractor in self._feature_cache:
            return self._feature_cache[extractor], self._test_feature_cache.get(extractor)

        train_images = self._channels_last(self.train_images)
        has_test = self.test_images is not None and len(self.test_images) > 0

        if has_test:
            combined = np.concatenate(
                [train_images, self._channels_last(self.test_images)], axis=0
            )
        else:
            combined = train_images

        features = np.asarray(get_feature_extraction(extractor, combined))

        num_train = len(train_images)
        self._feature_cache[extractor] = features[:num_train]
        if has_test:
            self._test_feature_cache[extractor] = features[num_train:]

        return self._feature_cache[extractor], self._test_feature_cache.get(extractor)

    # ------------------------------------------------------------------
    # Design space
    # ------------------------------------------------------------------

    def _option_values(self, gene: str) -> List[int]:
        """The available values for one gene, after availability filtering."""
        values = list(THESIS_OPTIONS[gene])

        if self.extra_options:
            # The components implement options past the thesis's ranges; see
            # clustering_composition.utils for the full dispatch tables.
            implemented = {
                'cluster_creation': list(range(8)),
                'cluster_splitting': list(range(16)),
                'feature_extraction': list(range(10)),
            }
            values = implemented.get(gene, values)

        available = []
        excluded = []
        for value in values:
            requirement = OPTION_REQUIREMENTS.get((gene, value))
            if requirement is None or optional_deps.available(requirement):
                available.append(value)
            else:
                excluded.append(value)

        if excluded:
            if self.strict_design_space:
                optional_deps.require(
                    OPTION_REQUIREMENTS[(gene, excluded[0])],
                    f"{gene}={excluded[0]}",
                )
            self.excluded_options[gene] = excluded

        if not available:
            raise RuntimeError(
                f"Every option for '{gene}' needs a library that is not "
                f"installed. Install them with "
                f"'pip install -r requirements-clustering.txt'."
            )

        return available

    def get_design_space(self) -> Dict[str, Any]:
        """
        Get the design space for clustering composition.

        Returns the eleven options of the thesis clustering chromosome as
        categorical genes, with options whose dependencies are unavailable
        dropped (see :attr:`excluded_options`).
        """
        self.excluded_options = {}
        spec = {
            gene: {'type': 'categorical', 'options': self._option_values(gene)}
            for gene in CHROMOSOME_ORDER
        }

        if self.excluded_options:
            report = ', '.join(
                f"{gene}={values}" for gene, values in sorted(self.excluded_options.items())
            )
            warnings.warn(
                f"Dropped design options whose dependencies are missing: {report}. "
                f"Missing: {optional_deps.unavailable_report()}. "
                f"Install with 'pip install -r requirements-clustering.txt', "
                f"or pass strict_design_space=True to fail instead.",
                RuntimeWarning,
                stacklevel=2,
            )

        return spec

    def get_default_design(self) -> Dict[str, Any]:
        """A hand-picked design to seed the search from.

        Any gene value that is unavailable in this environment falls back to
        the first available value for that gene, so the seed is always a design
        that can actually be run.
        """
        space = self.get_structured_design_space()
        design = dict(DEFAULT_DESIGN)

        for gene, value in design.items():
            options = space.spec[gene]['options']
            if value not in options:
                design[gene] = options[0]

        return design

    def get_design_algorithm_config(self) -> Dict[str, Any]:
        """
        GA settings for this application.

        The population is small for the same reason as the CNN's: every
        fitness evaluation applies a real composition step over all the
        instances, so one generation already costs several timesteps. There is
        no structural mutation because this design space has no variable-length
        part - all eleven genes are scalar.
        """
        return {
            'population_size': 8,
            'tournament_size': 3,
            'crossover_rate': 0.75,
            'mutation_rate': 0.9,
            # Higher than the CNN's per-gene rate: with eleven genes and no
            # block lists, 0.12 would change barely one gene per mutated
            # individual.
            'gene_mutation_rate': 0.2,
            'structure_mutation_rate': 0.0,
            'elite_size': 1,
            'generations_per_step': 1,
        }

    # ------------------------------------------------------------------
    # Design <-> chromosome
    # ------------------------------------------------------------------

    def design_to_chromosome(self, design: Dict[str, Any]) -> List[int]:
        """Convert a design dictionary to the positional chromosome.

        Args:
            design: Design keyed by gene name.

        Returns:
            The eleven gene values, in chromosome order.
        """
        return [int(design[gene]) for gene in CHROMOSOME_ORDER]

    def chromosome_to_design(self, chromosome: Sequence[int]) -> Dict[str, int]:
        """Convert a positional chromosome to a design dictionary.

        Args:
            chromosome: The eleven gene values, in chromosome order.

        Returns:
            The design, keyed by gene name.
        """
        return {gene: int(value) for gene, value in zip(CHROMOSOME_ORDER, chromosome)}

    # ------------------------------------------------------------------
    # Carrying clusters across a change of feature space
    # ------------------------------------------------------------------

    def _reproject_clusters(
        self,
        clusters: Optional[Sequence[Any]],
        features: np.ndarray,
    ) -> List[Any]:
        """Re-express clusters in a new feature space, keeping their membership.

        A cluster's membership (``vector_indices``) indexes the instance
        sample and so is independent of how those instances are represented.
        Its ``vectors`` and ``centroid`` are not. When the design's feature
        extractor changes, this keeps the former and recomputes the latter, so
        the composition the run has built survives the change of
        representation.

        Args:
            clusters: Clusters expressed in the previous feature space.
            features: The instance sample in the new feature space.

        Returns:
            The clusters, re-expressed. Clusters whose membership is empty are
            dropped - they carry no information into the new space, and a
            zero-member cluster makes centroid computation a division by zero.
        """
        if not clusters:
            return []

        reprojected = []
        for source in clusters:
            indices = [
                int(index) for index in source.vector_indices
                if 0 <= int(index) < len(features)
            ]
            if not indices:
                continue

            reprojected.append(
                Cluster(
                    identifier=source.identifier,
                    vectors=[features[index] for index in indices],
                    vector_indices=indices,
                )
            )

        self.num_reprojections += 1
        return reprojected

    def _features_for(self, design: Dict[str, Any]) -> np.ndarray:
        """The instance sample's features under ``design``'s extractor."""
        features, _ = self._ensure_features(int(design['feature_extraction']))
        return features

    def _prepare_state_for(self, design: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """The run state to apply ``design`` to, reprojected if it must be.

        Returns:
            The run's state list, or None if no timestep has run yet.
        """
        if not self.state:
            return None

        previous = self.current_design
        changed_extractor = (
            previous is not None
            and int(previous['feature_extraction']) != int(design['feature_extraction'])
        )

        if changed_extractor:
            features = self._features_for(design)
            self.state[-1]['clusters'] = self._reproject_clusters(
                self.state[-1]['clusters'], features
            )
            self.state[-1]['data'] = features

        return self.state

    # ------------------------------------------------------------------
    # Timestep execution
    # ------------------------------------------------------------------

    def exec_design(self, design: Dict[str, Any], timestep: int) -> Dict[str, float]:
        """Apply a design for one timestep and measure it.

        This is ``aa.exec(c, dataset, t)``: it advances the run's single
        clustering by one composition step under ``design``.

        Args:
            design: Design to apply.
            timestep: Current timestep.

        Returns:
            Metrics for this timestep, including 'performance' (clustering
            accuracy), the internal and external cluster-quality indices, and
            the number of clusters.
        """
        if self.train_images is None:
            self.load_data()

        start_time = time.time()

        if self.current_design is not None and design != self.current_design:
            self.num_design_changes += 1

        # Fill the cache first, so run() finds the jointly-extracted
        # features rather than extracting the clustered set on its own.
        self._ensure_features(int(design['feature_extraction']))

        state = self._prepare_state_for(design)
        chromosome = self.design_to_chromosome(design)

        performance, self.state = run_clustering_timestep(
            chromosome,
            self._as_dataloader(),
            timestep,
            state=state,
            feature_cache=self._feature_cache,
            verbose=self.verbose,
        )

        self.current_design = copy.deepcopy(design)
        self.is_trained = True
        self.num_timesteps_executed += 1

        metrics = dict(self.state[-1]['metrics'])
        metrics['performance'] = float(performance)
        metrics['timestep_time'] = time.time() - start_time

        return metrics

    # ------------------------------------------------------------------
    # Candidate scoring
    # ------------------------------------------------------------------

    def evaluate_candidate(self, design: Dict[str, Any], timestep: int) -> float:
        """Score one candidate design for the genetic algorithm.

        The candidate is applied to a *copy* of the run's clustering, so
        scoring a population does not advance or corrupt the live state. Only
        the cluster objects are copied; the images, labels and cached features
        are shared, since the composition operators never mutate them.

        Args:
            design: Candidate design.
            timestep: Current timestep.

        Returns:
            Clustering accuracy after applying the candidate for one timestep,
            or -1.0 for a design that fails. -1.0 is the value the components
            themselves report for a degenerate partition, so a failing design
            and a useless one are scored alike rather than a failure being
            rewarded.
        """
        if self.train_images is None:
            self.load_data()

        chromosome = self.design_to_chromosome(design)
        self._ensure_features(int(design['feature_extraction']))

        probe_state: Optional[List[Dict[str, Any]]] = None
        if self.state:
            live = self.state[-1]
            clusters = live['clusters']

            # If this candidate extracts features differently from the design
            # that produced the live clusters, it must see them reprojected -
            # exactly as exec_design would do were the candidate adopted.
            if int(self.current_design['feature_extraction']) != int(design['feature_extraction']):
                features, _ = self._ensure_features(int(design['feature_extraction']))
                clusters = self._reproject_clusters(clusters, features)
            else:
                features = live['data']

            probe_state = [{
                'clusters': copy.deepcopy(clusters),
                'images': live['images'],
                'labels': live['labels'],
                'data': features,
                'metrics': live.get('metrics', {}),
            }]

        try:
            performance, _ = run_clustering_timestep(
                chromosome,
                self._as_dataloader(),
                timestep,
                state=probe_state,
                feature_cache=self._feature_cache,
                verbose=False,
            )
        except Exception as exc:
            # A component can fail on a particular partition - splitting a
            # one-member cluster, say. The genetic algorithm must keep going,
            # so the design is scored as useless rather than killing the run.
            self.num_candidates_failed += 1
            warnings.warn(
                f"Candidate design {design} failed to evaluate "
                f"({type(exc).__name__}: {exc}); scored as -1.0.",
                RuntimeWarning,
                stacklevel=2,
            )
            return -1.0

        return float(performance)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _assign_held_out(
        self,
        design: Dict[str, Any],
        distance_metric: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Assign the held-out instances to the run's clusters.

        The clusters are built over the training sample, so measuring on
        held-out data needs two things that are taken from the training sample
        only: the cluster centroids, and the cluster-to-class mapping. Each
        held-out instance then goes to its nearest centroid and inherits that
        cluster's class.

        Inferring the mapping on the training sample rather than re-inferring
        it from the held-out labels is the stricter choice - re-inferring would
        let the test labels pick the most favourable mapping, which is
        acceptable for the unsupervised score the components report on their
        own clustering but not for a held-out one.

        Args:
            design: The design whose metric and genes to use.
            distance_metric: Override the design's distance-metric gene. Used
                to report the Euclidean diagnostic alongside the design's own
                number; see :meth:`evaluate`.

        Returns:
            Tuple of (predicted_labels, true_labels) for the held-out set.
        """
        clusters = self.state[-1]['clusters']

        # Both halves of the shared extraction, so the held-out features and
        # the centroids live in the same space (see _ensure_features).
        train_features, test_features = self._ensure_features(
            int(design['feature_extraction'])
        )

        # The training sample's cluster assignment, as contiguous integers,
        # and the class each cluster stands for.
        train_assignment = convert_clusters_to_label_representations(
            train_features, clusters, design['membership_limit'], design['distance_metric']
        )
        train_assignment = np.asarray(train_assignment).flatten()
        identifiers = list(dict.fromkeys(train_assignment.tolist()))
        train_indexed = np.array([identifiers.index(a) for a in train_assignment.tolist()])
        cluster_to_class = infer_cluster_labels(
            train_indexed, np.asarray(self.train_labels).flatten().astype(np.int64)
        )

        # Nearest centroid, in the metric the design specifies.
        from scipy.spatial.distance import cdist
        metric_gene = (
            design['distance_metric'] if distance_metric is None else distance_metric
        )
        metric = get_distance_metric(metric_gene).replace('manhattan', 'cityblock')
        centroids = np.vstack([
            np.asarray(c.centroid, dtype=float).ravel() for c in clusters
        ])
        flat_test = as_matrix(test_features)
        distances = cdist(flat_test, centroids, metric=metric)
        nearest = np.argmin(distances, axis=1)

        # Map the chosen cluster back onto the contiguous index the mapping is
        # keyed by, then onto a class.
        cluster_identifiers = [c.identifier for c in clusters]
        nearest_indexed = np.array([
            identifiers.index(cluster_identifiers[n])
            if cluster_identifiers[n] in identifiers else -1
            for n in nearest
        ])
        predicted = infer_data_labels(nearest_indexed, cluster_to_class)

        return predicted, np.asarray(self.test_labels).flatten().astype(np.int64)

    def evaluate(self, design: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Evaluate the run's clustering on the held-out instances.

        Args:
            design: Optional design to apply before evaluating. Applying it
                reprojects the existing clusters where the feature space
                changes; it does not re-cluster from scratch, so the number
                reported is the one the run actually earned.

        Returns:
            Test metrics, including 'test_performance' and 'test_accuracy'.
        """
        if self.train_images is None:
            self.load_data()

        if not self.state:
            raise RuntimeError(
                "No clustering to evaluate; run at least one timestep first"
            )

        design = design or self.current_design
        if design != self.current_design:
            self._prepare_state_for(design)

        train_metrics = self.state[-1]['metrics']
        clusters = self.state[-1]['clusters']

        if not clusters or self.test_images is None or len(self.test_images) == 0:
            # Nothing to assign held-out instances to; report the training
            # sample's score and say so, rather than inventing a test number.
            return {
                'test_performance': float(train_metrics.get('fitness', 0.0)),
                'test_accuracy': float(train_metrics.get('fitness', 0.0)),
                'held_out_evaluated': 0.0,
                'num_clusters': float(len(clusters)),
                'silhouette': float(train_metrics.get('silhouette', 0.0)),
                'ari': float(train_metrics.get('ari', 0.0)),
            }

        from sklearn.metrics import accuracy_score, adjusted_rand_score
        predicted, true_labels = self._assign_held_out(design)
        test_accuracy = float(accuracy_score(true_labels, predicted))

        # The same assignment under Euclidean distance, as a diagnostic.
        #
        # This is reported because the gap between the two is large and
        # systematic, and without it a held-out score far below the training
        # score looks like a bug. A cluster's centroid is an arithmetic mean
        # (``cluster.set_centroid``), which is the Euclidean representative
        # point - but several creation components (mini-batch k-means, mean
        # shift) are Euclidean-only in scikit-learn and ignore the
        # distance-metric gene when forming clusters. Training accuracy is
        # computed from stored membership and so never consults the metric
        # either, which means the gene can be left at a value that is
        # incoherent with the geometry the clusters were built in, and the
        # held-out assignment is the first place that shows. Measured on
        # MNIST at 128 instances with the default design, training accuracy
        # was 0.6406 for *every* metric while held-out accuracy ran from
        # 0.5703 (Euclidean) down to 0.0938 (Canberra).
        #
        # Consequence worth keeping in mind: the design algorithm optimises
        # training accuracy, so it gets no signal about this gene at all and
        # leaves it to drift.
        euclidean_predicted, _ = self._assign_held_out(design, distance_metric=0)
        euclidean_accuracy = float(accuracy_score(true_labels, euclidean_predicted))

        return {
            'test_performance': test_accuracy,
            'test_accuracy': test_accuracy,
            'test_ari': float(adjusted_rand_score(true_labels, predicted)),
            'test_accuracy_euclidean': euclidean_accuracy,
            'held_out_evaluated': float(len(true_labels)),
            'train_performance': float(train_metrics.get('fitness', 0.0)),
            'num_clusters': float(len(clusters)),
            'silhouette': float(train_metrics.get('silhouette', 0.0)),
            'ari': float(train_metrics.get('ari', 0.0)),
        }

    # ------------------------------------------------------------------
    # Static-design path
    # ------------------------------------------------------------------

    def train(self, design: Dict[str, Any], timesteps: Optional[int] = None) -> Dict[str, float]:
        """
        Apply a single fixed design for a number of timesteps.

        This is the static-design path, used by the baselines and the test
        script. It resets the clustering state first, so it is a self-contained
        run rather than a continuation; the meta-learning approaches drive
        :meth:`exec_design` one timestep at a time instead.

        Args:
            design: Design to apply
            timesteps: Number of timesteps (default: 60, the thesis budget)

        Returns:
            Dictionary of performance metrics
        """
        if timesteps is None:
            timesteps = self.get_num_timesteps()

        self.reset_run_state()
        if self.train_images is None:
            self.load_data()

        start_time = time.time()
        performances = []

        for timestep in range(timesteps):
            metrics = self.exec_design(design, timestep)
            performances.append(metrics['performance'])

            if (timestep + 1) % 10 == 0:
                print(f"Timestep [{timestep+1}/{timesteps}], "
                      f"Accuracy: {metrics['performance']:.4f}, "
                      f"Clusters: {int(metrics['num_clusters'])}")

        return {
            'performance': max(performances) if performances else 0.0,
            'accuracy': max(performances) if performances else 0.0,
            'final_performance': performances[-1] if performances else 0.0,
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
        channels, height, width = self.dataset_info['input_shape']
        return {
            'num_instances': float(self.num_instances),
            'num_test_instances': float(self.num_test_instances),
            'num_classes': float(self.dataset_info['num_classes']),
            'input_channels': float(channels),
            'input_height': float(height),
            'input_width': float(width),
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
        if self.train_images is None:
            self.load_data()

        if self.meta_feature_extractor is None:
            self.meta_feature_extractor = ClusteringMetaFeatureExtractor(self.dataset_info)

        clusters = self.state[-1]['clusters'] if self.state else None
        metrics = self.state[-1]['metrics'] if self.state else None
        feature_dim = int(metrics.get('feature_dim', 0)) if metrics else 0

        features = self.meta_feature_extractor.extract_meta_features(
            clusters=clusters,
            metrics=metrics,
            num_instances=len(self.train_images),
            feature_dim=feature_dim,
            timestep=timestep,
        )

        # Record this timestep only after extracting, so 'performance_delta'
        # compares against the previous timestep rather than against itself.
        if metrics is not None:
            self.meta_feature_extractor.observe(
                metrics.get('fitness', 0.0), len(clusters or [])
            )

        return features

    # ------------------------------------------------------------------
    # Lifecycle and metadata
    # ------------------------------------------------------------------

    def reset_run_state(self) -> None:
        """Discard the clustering so a new run starts clean."""
        self.state = None
        self.current_design = None
        self.meta_feature_extractor = None
        self.is_trained = False
        self.num_design_changes = 0
        self.num_timesteps_executed = 0
        self.num_candidates_failed = 0
        self.num_reprojections = 0

        # Features are a pure function of the extractor and the (unchanged)
        # instance sample, so the cache stays valid across runs. The cluster
        # identifier counter is reset so two runs produce identical logs.
        label_utils.reset_identifiers()

        np.random.seed(self.random_seed)

    def get_application_type(self) -> str:
        """Get the type of application."""
        return 'composition'

    def supports_dynamic_designs(self) -> bool:
        """The design may change between timesteps; the clustering carries over."""
        return True

    def has_builtin_design_algorithm(self) -> bool:
        """This application applies the design it is given; the GA searches."""
        return False

    def get_num_timesteps(self) -> int:
        """Default number of timesteps for this application (thesis: 60)."""
        return 60

    def get_run_statistics(self) -> Dict[str, Any]:
        """Summary of what happened during the current run."""
        metrics = self.state[-1]['metrics'] if self.state else {}
        return {
            'num_timesteps_executed': self.num_timesteps_executed,
            'num_design_changes': self.num_design_changes,
            'num_candidates_failed': self.num_candidates_failed,
            'num_reprojections': self.num_reprojections,
            'num_clusters': int(metrics.get('num_clusters', 0)),
            'feature_dim': int(metrics.get('feature_dim', 0)),
            'excluded_design_options': dict(self.excluded_options),
            'final_metrics': {
                key: float(value) for key, value in metrics.items()
                if isinstance(value, (int, float, bool))
            },
        }
