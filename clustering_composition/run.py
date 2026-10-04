"""
One timestep of the clustering composition application.

:func:`run` is ``aa.exec(c, dataset, t)`` of Algorithm 9 for this application:
it advances a clustering by a single composition step under the design
``chromosome`` and scores the result. The clusters themselves live in the
``state`` list that is passed in and returned, so a run is one clustering
evolving over its timesteps rather than an independent clustering per
timestep.

The chromosome is the eleven-gene clustering chromosome of
``reference/thesis_chromosomes.py``, in that order::

    0  distance metric          4  cluster removal      8  stopping criteria
    1  cluster initialization   5  cluster merging      9  step
    2  cluster creation         6  cluster splitting    10 feature extraction
    3  cluster addition         7  membership limit

Gene 9, the *step*, is what makes this a composition application: it selects
which single composition operator is applied this timestep.

Changes from the recovered version
----------------------------------
* The two ``import *`` lines are now explicit. ``stopping_criteria`` and
  ``utils`` both define ``check_for_changes_in_cluster_assignments`` with
  different signatures, and the code worked only because the star-imports
  happened in the order that let the one-argument version win.
* Feature extraction is optionally memoised. It depends only on gene 10 and
  on the (fixed) image batch, but the genetic algorithm scores a whole
  population per generation, and a ResNet50 pass per candidate dominated
  everything else. Pass a dict as ``feature_cache`` to compute each
  extractor's output once.
* Per-timestep metrics are recorded in the state entry rather than only
  printed, so the application can report and build meta-features from them.
* Printing is opt-in. The function is called once per candidate design, and
  printing a line each time buried the actual run output.
"""

import numpy as np
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    mutual_info_score, adjusted_rand_score,
)

from clustering_composition.utils import (
    get_feature_extraction,
    get_cluster_initialization,
    get_cluster_creation,
    get_cluster_addition,
    get_cluster_removal,
    get_cluster_merging,
    get_cluster_splitting,
    convert_clusters_to_label_representations,
    clustering_accuracy,
    convert_labels,
    flatten_or_return,
)
from clustering_composition.stopping_criteria import (
    check_for_changes_in_cluster_assignments,
    check_for_changes_in_cluster_variance,
)


def extract_features(chromosome, images, feature_cache=None):
    """Feature-extract ``images`` under the chromosome's extractor gene.

    Args:
        chromosome: The clustering chromosome.
        images: Image batch, shaped (N, H, W, C).
        feature_cache: Optional dict memoising extractor index -> features.
            Feature extraction depends only on the extractor and the images,
            both fixed within a run, so this turns a per-candidate cost into a
            per-extractor one.

    Returns:
        The feature matrix, as a numpy array.
    """
    extractor = chromosome[-1]

    if feature_cache is not None and extractor in feature_cache:
        return feature_cache[extractor]

    features = np.array(get_feature_extraction(extractor, images))

    if feature_cache is not None:
        feature_cache[extractor] = features

    return features


def as_matrix(data):
    """Flatten a batch of feature vectors to a 2-D ``(n_instances, n_features)`` matrix.

    ``flatten_or_return`` cannot do this reliably for a batch. It is used for
    both single instances and whole batches, and for a three-dimensional input
    with ``expected_shape=2`` it returns ``reshape(1, -1)`` - correct for one
    ``(H, W, C)`` instance, but for a batch of ``(30, 2)`` keypoint features
    it collapses all N instances into a single row. That surfaced as
    "Found input variables with inconsistent numbers of samples: [1, 48]" from
    ``silhouette_score`` once the SIFT extractor started returning real
    keypoints instead of falling back to raw pixels.

    Here the leading axis is known to be the instance axis, so the reshape is
    unambiguous.

    Args:
        data: Feature array or sequence, instances along the first axis.

    Returns:
        A 2-D float array with one row per instance.
    """
    array = np.asarray(data, dtype=float)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array.reshape(array.shape[0], -1)


def _cluster_quality(data, labels_pred, labels_true):
    """Internal and external cluster-quality indices.

    Each index is undefined for a single-cluster partition; the recovered code
    reported -1 in that case and that is kept, so the numbers stay comparable
    with ``sample_runs/``.

    Args:
        data: The feature matrix the clustering was computed on.
        labels_pred: Predicted cluster labels, as contiguous integers.
        labels_true: Ground-truth labels.

    Returns:
        Dictionary of silhouette, Davies-Bouldin, Calinski-Harabasz, adjusted
        Rand index and mutual information.
    """
    flat = as_matrix(data)
    degenerate = len(list(set(labels_pred))) == 1

    return {
        'silhouette': -1 if degenerate else silhouette_score(flat, labels_pred),
        'db_index': -1 if degenerate else davies_bouldin_score(flat, labels_pred),
        'ch_index': -1 if degenerate else calinski_harabasz_score(flat, labels_pred),
        'ari': adjusted_rand_score(labels_true, labels_pred),
        'mi': mutual_info_score(labels_true, labels_pred),
    }


def run(chromosome, dataloader, timestep, state=None, feature_cache=None, verbose=False):
    """Advance the clustering by one composition step and score it.

    Args:
        chromosome: The eleven-gene clustering chromosome.
        dataloader: Something iterable yielding ``(images, labels)``; only the
            first batch is used, and only when ``state`` is None. Images are
            expected as (N, C, H, W), as the project's loaders produce them.
        timestep: Current timestep, for reporting.
        state: The run's state list, or None to initialise a new clustering.
        feature_cache: Optional dict memoising feature extraction.
        verbose: Print the per-timestep metric line.

    Returns:
        ``(fitness, state)``, where fitness is clustering accuracy and state
        has this timestep's entry appended. The entry carries the clusters,
        the images, the labels, the features and a ``metrics`` dictionary.
    """
    if state is None:
        images = None
        labels = None
        for i, data in enumerate(dataloader, 0):
            images, labels = data
            break

        # Labels must be a value-hashable array: ``set()`` over a torch tensor
        # hashes its elements by identity, which made every instance look like
        # its own class and set n_clusters to the batch size.
        labels = np.asarray(labels)
        images = np.asarray(images)

        n_clusters = len(list(set(labels.tolist())))

        # (N, C, H, W) -> (N, W, H, C). This is the recovered reshape, kept as
        # it was: for the square images the loaders produce it is the
        # channels-last layout the Keras extractors expect.
        images = images.reshape((images.shape[0], images.shape[-1], images.shape[-2], images.shape[-3]))

        data = extract_features(chromosome, images, feature_cache)
        clusters = get_cluster_initialization(
            chromosome[1], data, n_clusters, chromosome[-4], chromosome[0]
        )
        stopped = False
        state = []
    else:
        clusters = state[-1]['clusters']
        images = state[-1]['images']
        labels = state[-1]['labels']
        n_clusters = len(list(set(np.asarray(labels).tolist())))
        data = extract_features(chromosome, images, feature_cache)

        if chromosome[8] == 0:
            stopped = check_for_changes_in_cluster_assignments(state)
        if chromosome[8] == 1:
            stopped = check_for_changes_in_cluster_variance(state)

    if stopped == False:
        # Gene 9 selects the single composition operator for this timestep.
        if chromosome[-2] == 0:
            clusters = get_cluster_creation(chromosome[2], data, clusters, n_clusters, chromosome[-4], chromosome[0])
        if chromosome[-2] == 1:
            clusters = get_cluster_addition(chromosome[3], data, clusters, n_clusters, chromosome[-4], chromosome[0])
        if chromosome[-2] == 2:
            clusters = get_cluster_removal(chromosome[4], data, clusters, n_clusters, chromosome[-4], chromosome[0])
        if chromosome[-2] == 3:
            clusters = get_cluster_merging(chromosome[5], data, clusters, n_clusters, chromosome[-4], chromosome[0])
        if chromosome[-2] == 4:
            clusters = get_cluster_splitting(chromosome[6], data, clusters, n_clusters, chromosome[-4], chromosome[0])

    if clusters is None:
        clusters = []

    predicted_labels = convert_clusters_to_label_representations(
        data, clusters, chromosome[-4], chromosome[0]
    )
    fitness = clustering_accuracy(predicted_labels, labels)

    labels_pred = np.array(predicted_labels).flatten().astype(np.int64)
    labels_true = np.array(labels).flatten().astype(np.int64)
    labels_pred = convert_labels(labels_pred, labels_true)

    metrics = _cluster_quality(data, labels_pred, labels_true)
    metrics['fitness'] = fitness
    metrics['num_clusters'] = len(clusters)
    metrics['stopped'] = bool(stopped)
    metrics['feature_dim'] = int(as_matrix(data).shape[1])

    if verbose:
        print(
            str(timestep) + '\t' + str(fitness) + '\t' + str(metrics['silhouette'])
            + '\t' + str(metrics['db_index']) + '\t' + str(metrics['ch_index'])
            + '\t' + str(metrics['ari']) + '\t' + str(metrics['mi'])
            + '\t' + str(chromosome)
        )

    state.append({
        'clusters': clusters,
        'images': images,
        'labels': labels,
        'data': data,
        'metrics': metrics,
    })

    return fitness, state
