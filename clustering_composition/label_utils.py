"""
Label-to-instance helpers and cluster identifiers, shared by the clustering
components.

:func:`get_images_for_label` and :func:`get_image_indices_for_label` turn a
flat label assignment (what scikit-learn's estimators return as ``labels_``)
into the per-cluster vector lists that
:class:`clustering_composition.cluster.cluster` expects.

They originally lived in ``dataset/utils.py``, a module that is not part of
this bundle - ``cluster_creation.py`` carried its own copy and every other
component imported the missing one. Both copies were identical, so they are
collected here once and imported from both places.

The string comparison in the fallback branch is deliberate: cluster
identifiers are sometimes numpy integers and sometimes strings, and ``==``
between those two is False even when they denote the same cluster.

Cluster identifiers
-------------------
Every component that builds clusters needs fresh identifiers for them, and
each one minted them as ``[str(time.time()).replace('.','') for ul in
unique_labels]``. A list comprehension iterates faster than the clock ticks,
so those identifiers were *not* unique: measured on the development machine,
10 labels produced 7 distinct identifiers and 50 produced 25.

That is not a cosmetic problem. Clusters are matched to their instances by
identifier, so two clusters sharing one silently became a single merged
cluster holding the union of their members, and roughly half of all clusters
were lost at every composition step. :func:`new_identifiers` replaces the
clock with a counter.

Identifiers must be strings of digits: ``convert_clusters_to_label_
representations`` returns them in an array alongside ``-1`` for unassigned
instances, and ``clustering_accuracy`` casts that array with
``astype(np.int64)``. A plain counter satisfies this; a nanosecond timestamp
would not, being too wide for int64 once made unique. Only the distinctness of
the values matters downstream - ``convert_labels`` re-indexes them to
contiguous integers before anything is measured.
"""

import itertools
from copy import deepcopy

# Starts at 1, so no identifier can collide with the -1 used for an
# unassigned instance.
_identifier_counter = itertools.count(1)


def new_identifiers(count):
    """Mint ``count`` distinct cluster identifiers.

    Args:
        count: How many identifiers are needed.

    Returns:
        A list of ``count`` distinct digit strings.
    """
    return [str(next(_identifier_counter)) for _ in range(int(count))]


def new_identifier():
    """Mint a single distinct cluster identifier.

    Returns:
        A digit string, distinct from every other identifier minted.
    """
    return str(next(_identifier_counter))


def reset_identifiers():
    """Restart identifier numbering, so a fresh run reproduces the last one.

    Only the distinctness of identifiers affects results, but resetting keeps
    the values themselves identical between runs, which makes two runs' logs
    comparable line by line.
    """
    global _identifier_counter
    _identifier_counter = itertools.count(1)


def get_images_for_label(label, images, labels):
    """Vectors whose assigned label is ``label``.

    Args:
        label: The cluster label to collect.
        images: The dataset instances, in assignment order.
        labels: One label per instance.

    Returns:
        A list of the matching instances.
    """
    _images = []

    for idx, image in enumerate(images):
        if labels[idx] == label:
            _images.append(deepcopy(image))
        else:
            if str(labels[idx]) == str(label):
                _images.append(deepcopy(image))

    return _images


def get_image_indices_for_label(label, images, labels):
    """Indices of the vectors whose assigned label is ``label``.

    Args:
        label: The cluster label to collect.
        images: The dataset instances, in assignment order.
        labels: One label per instance.

    Returns:
        A list of the matching indices.
    """
    _images = []

    for idx, image in enumerate(images):
        if labels[idx] == label:
            _images.append(deepcopy(idx))
        else:
            if str(labels[idx]) == str(label):
                _images.append(deepcopy(idx))

    return _images
