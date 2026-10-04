"""
Clustering composition application components.

One module per design option of the eleven-gene clustering chromosome, plus
:mod:`clustering_composition.run`, which applies a chromosome for one
timestep. ``applications/composition/clustering/`` wraps these in the
``BaseApplication`` interface that the meta-learning layer drives.

Submodules are imported lazily. Importing them eagerly pulled in TensorFlow
(via :mod:`clustering_composition.feature_extraction`) on any ``import
clustering_composition``, which cost seconds and failed outright on a machine
without it - even for callers that only wanted, say, the distance metrics.
"""

import importlib
from typing import Any

__all__ = [
    'cluster',
    'cluster_addition',
    'cluster_creation',
    'cluster_initialization',
    'cluster_merging',
    'cluster_removal',
    'cluster_splitting',
    'feature_extraction',
    'label_utils',
    'optional_deps',
    'run',
    'stopping_criteria',
    'utils',
]


def __getattr__(name: str) -> Any:
    """Import a submodule on first attribute access."""
    if name in __all__:
        module = importlib.import_module(f'{__name__}.{name}')
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
