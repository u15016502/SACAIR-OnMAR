"""
Meta-learning support.

``mar_support`` holds the machinery shared by the four OnMAR/OffMAR variants:
meta-feature encoding, the GA-backed design algorithm, and repository pruning.
"""

from metalearner.mar_support import (
    MetaFeatureEncoder,
    DesignAlgorithm,
    prune_repository,
    load_approach,
    load_approach_config,
)

__all__ = [
    'MetaFeatureEncoder',
    'DesignAlgorithm',
    'prune_repository',
    'load_approach',
    'load_approach_config',
]
