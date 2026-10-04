"""
Shared machinery for the OnMAR and OffMAR variants.

The four approach variants (OnMAR/OffMAR x accuracy/design prediction) differ
only in *when* the meta-learner is trained and *what* it predicts. Everything
below is common to all four, and lives here so the four implementations cannot
drift apart:

    MetaFeatureEncoder  - meta-features -> fixed-length vector
    DesignAlgorithm     - the GA that proposes designs (Algorithm 9 line 7/10)
    prune_repository    - the theta_p pruning step
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from genetic_algorithm.ga import GeneticAlgorithm


# ----------------------------------------------------------------------
# Loading the approach implementations
# ----------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

# The four approach modules live in directories whose names contain a hyphen
# ('OnMAR/accuracy-prediction'), which is not a legal Python identifier, so
# they cannot be reached by a normal import and have to be loaded by path.
APPROACH_PATHS = {
    'onmar-accuracy': REPO_ROOT / 'OnMAR' / 'accuracy-prediction' / 'onmar.py',
    'onmar-design': REPO_ROOT / 'OnMAR' / 'design-prediction' / 'onmar.py',
    'offmar-accuracy': REPO_ROOT / 'OffMAR' / 'accuracy-prediction' / 'offmar.py',
    'offmar-design': REPO_ROOT / 'OffMAR' / 'design-prediction' / 'offmar.py',
}

APPROACH_CONFIG_PATHS = {
    key: path.parent / 'config.py' for key, path in APPROACH_PATHS.items()
}

_loaded_modules: Dict[str, ModuleType] = {}


def _load_module_from_path(module_name: str, path: Path) -> ModuleType:
    """Import a module from an explicit file path."""
    if module_name in _loaded_modules:
        return _loaded_modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_name} from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _loaded_modules[module_name] = module
    return module


def load_approach(name: str) -> ModuleType:
    """Load one of the four approach modules.

    Args:
        name: One of 'onmar-accuracy', 'onmar-design', 'offmar-accuracy',
            'offmar-design'.

    Returns:
        The module, from which the approach class can be taken.

    Example:
        >>> module = load_approach('onmar-accuracy')
        >>> onmar = module.OnMARAccuracyPrediction(application=app)
    """
    key = name.lower()
    if key not in APPROACH_PATHS:
        raise ValueError(
            f"Unknown approach '{name}'. Choose from {sorted(APPROACH_PATHS)}"
        )
    return _load_module_from_path(f"mar_{key.replace('-', '_')}", APPROACH_PATHS[key])


def load_approach_config(name: str) -> ModuleType:
    """Load the config module that sits beside an approach.

    Args:
        name: As for :func:`load_approach`.

    Returns:
        The approach's config module.
    """
    key = name.lower()
    if key not in APPROACH_CONFIG_PATHS:
        raise ValueError(
            f"Unknown approach '{name}'. Choose from {sorted(APPROACH_CONFIG_PATHS)}"
        )
    return _load_module_from_path(
        f"mar_{key.replace('-', '_')}_config", APPROACH_CONFIG_PATHS[key]
    )


# ----------------------------------------------------------------------
# Meta-feature encoding
# ----------------------------------------------------------------------

def _flatten_items(obj: Any, prefix: str = '') -> List[Tuple[str, float]]:
    """Flatten a nested meta-feature structure into ``(key, value)`` pairs."""
    items: List[Tuple[str, float]] = []

    if isinstance(obj, dict):
        for key in sorted(obj.keys()):
            child = f"{prefix}.{key}" if prefix else str(key)
            items.extend(_flatten_items(obj[key], child))
    elif isinstance(obj, (list, tuple, np.ndarray)):
        flat = np.asarray(obj).ravel()
        for index, value in enumerate(flat):
            items.append((f"{prefix}[{index}]", float(value)))
    else:
        try:
            value = float(obj)
        except (TypeError, ValueError):
            value = 0.0
        if not np.isfinite(value):
            value = 0.0
        items.append((prefix or 'value', value))

    return items


class MetaFeatureEncoder:
    """Encodes meta-features to a fixed-length vector.

    Which meta-features exist varies by timestep: features derived from the
    model need a model, weight-distance features need a previous timestep, and
    landscape features need some history. Encoding whatever happens to be
    present would produce vectors of different lengths across timesteps, and
    no meta-learner can be fitted on ragged input - it fails with an
    inhomogeneous-shape error the first time it is trained.

    So the schema is fixed once and then held: the application's declared
    schema if it has one, otherwise the keys seen at the first encode. After
    that, a missing key encodes as 0.0 and an unexpected key is dropped (once,
    with a warning), so the vector length never changes mid-run.
    """

    def __init__(self, feature_names: Optional[Sequence[str]] = None):
        """
        Args:
            feature_names: The application's declared schema, if any.
        """
        self.feature_names: Optional[List[str]] = (
            list(feature_names) if feature_names else None
        )
        self._warned_keys: set = set()

    @property
    def length(self) -> int:
        """Encoded length, or 0 before the schema is fixed."""
        return len(self.feature_names) if self.feature_names else 0

    def encode(self, meta_features: Dict[str, Any]) -> np.ndarray:
        """Encode meta-features, fixing the schema on first use.

        Args:
            meta_features: Meta-features for one timestep.

        Returns:
            Vector of length :attr:`length`.
        """
        items = dict(_flatten_items(meta_features))

        if self.feature_names is None:
            self.feature_names = sorted(items.keys())

        unexpected = set(items) - set(self.feature_names)
        new_unexpected = unexpected - self._warned_keys
        if new_unexpected:
            self._warned_keys |= new_unexpected
            print(
                f"  Note: ignoring {len(new_unexpected)} meta-feature(s) outside "
                f"the fixed schema: {sorted(new_unexpected)[:4]}"
                f"{' ...' if len(new_unexpected) > 4 else ''}"
            )

        return np.asarray(
            [items.get(name, 0.0) for name in self.feature_names], dtype=np.float32
        )


# ----------------------------------------------------------------------
# Design algorithm
# ----------------------------------------------------------------------

class DesignAlgorithm:
    """The design algorithm of Algorithm 9, ``c <- design_algorithm(dataset, t)``.

    For applications that merely apply the design they are given (the CNN), a
    genetic algorithm searches the design space, advancing one generation per
    call with the population persisting between calls - so the search is
    cumulative across the run rather than an independent draw each timestep.

    For applications whose own ``train`` already embeds a search (segmentation,
    Fuzzy ART), that search is used instead and this class just extracts the
    design it reports.
    """

    def __init__(
        self,
        application,
        random_seed: int = 42,
        config_overrides: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            application: The application being optimised.
            random_seed: Seed for the GA.
            config_overrides: GA settings overriding the application's
                suggestions.
        """
        self.application = application
        self.uses_builtin = application.has_builtin_design_algorithm()
        self.ga: Optional[GeneticAlgorithm] = None

        if not self.uses_builtin:
            config = dict(application.get_design_algorithm_config())
            config.update(config_overrides or {})
            self.ga = GeneticAlgorithm(
                design_space=application.get_structured_design_space(),
                fitness_fn=application.evaluate_candidate,
                population_fitness_fn=getattr(application, 'evaluate_population', None),
                random_seed=random_seed,
                **config,
            )

    def propose(
        self,
        timestep: int,
        incumbent: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Produce a design for this timestep.

        Args:
            timestep: Current timestep.
            incumbent: The design currently in use, seeded into the GA's
                initial population so a known-good design is not lost.

        Returns:
            A design.
        """
        if self.ga is not None:
            # With no incumbent - which is the case at the start of a run,
            # since the approaches pass initial_design=None - fall back to the
            # application's default design as the seed.
            #
            # This matters more than it looks. A run gives the design
            # algorithm only theta_t calls, so a 14-timestep run evolves 7
            # generations; from an entirely random population that is far too
            # little to discover a competent architecture, and the GA returns
            # the best of a weak lineage. Seeding one known-good individual
            # gives the search a floor to improve on rather than a cold start.
            seed = incumbent
            if seed is None:
                seed = self._default_seed()
            return self.ga.step(timestep, seed_design=seed)

        # The application searches for itself; take the design it reports.
        results = self.application.train(
            incumbent if incumbent is not None else {}, timesteps=1
        )
        design = results.get('best_design') or results.get('design') or incumbent
        if design is None:
            design = self.application.sample_design()
        return design

    def _default_seed(self) -> Optional[Dict[str, Any]]:
        """The application's default design, if it offers one."""
        getter = getattr(self.application, 'get_default_design', None)
        if getter is None:
            return None
        try:
            design = getter()
        except Exception:
            return None
        return design if self.application.validate_design(design) else None

    def get_statistics(self) -> Dict[str, Any]:
        """Effort spent by the design algorithm."""
        if self.ga is None:
            return {'design_algorithm': 'application-builtin'}
        stats = self.ga.get_statistics()
        stats['design_algorithm'] = 'genetic-algorithm'
        return stats


# ----------------------------------------------------------------------
# Knowledge repository pruning
# ----------------------------------------------------------------------

def prune_repository(
    entries: List[Dict[str, Any]],
    vectors_x: List[np.ndarray],
    vectors_y: List[Any],
    theta_p: float,
    min_keep: int = 3,
) -> Tuple[List[Dict[str, Any]], List[np.ndarray], List[Any], Dict[str, Any]]:
    """Drop repository entries whose performance fell below theta_p.

    This is the design-prediction pruning step: the meta-learner is trained to
    reproduce designs, so it should only be shown designs worth reproducing.

    Pruning happens once, immediately before the meta-learner is first
    trained - half way through the run. Applying it on every update instead
    would empty the repository continuously whenever performance sits below
    theta_p, leaving the meta-learner untrained and the whole mechanism inert.

    If the threshold would leave too little to fit on, the best ``min_keep``
    entries are retained instead and the shortfall is reported, so a run
    produces a usable (if weakly informed) meta-learner rather than silently
    falling back to random designs.

    Args:
        entries: Repository entries, each with a 'performance' value.
        vectors_x: Pre-encoded inputs, index-aligned with ``entries``.
        vectors_y: Pre-encoded targets, index-aligned with ``entries``.
        theta_p: Performance threshold.
        min_keep: Entries to keep if the threshold is too strict.

    Returns:
        ``(entries, vectors_x, vectors_y, info)`` after pruning.
    """
    if not entries:
        return entries, vectors_x, vectors_y, {
            'pruned': 0, 'kept': 0, 'relaxed': False,
        }

    kept = [i for i, entry in enumerate(entries) if entry['performance'] >= theta_p]
    relaxed = False

    if len(kept) < min_keep:
        # Threshold too strict for what this run actually achieved: keep the
        # best entries so the meta-learner still has something to learn from.
        relaxed = True
        ranked = sorted(
            range(len(entries)), key=lambda i: entries[i]['performance'], reverse=True
        )
        kept = sorted(ranked[:min(min_keep, len(entries))])

    info = {
        'pruned': len(entries) - len(kept),
        'kept': len(kept),
        'relaxed': relaxed,
        'theta_p': theta_p,
        'best_performance': max(entry['performance'] for entry in entries),
    }

    return (
        [entries[i] for i in kept],
        [vectors_x[i] for i in kept],
        [vectors_y[i] for i in kept],
        info,
    )
