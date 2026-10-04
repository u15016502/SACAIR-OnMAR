"""
Optional third-party dependencies of the clustering components.

Several design options in the clustering design space are implemented on top
of libraries that are not in ``requirements-min.txt``:

    k-medoids cluster creation      kmedoids
    DBSCAN cluster creation         kneed          (knee detection for eps)
    Markov cluster addition         markov_clustering, networkx
    CNN feature extraction          tensorflow, keras
    SIFT / BRIEF feature extraction opencv-python

These used to be imported at module scope, which meant a missing one made the
whole component module unimportable - a missing ``kneed`` took out all seven
cluster-creation options, not just the one that needs it. Here each is
imported once, failures are recorded rather than raised, and
:func:`available` lets the application drop the options it cannot run from the
design space instead of letting the genetic algorithm sample designs that
raise.

``pip install -r requirements-clustering.txt`` installs the lot.
"""

from typing import Any, Dict, Optional
import importlib


# Maps a short feature name to the modules it needs.
_FEATURE_MODULES: Dict[str, tuple] = {
    'kmedoids': ('kmedoids',),
    'kneed': ('kneed',),
    'markov': ('markov_clustering', 'networkx'),
    'tensorflow': ('tensorflow', 'keras'),
    'opencv': ('cv2',),
    'opencv_contrib': ('cv2',),
}

# Some features need more than the module to import: the attribute they use
# may live in a different distribution of the same package. BRIEF needs
# ``cv2.xfeatures2d``, which ships in opencv-contrib-python and not in
# opencv-python or opencv-python-headless. Without this check BRIEF looked
# available, then failed with "module 'cv2' has no attribute 'xfeatures2d'"
# and silently fell back to clustering raw pixels.
_FEATURE_PREDICATES = {
    'opencv_contrib': lambda: hasattr(_module('cv2'), 'xfeatures2d'),
}

_modules: Dict[str, Optional[Any]] = {}
_errors: Dict[str, str] = {}


def _module(name: str) -> Optional[Any]:
    """Import ``name`` once, caching both success and failure."""
    if name not in _modules:
        try:
            _modules[name] = importlib.import_module(name)
        except Exception as exc:  # ImportError, but also build-time failures
            _modules[name] = None
            _errors[name] = f"{type(exc).__name__}: {exc}"
    return _modules[name]


def available(feature: str) -> bool:
    """Whether every module backing ``feature`` imported successfully.

    Args:
        feature: One of the keys of :data:`_FEATURE_MODULES`.

    Returns:
        True if the feature can be used.
    """
    if feature not in _FEATURE_MODULES:
        raise KeyError(f"Unknown optional feature: {feature}")

    if not all(_module(name) is not None for name in _FEATURE_MODULES[feature]):
        return False

    predicate = _FEATURE_PREDICATES.get(feature)
    return True if predicate is None else bool(predicate())


def require(feature: str, option: str) -> None:
    """Raise a pointed error if ``feature`` is missing.

    Args:
        feature: The optional feature needed.
        option: The design option that needs it, for the message.

    Raises:
        ImportError: If the feature is unavailable.
    """
    if available(feature):
        return

    missing = [
        f"{name} ({_errors.get(name, 'not installed')})"
        for name in _FEATURE_MODULES[feature]
        if _module(name) is None
    ]
    if not missing:
        # Imported, but the attribute the feature needs is absent.
        missing = [f"{feature}: installed modules lack the required API"]
    raise ImportError(
        f"The design option '{option}' needs {feature}, which is unavailable: "
        f"{'; '.join(missing)}. Install it with "
        f"'pip install -r requirements-clustering.txt', or leave it out of the "
        f"design space (ClusteringCompositionApplication does this "
        f"automatically unless strict_design_space=True)."
    )


def unavailable_report() -> Dict[str, str]:
    """Every optional feature that is missing, with the reason.

    Returns:
        Feature name -> reason, for features that cannot be used.
    """
    report = {}
    for feature, names in _FEATURE_MODULES.items():
        if not available(feature):
            reasons = [
                f"{name}: {_errors.get(name, 'not installed')}"
                for name in names
                if _module(name) is None
            ] or [f"{feature}: installed modules lack the required API"]
            report[feature] = '; '.join(reasons)
    return report


# ----------------------------------------------------------------------
# Accessors used by the component modules.
# ----------------------------------------------------------------------

def kmedoids():
    """The ``kmedoids`` module."""
    require('kmedoids', 'cluster_creation=4 (k-medoids)')
    return _module('kmedoids')


def knee_locator():
    """``kneed.KneeLocator``."""
    require('kneed', 'cluster_creation=2 (DBSCAN)')
    return _module('kneed').KneeLocator


def markov():
    """The ``markov_clustering`` and ``networkx`` modules, as a pair."""
    require('markov', 'cluster_addition=4 (Markov clustering)')
    return _module('markov_clustering'), _module('networkx')


def networkx_sparse_matrix(graph):
    """Adjacency matrix of ``graph`` as a SciPy sparse *matrix*.

    Two separate library changes have to be absorbed here.

    ``networkx.to_scipy_sparse_matrix`` was removed in networkx 3.0 and
    replaced by ``to_scipy_sparse_array``, so the new name is used when
    present.

    The result is then forced to a ``csr_matrix``, because the replacement
    returns a sparse *array* and ``markov_clustering`` chooses between its
    sparse and dense code paths with ``scipy.sparse.isspmatrix``, which is
    False for sparse arrays. Handed an array it took the dense path and called
    ``np.linalg.matrix_power`` on the sparse object, which numpy sees as a
    0-dimensional object array - "0-dimensional array given. Array must be at
    least two-dimensional".

    Args:
        graph: A networkx graph.

    Returns:
        The adjacency matrix, as a ``scipy.sparse.csr_matrix``.
    """
    from scipy.sparse import csr_matrix

    nx = _module('networkx')
    if hasattr(nx, 'to_scipy_sparse_array'):
        adjacency = nx.to_scipy_sparse_array(graph)
    else:
        adjacency = nx.to_scipy_sparse_matrix(graph)

    return csr_matrix(adjacency)


def tensorflow():
    """The ``tensorflow`` and ``keras`` modules, as a pair."""
    require('tensorflow', 'feature_extraction=0..4 (pretrained CNN features)')
    return _module('tensorflow'), _module('keras')


def opencv():
    """The ``cv2`` module, for SIFT."""
    require('opencv', 'feature_extraction=6 (SIFT)')
    return _module('cv2')


def opencv_contrib():
    """The ``cv2`` module, checked for the contrib API that BRIEF needs."""
    require('opencv_contrib', 'feature_extraction=8 (BRIEF)')
    return _module('cv2')
