"""
Video classification configuration application components.

A Temporal Segment Networks (TSN) implementation plus the pieces the video
design space needs: :mod:`video_configuration.keyframes` for the segment
sampling strategies, :mod:`video_configuration.synthetic` for a generated
dataset the pipeline can be verified against, and
:mod:`video_configuration.run` for one timestep.
``applications/configuration/video/`` wraps these in the ``BaseApplication``
interface the meta-learning layer drives.

Submodules are imported lazily. ``from run import run`` at package scope was
both an absolute import of a sibling module (so it failed from anywhere but
``video_configuration/``) and a reference to a file that was empty; importing
eagerly would also pull in torch and torchvision on any ``import
video_configuration``.
"""

import importlib
from typing import Any

__all__ = [
    'dataset',
    'keyframes',
    'models',
    'ops',
    'run',
    'synthetic',
    'transforms',
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
