"""
One timestep of the video classification configuration application.

This module was an empty file. ``reference/thesis_chromosomes.py`` calls it as
``video_configuration_run(chromosome, trainX, trainY, testX, testY,
timestep)``, so a stub existed for the original three-application driver, but
nothing was ever committed into it - the README's description of ``run.py`` as
a training entry point described a file of zero bytes.

:func:`run` provides that entry point, in the same shape as
:func:`clustering_composition.run.run`: it applies a positional chromosome for
one timestep and returns the performance together with the state that carries
the training forward. The work itself lives in
:class:`applications.configuration.video.video_application.VideoConfigurationApplication`,
because a timestep of this application is a training epoch over a persistent
network and optimiser - state an application object holds naturally and a
function cannot.

The chromosome is the seven-gene video chromosome, in that order::

    0  keyframe extraction      4  learning rate
    1  number of segments       5  dropout
    2  base architecture        6  gradient norm clipping
    3  consensus function
"""

from typing import Any, Dict, List, Sequence, Tuple


# The seven genes, in the order the thesis chromosome lists them.
CHROMOSOME_ORDER: List[str] = [
    'keyframe_extraction',
    'num_segments',
    'base_architecture',
    'consensus_function',
    'learning_rate',
    'dropout',
    'gradient_norm_clipping',
]


def chromosome_to_design(chromosome: Sequence[Any]) -> Dict[str, Any]:
    """Convert a positional chromosome to a design dictionary.

    Args:
        chromosome: The seven gene values, in chromosome order.

    Returns:
        The design, keyed by gene name.
    """
    if len(chromosome) != len(CHROMOSOME_ORDER):
        raise ValueError(
            f"The video chromosome has {len(CHROMOSOME_ORDER)} genes "
            f"({', '.join(CHROMOSOME_ORDER)}); got {len(chromosome)}."
        )

    design = dict(zip(CHROMOSOME_ORDER, chromosome))
    for gene in ('keyframe_extraction', 'num_segments', 'base_architecture',
                 'consensus_function'):
        design[gene] = int(design[gene])
    for gene in ('learning_rate', 'dropout', 'gradient_norm_clipping'):
        design[gene] = float(design[gene])

    return design


def design_to_chromosome(design: Dict[str, Any]) -> List[Any]:
    """Convert a design dictionary to the positional chromosome.

    Args:
        design: Design keyed by gene name.

    Returns:
        The seven gene values, in chromosome order.
    """
    return [design[gene] for gene in CHROMOSOME_ORDER]


def run(
    chromosome: Sequence[Any],
    timestep: int,
    application=None,
    **application_kwargs,
) -> Tuple[float, Any]:
    """Apply a video design for one timestep and measure it.

    Args:
        chromosome: The seven-gene video chromosome.
        timestep: Current timestep.
        application: An existing
            :class:`~applications.configuration.video.video_application.VideoConfigurationApplication`
            to advance. Pass the one returned by a previous call so the
            network and optimiser carry across timesteps; None builds a fresh
            one, which starts training over.
        **application_kwargs: Forwarded to the application's constructor when
            one is built (``dataset_name``, ``train_list``, ``batch_size`` and
            so on).

    Returns:
        ``(performance, application)``, where performance is validation
        accuracy for this timestep and the application carries the training
        state forward into the next call.
    """
    # Imported here rather than at module scope: the application pulls in
    # torch and torchvision, and this module is also used just for the
    # chromosome conversions above.
    from applications.configuration.video.video_application import (
        VideoConfigurationApplication,
    )

    if application is None:
        application = VideoConfigurationApplication(**application_kwargs)
        application.load_data()

    metrics = application.exec_design(chromosome_to_design(chromosome), timestep)
    return float(metrics['performance']), application
