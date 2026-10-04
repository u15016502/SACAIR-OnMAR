"""
Keyframe extraction strategies for Temporal Segment Networks.

A TSN does not see a whole video. It divides the frames into ``num_segments``
segments and takes one snippet from each, and *which* frame it takes is the
``keyframe_extraction`` gene - the first of the seven options in the video
design space (``reference/thesis_chromosomes.py``).

The recovered ``TSNDataSet`` hard-coded three behaviours and chose between
them by training mode, not by design: ``_sample_indices`` (random within each
segment) for training, ``_get_val_indices`` and ``_get_test_indices``
(both the segment centre) otherwise. The design space declares five
strategies, so the three are generalised to five here and selected by gene
value rather than by mode:

    1  random     one uniformly random frame per segment (TSN's training
                  sampler; the most augmentation)
    2  centre     the middle frame of each segment (TSN's evaluation sampler;
                  deterministic)
    3  uniform    frames evenly spaced across the whole video, ignoring
                  segment boundaries
    4  dense      a contiguous run of frames from the video's middle, which
                  trades temporal coverage for fine-grained motion
    5  strided    every k-th frame from a random start, k chosen so the
                  frames span the video

Every strategy returns ``num_segments`` 1-based frame indices, which is the
convention ``TSNDataSet.get`` expects, and clamps them into the video's
length so a video shorter than its segment count still yields a sample
(repeating frames rather than reading past the end).
"""

from typing import Optional
import numpy as np


# Gene value -> strategy name. The design space uses the thesis's 1-based
# numbering.
KEYFRAME_STRATEGIES = {
    1: 'random',
    2: 'centre',
    3: 'uniform',
    4: 'dense',
    5: 'strided',
}


def strategy_name(gene_value: int) -> str:
    """The strategy a gene value selects.

    Args:
        gene_value: The ``keyframe_extraction`` gene, 1-5.

    Returns:
        The strategy's name.

    Raises:
        ValueError: If the value is not a known strategy.
    """
    if int(gene_value) not in KEYFRAME_STRATEGIES:
        raise ValueError(
            f"Unknown keyframe extraction strategy {gene_value}; "
            f"expected one of {sorted(KEYFRAME_STRATEGIES)}."
        )
    return KEYFRAME_STRATEGIES[int(gene_value)]


def sample_indices(
    num_frames: int,
    num_segments: int,
    new_length: int = 1,
    strategy: int = 2,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Choose which frames of a video to read.

    Args:
        num_frames: Frames available in the video.
        num_segments: Snippets to take.
        new_length: Frames consumed per snippet (more than 1 for optical
            flow), so an index is never chosen so late that its snippet would
            run past the end.
        strategy: The ``keyframe_extraction`` gene, 1-5.
        rng: Optional numpy Generator, for the strategies that are random.

    Returns:
        ``num_segments`` 1-based frame indices.
    """
    rng = np.random.default_rng() if rng is None else rng
    name = strategy_name(strategy)

    # The last index a snippet may start at.
    last_start = max(1, num_frames - new_length + 1)

    if name == 'random':
        segment_length = last_start // num_segments
        if segment_length > 0:
            offsets = (
                np.arange(num_segments) * segment_length
                + rng.integers(segment_length, size=num_segments)
            )
        else:
            # Fewer frames than segments: sample with replacement rather than
            # returning all zeros, so the snippets still differ.
            offsets = np.sort(rng.integers(last_start, size=num_segments))

    elif name == 'centre':
        tick = last_start / float(num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(num_segments)])

    elif name == 'uniform':
        # Evenly spaced over the whole video, endpoints included.
        offsets = np.linspace(0, last_start - 1, num=num_segments).astype(int)

    elif name == 'dense':
        # A contiguous block centred on the video's midpoint.
        start = max(0, (last_start - num_segments) // 2)
        offsets = start + np.arange(num_segments)

    else:  # strided
        stride = max(1, last_start // num_segments)
        max_start = max(1, last_start - stride * (num_segments - 1))
        start = int(rng.integers(max_start))
        offsets = start + np.arange(num_segments) * stride

    # Clamp into range, then convert to the 1-based indices TSNDataSet uses.
    offsets = np.clip(offsets, 0, last_start - 1)
    return offsets.astype(int) + 1
