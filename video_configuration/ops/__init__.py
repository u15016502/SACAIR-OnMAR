"""TSN tensor operations.

``from ops.basic_ops import *`` only resolved when ``video_configuration/``
was itself on sys.path; the import is now package-relative.
"""

from video_configuration.ops.basic_ops import (
    CONSENSUS_TYPES,
    ConsensusModule,
    Identity,
    SegmentConsensus,
)

__all__ = ['CONSENSUS_TYPES', 'ConsensusModule', 'Identity', 'SegmentConsensus']
