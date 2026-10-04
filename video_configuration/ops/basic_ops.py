"""
Segment consensus for Temporal Segment Networks.

A TSN classifies each of ``num_segments`` snippets independently and then
*combines* the per-segment scores into one video-level prediction. That
combination is the consensus function, and which one to use is one of the
seven genes of the video design space.

Why this was rewritten
----------------------
The recovered implementation subclassed ``torch.autograd.Function`` in the
pre-0.4 style - instance ``__init__``, instance ``forward``/``backward``, and
invocation as ``SegmentConsensus(type, dim)(input)`` - and hand-wrote the
backward pass. PyTorch removed support for that style, so every forward pass
raised::

    RuntimeError: Legacy autograd function with non-static forward method is
    deprecated. Please use new-style autograd function with static forward
    method.

It would be possible to port it to the static-method API, but there is no
reason to: every consensus function here is a composition of differentiable
tensor operations, so autograd derives the backward pass itself. That removes
the hand-written gradients (and the chance of their being wrong) along with
the legacy API.

It also makes three of the five consensus functions real. The recovered
``SegmentConsensus.forward`` only implemented ``avg`` and ``identity`` and
returned ``None`` for everything else, while ``ConsensusModule`` quietly
rewrote ``rnn`` to ``identity``. So a design selecting ``max`` or ``topk``
produced ``None`` and crashed, and one selecting ``rnn`` silently got
``identity``.

The five functions
------------------
``avg``       mean over segments - the TSN paper's default
``max``       max over segments
``topk``      mean of the k highest-scoring segments
``identity``  no consensus; per-segment scores are kept
``weighted``  learnable per-segment weights, softmax-normalised

``weighted`` stands in for the fifth option, which the original command line
called ``rnn``. No recurrent consensus was ever implemented in this codebase -
``ConsensusModule`` aliased it to ``identity``, which made two of the five
options identical - so rather than reproduce a duplicate, the fifth option is
the simplest consensus that actually differs from the other four by *learning*
how to combine segments. ``rnn`` is accepted as an alias for it.
"""

import torch
import torch.nn as nn


CONSENSUS_TYPES = ('avg', 'max', 'topk', 'identity', 'weighted')


class Identity(nn.Module):
    """Passes its input through unchanged."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Return the input untouched."""
        return input


class SegmentConsensus(nn.Module):
    """Combines per-segment scores into one video-level score.

    Args:
        consensus_type: One of :data:`CONSENSUS_TYPES`, or ``'rnn'`` as an
            alias for ``'weighted'``.
        dim: The segment axis.
        k: Number of segments averaged by ``topk``.
        num_segments: Number of segments, needed to size ``weighted``'s
            parameters.
    """

    def __init__(self, consensus_type: str, dim: int = 1, k: int = 3,
                 num_segments: int = 3):
        super().__init__()

        # 'rnn' is what the original CLI called the fifth option; see the
        # module docstring for why it maps here.
        if consensus_type == 'rnn':
            consensus_type = 'weighted'

        if consensus_type not in CONSENSUS_TYPES:
            raise ValueError(
                f"Unknown consensus type {consensus_type!r}; "
                f"expected one of {CONSENSUS_TYPES} (or 'rnn')."
            )

        self.consensus_type = consensus_type
        self.dim = dim
        self.k = k
        self.num_segments = num_segments

        if consensus_type == 'weighted':
            # One logit per segment, softmaxed at forward time so the weights
            # stay a convex combination however they are trained.
            self.segment_logits = nn.Parameter(torch.zeros(num_segments))
        else:
            self.segment_logits = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Combine over the segment axis.

        Args:
            input: Scores shaped (batch, num_segments, num_classes).

        Returns:
            The combined scores, keeping the segment axis with size 1 - except
            for ``identity``, which returns its input unchanged.
        """
        if self.consensus_type == 'identity':
            return input

        if self.consensus_type == 'avg':
            return input.mean(dim=self.dim, keepdim=True)

        if self.consensus_type == 'max':
            return input.max(dim=self.dim, keepdim=True).values

        if self.consensus_type == 'topk':
            # k cannot exceed the number of segments actually present, which
            # is itself under search and so is not known until forward time.
            k = max(1, min(self.k, input.size(self.dim)))
            top = input.topk(k, dim=self.dim).values
            return top.mean(dim=self.dim, keepdim=True)

        # weighted
        num_segments = input.size(self.dim)
        logits = self.segment_logits
        if logits.numel() != num_segments:
            # The design's segment count can differ from the one this module
            # was built with; use the leading slice, or pad with zeros, so a
            # mismatch degrades to a uniform weighting rather than raising.
            if logits.numel() > num_segments:
                logits = logits[:num_segments]
            else:
                padding = torch.zeros(
                    num_segments - logits.numel(),
                    device=logits.device, dtype=logits.dtype,
                )
                logits = torch.cat([logits, padding])

        weights = torch.softmax(logits, dim=0)
        # Broadcast the weights along the segment axis only.
        shape = [1] * input.dim()
        shape[self.dim] = num_segments
        return (input * weights.view(shape)).sum(dim=self.dim, keepdim=True)


class ConsensusModule(nn.Module):
    """The consensus function as used by :class:`video_configuration.models.TSN`.

    Args:
        consensus_type: One of :data:`CONSENSUS_TYPES`, or ``'rnn'``.
        dim: The segment axis.
        k: Number of segments averaged by ``topk``.
        num_segments: Number of segments.
    """

    def __init__(self, consensus_type: str, dim: int = 1, k: int = 3,
                 num_segments: int = 3):
        super().__init__()
        self.consensus = SegmentConsensus(
            consensus_type, dim=dim, k=k, num_segments=num_segments
        )
        self.consensus_type = self.consensus.consensus_type
        self.dim = dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Combine per-segment scores over the segment axis.

        Args:
            input: Scores shaped (batch, num_segments, num_classes).

        Returns:
            The combined scores; see :meth:`SegmentConsensus.forward`.
        """
        return self.consensus(input)
