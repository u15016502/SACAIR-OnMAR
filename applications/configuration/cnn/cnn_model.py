"""
Configurable CNN model.

Builds a CNN from a per-layer design, following the chromosome layout of
Appendix A.3: each convolutional layer carries its own filter count, batch
normalisation flag, activation, dropout rate and pooling size, and each dense
layer its own node count, batch normalisation flag, activation and dropout.

Design format
-------------
    {
        'conv_layers': [
            {'filters': 64, 'batch_norm': 1, 'activation': 3,
             'dropout': -1.0, 'max_pool': 2},
            ...
        ],
        'dense_layers': [
            {'nodes': 128, 'batch_norm': 1, 'activation': 3, 'dropout': 0.3},
            ...
        ],
        'optimizer': 1,
        'learning_rate': 0.001,
    }

A dropout of ``-1.0`` means no dropout, and a ``max_pool`` of ``0`` means no
pooling, matching the thesis encoding.
"""

from typing import Any, Dict, Iterable, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Thesis activation numbering (Appendix A.3).
ACTIVATIONS = {
    1: 'elu',
    2: 'gelu',
    3: 'relu',
    4: 'selu',
    5: 'sigmoid',
    6: 'softmax',
    7: 'softplus',
    8: 'swish',
    9: 'tanh',
}


def build_optimizer(parameters: Iterable, design: Dict[str, Any]) -> optim.Optimizer:
    """Create the optimiser a design specifies.

    Lives here rather than on the application so that worker processes
    evaluating candidate designs can build one without importing the
    application module.

    Args:
        parameters: Parameters to optimise.
        design: Design carrying 'optimizer' and 'learning_rate'.

    Returns:
        The optimiser.
    """
    lr = float(design.get('learning_rate', 0.001))
    opt_type = int(design.get('optimizer', 1))

    if opt_type == 1:
        return optim.Adam(parameters, lr=lr)
    if opt_type == 2:
        return optim.Adamax(parameters, lr=lr)
    if opt_type == 3:
        return optim.RMSprop(parameters, lr=lr)
    if opt_type == 4:
        return optim.Adagrad(parameters, lr=lr)
    if opt_type == 5:
        return optim.Adadelta(parameters, lr=lr)
    if opt_type == 6:
        return optim.SGD(parameters, lr=lr, momentum=0.9)
    if opt_type == 7:
        return optim.NAdam(parameters, lr=lr)
    return optim.Adam(parameters, lr=lr)


def apply_activation(x: torch.Tensor, activation_id: int) -> torch.Tensor:
    """Apply a thesis-numbered activation function."""
    if activation_id == 1:
        return F.elu(x)
    if activation_id == 2:
        return F.gelu(x)
    if activation_id == 3:
        return F.relu(x)
    if activation_id == 4:
        return F.selu(x)
    if activation_id == 5:
        return torch.sigmoid(x)
    if activation_id == 6:
        return F.softmax(x, dim=1)
    if activation_id == 7:
        return F.softplus(x)
    if activation_id == 8:
        return F.silu(x)
    if activation_id == 9:
        return torch.tanh(x)
    return F.relu(x)


class ConfigurableCNN(nn.Module):
    """A CNN whose architecture is built from a per-layer design."""

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        num_classes: int,
        design: Dict[str, Any],
        max_flat_features: int = 8192,
    ):
        """
        Args:
            input_shape: ``(channels, height, width)``.
            num_classes: Number of output classes.
            design: Per-layer design dictionary (see module docstring).
            max_flat_features: Cap on the flattened feature count entering the
                first dense layer. A design with many filters and no pooling
                would otherwise produce a dense layer of billions of
                parameters; an adaptive pool is inserted to stay under the cap.
        """
        super().__init__()

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.design = design
        self.max_flat_features = max_flat_features

        self.conv_specs: List[Dict[str, Any]] = list(design.get('conv_layers', []))
        self.dense_specs: List[Dict[str, Any]] = list(design.get('dense_layers', []))

        # --- Convolutional stack -------------------------------------
        self.conv_blocks = nn.ModuleList()
        in_channels = input_shape[0]

        for spec in self.conv_specs:
            filters = int(spec['filters'])
            block = nn.ModuleDict({
                'conv': nn.Conv2d(in_channels, filters, kernel_size=3, padding=1)
            })
            if int(spec.get('batch_norm', 0)) == 1:
                block['norm'] = nn.BatchNorm2d(filters)
            self.conv_blocks.append(block)
            in_channels = filters

        # Trace the spatial dimensions once so the dense stack can be sized,
        # and record the pooling actually applied at each layer (a design may
        # ask for more pooling than the feature map can take).
        self.pool_plan, self.adaptive_pool_size, self.flat_features = self._plan_spatial()

        # --- Dense stack ---------------------------------------------
        self.dense_blocks = nn.ModuleList()
        in_features = self.flat_features

        for spec in self.dense_specs:
            nodes = int(spec['nodes'])
            block = nn.ModuleDict({'linear': nn.Linear(in_features, nodes)})
            if int(spec.get('batch_norm', 0)) == 1:
                block['norm'] = nn.BatchNorm1d(nodes)
            self.dense_blocks.append(block)
            in_features = nodes

        self.classifier = nn.Linear(in_features, num_classes)

    # ------------------------------------------------------------------

    def _plan_spatial(self) -> Tuple[List[int], int, int]:
        """Work out the pooling actually applied, and the flattened size.

        Returns:
            ``(pool_plan, adaptive_pool_size, flat_features)`` where
            ``pool_plan[i]`` is the pooling kernel used after conv layer ``i``
            (0 for none) and ``adaptive_pool_size`` is the side length of a
            final adaptive pool (0 if not needed).
        """
        _, height, width = self.input_shape
        pool_plan: List[int] = []

        for spec in self.conv_specs:
            requested = int(spec.get('max_pool', 0))
            # Pool only when the feature map can actually absorb it; a design
            # asking for 8x8 pooling on a 7x7 map would otherwise fail.
            if requested >= 2 and height >= requested and width >= requested:
                applied = requested
            elif requested >= 2 and min(height, width) >= 2:
                applied = min(height, width, requested)
            else:
                applied = 0

            if applied >= 2:
                height //= applied
                width //= applied
            pool_plan.append(applied)

        channels = int(self.conv_specs[-1]['filters']) if self.conv_specs else self.input_shape[0]
        flat = channels * height * width
        adaptive_pool_size = 0

        if flat > self.max_flat_features:
            # Shrink the spatial dimensions to the largest square that fits
            # under the cap, so the first dense layer stays a sane size.
            target = max(1, int((self.max_flat_features / channels) ** 0.5))
            target = min(target, height, width)

            # The side must divide both spatial dimensions exactly: MPS
            # refuses adaptive pooling when the input is not divisible by the
            # output, and an exact divisor is a cleaner pooling anyway. 1
            # always divides, so this terminates.
            side = next(
                (s for s in range(target, 0, -1) if height % s == 0 and width % s == 0),
                1,
            )
            adaptive_pool_size = side
            flat = channels * side * side

        return pool_plan, adaptive_pool_size, int(flat)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape ``(batch, channels, height, width)``.

        Returns:
            Logits of shape ``(batch, num_classes)``.
        """
        for block, spec, pool in zip(self.conv_blocks, self.conv_specs, self.pool_plan):
            x = block['conv'](x)
            if 'norm' in block:
                x = block['norm'](x)
            x = apply_activation(x, int(spec.get('activation', 3)))

            if pool >= 2:
                x = F.max_pool2d(x, pool)

            dropout = float(spec.get('dropout', -1.0))
            if dropout > 0.0:
                x = F.dropout(x, p=min(dropout, 0.95), training=self.training)

        if self.adaptive_pool_size:
            x = F.adaptive_avg_pool2d(x, self.adaptive_pool_size)

        x = x.flatten(1)

        for block, spec in zip(self.dense_blocks, self.dense_specs):
            x = block['linear'](x)
            if 'norm' in block:
                # BatchNorm1d needs more than one sample; skip it for a
                # single-sample batch rather than raising.
                if not (self.training and x.shape[0] < 2):
                    x = block['norm'](x)
            x = apply_activation(x, int(spec.get('activation', 3)))

            dropout = float(spec.get('dropout', -1.0))
            if dropout > 0.0:
                x = F.dropout(x, p=min(dropout, 0.95), training=self.training)

        return self.classifier(x)

    # ------------------------------------------------------------------

    def get_num_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def estimate_macs(self) -> int:
        """Estimate multiply-accumulate operations for one forward pass.

        Parameter count is a poor guide to how long a CNN takes to train:
        convolutions hold few parameters but do most of the arithmetic. A
        512-filter convolution applied to an unpooled 28x28 map is about
        1.9 GMAC per image per layer, so a design with several of them costs
        orders of magnitude more than its parameter count suggests - enough
        for one such candidate to dominate an entire GA generation.

        Returns:
            Estimated MACs per input image.
        """
        _, height, width = self.input_shape
        channels = self.input_shape[0]
        macs = 0

        for spec, pool in zip(self.conv_specs, self.pool_plan):
            filters = int(spec['filters'])
            # 3x3 kernel over the current map, before this layer's pooling
            macs += height * width * filters * channels * 9
            channels = filters
            if pool >= 2:
                height //= pool
                width //= pool

        in_features = self.flat_features
        for spec in self.dense_specs:
            nodes = int(spec['nodes'])
            macs += in_features * nodes
            in_features = nodes

        macs += in_features * self.num_classes
        return int(macs)

    def describe(self) -> str:
        """One-line summary of the built architecture."""
        conv = "-".join(
            f"{int(s['filters'])}" + (f"p{p}" if p else "")
            for s, p in zip(self.conv_specs, self.pool_plan)
        )
        dense = "-".join(str(int(s['nodes'])) for s in self.dense_specs)
        return (
            f"conv[{conv}] flat={self.flat_features} dense[{dense}] "
            f"params={self.get_num_parameters():,}"
        )
