"""
Pretrained initialisation for configurable CNNs.

A design in this application is an arbitrary stack of 3x3 convolutions whose
filter counts come from the design space, so no pretrained network matches it
exactly. What *can* be reused is the filters themselves: a donor network's 3x3
convolution kernels are copied into each layer of the design, sliced to the
shape that layer actually needs.

Which parts get pretrained values
---------------------------------
  * first convolution  - the donor's input-level 7x7 kernels, centre-cropped
    to 3x3. These are the edge and colour-blob detectors, and they are the
    part of a pretrained network that transfers most reliably.
  * later convolutions - the donor's 3x3 kernels from a block of similar
    width, sliced to the design's filter count.
  * everything else    - left at PyTorch's default initialisation. Batch-norm
    statistics belong to the donor's activation scales, not the design's, and
    the dense layers cannot be matched at all: their input size depends on the
    flattened feature-map size, which is a property of the design and the
    dataset rather than of the donor.

Slicing a pretrained tensor is not the same as using the donor network, and
the result is an *initialisation*, not a trained model. It is reported as a
fraction so a run can state how much of it was actually pretrained.
"""

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn


# Donor networks worth using here hold most of their representation in 3x3
# convolutions. ResNet-18 is the default because it is small (~45MB) and every
# convolution after its stem is 3x3.
SUPPORTED_DONORS = ('resnet18', 'resnet34', 'vgg11_bn', 'vgg16')


class PretrainedInitialiser:
    """Copies a donor network's convolution kernels into a configurable CNN."""

    def __init__(self, donor: str = 'resnet18', verbose: bool = True):
        """
        Args:
            donor: Donor architecture; one of SUPPORTED_DONORS.
            verbose: Print what was initialised, and why anything was skipped.
        """
        if donor not in SUPPORTED_DONORS:
            raise ValueError(
                f"Unknown donor '{donor}'. Choose from {list(SUPPORTED_DONORS)}"
            )

        self.donor_name = donor
        self.verbose = verbose
        self._stem_kernel: Optional[torch.Tensor] = None
        self._conv_kernels: Optional[List[torch.Tensor]] = None
        self._load_failed = False

    # ------------------------------------------------------------------

    def _load_donor(self) -> bool:
        """Fetch the donor's kernels, once. Returns False if unavailable.

        Pretrained weights are downloaded on first use, so this can fail on an
        offline machine (a cluster compute node, typically). That is not fatal:
        the design still builds, just with default initialisation, and the
        caller is told.
        """
        if self._conv_kernels is not None:
            return True
        if self._load_failed:
            return False

        try:
            from torchvision import models

            builders = {
                'resnet18': models.resnet18,
                'resnet34': models.resnet34,
                'vgg11_bn': models.vgg11_bn,
                'vgg16': models.vgg16,
            }
            donor = builders[self.donor_name](weights='IMAGENET1K_V1')

        except Exception as error:  # offline, or torchvision too old
            self._load_failed = True
            if self.verbose:
                print(f"    Pretrained donor '{self.donor_name}' unavailable "
                      f"({type(error).__name__}: {error}); "
                      f"falling back to default initialisation")
            return False

        stem = None
        kernels: List[torch.Tensor] = []

        for module in donor.modules():
            if not isinstance(module, nn.Conv2d):
                continue
            weight = module.weight.detach().clone()
            kernel_h, kernel_w = weight.shape[2], weight.shape[3]

            # The stem is the first convolution that reads the image itself.
            if stem is None and weight.shape[1] in (1, 3):
                stem = self._centre_crop(weight, 3)
                if kernel_h == 3 and kernel_w == 3:
                    kernels.append(weight)
                continue

            if kernel_h == 3 and kernel_w == 3:
                kernels.append(weight)

        if not kernels and stem is None:
            self._load_failed = True
            if self.verbose:
                print(f"    Donor '{self.donor_name}' exposed no usable 3x3 "
                      f"kernels; falling back to default initialisation")
            return False

        self._stem_kernel = stem
        # Widest first, so a layer can always be served by slicing down.
        self._conv_kernels = sorted(kernels, key=lambda w: w.shape[0], reverse=True)
        return True

    @staticmethod
    def _centre_crop(weight: torch.Tensor, size: int) -> torch.Tensor:
        """Centre-crop a convolution kernel to ``size`` x ``size``."""
        _, _, height, width = weight.shape
        if height <= size and width <= size:
            return weight
        top = (height - size) // 2
        left = (width - size) // 2
        return weight[:, :, top:top + size, left:left + size].clone()

    # ------------------------------------------------------------------

    def _fit_kernel(self, donor: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        """Reshape a donor kernel to a target convolution's exact shape.

        Args:
            donor: Donor weight, ``(D_out, D_in, 3, 3)``.
            target_shape: Target weight shape, ``(T_out, T_in, 3, 3)``.

        Returns:
            A tensor of ``target_shape`` holding donor values.
        """
        target_out, target_in = int(target_shape[0]), int(target_shape[1])
        weight = donor

        # --- input channels ---
        donor_in = weight.shape[1]
        if target_in == donor_in:
            pass
        elif target_in == 1:
            # Greyscale: average the donor's input channels, the standard way
            # to carry an RGB filter over to a single-channel image.
            weight = weight.mean(dim=1, keepdim=True)
        elif target_in < donor_in:
            weight = weight[:, :target_in]
        else:
            # Wider than the donor: repeat its channels and rescale, so the
            # layer's output magnitude stays comparable.
            repeats = (target_in + donor_in - 1) // donor_in
            weight = weight.repeat(1, repeats, 1, 1)[:, :target_in]
            weight = weight * (donor_in / target_in)

        # --- output channels ---
        donor_out = weight.shape[0]
        if target_out == donor_out:
            pass
        elif target_out < donor_out:
            weight = weight[:target_out]
        else:
            repeats = (target_out + donor_out - 1) // donor_out
            weight = weight.repeat(repeats, 1, 1, 1)[:target_out]

        return weight.contiguous()

    def _pick_donor(self, target_shape: torch.Size) -> Optional[torch.Tensor]:
        """Choose the donor kernel closest in width to the target."""
        if not self._conv_kernels:
            return None

        target_out = int(target_shape[0])
        # Prefer the narrowest donor that still covers the target, so values
        # are sliced down rather than tiled up.
        covering = [w for w in self._conv_kernels if w.shape[0] >= target_out]
        if covering:
            return min(covering, key=lambda w: w.shape[0])
        return max(self._conv_kernels, key=lambda w: w.shape[0])

    # ------------------------------------------------------------------

    def initialise(self, model: nn.Module, announce: Optional[bool] = None) -> Dict[str, Any]:
        """Copy donor kernels into a model's convolution layers, in place.

        Args:
            model: A ConfigurableCNN (or any module with a ``conv_blocks``
                ModuleList of blocks each holding a ``conv``).
            announce: Override the instance's verbosity for this call. Used to
                stay quiet while seeding the GA's candidate models, which
                happens several times per generation.

        Returns:
            Stats: whether the donor loaded, how many layers were initialised,
            and the fraction of the model's parameters that came from it.
        """
        stats: Dict[str, Any] = {
            'donor': self.donor_name,
            'available': False,
            'conv_layers_initialised': 0,
            'conv_layers_total': 0,
            'pretrained_parameter_fraction': 0.0,
        }

        conv_blocks = getattr(model, 'conv_blocks', None)
        if conv_blocks is None:
            return stats

        stats['conv_layers_total'] = len(conv_blocks)

        verbose = self.verbose if announce is None else announce
        previous_verbose, self.verbose = self.verbose, verbose
        try:
            donor_ready = self._load_donor()
        finally:
            self.verbose = previous_verbose

        if not donor_ready:
            return stats
        stats['available'] = True

        initialised_parameters = 0

        with torch.no_grad():
            for index, block in enumerate(conv_blocks):
                conv = block['conv']
                target_shape = conv.weight.shape

                # The first layer reads the image, so the donor's own
                # input-level kernels are the right source for it.
                if index == 0 and self._stem_kernel is not None:
                    donor = self._stem_kernel
                else:
                    donor = self._pick_donor(target_shape)

                if donor is None:
                    continue

                fitted = self._fit_kernel(donor.to(conv.weight.device), target_shape)
                conv.weight.copy_(fitted)
                if conv.bias is not None:
                    conv.bias.zero_()

                initialised_parameters += conv.weight.numel()
                stats['conv_layers_initialised'] += 1

        total_parameters = sum(p.numel() for p in model.parameters())
        if total_parameters:
            stats['pretrained_parameter_fraction'] = initialised_parameters / total_parameters

        if verbose:
            print(
                f"    Pretrained init from {self.donor_name}: "
                f"{stats['conv_layers_initialised']}/{stats['conv_layers_total']} "
                f"conv layers, "
                f"{stats['pretrained_parameter_fraction'] * 100:.1f}% of parameters "
                f"(dense layers and batch-norm left at default init)"
            )

        return stats
