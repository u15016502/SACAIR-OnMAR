"""
A generated video dataset, for exercising the video pipeline.

No video dataset loader exists in this repository: ``datasets/image_datasets``
covers MNIST, Fashion-MNIST, CIFAR-10 and CIFAR-100 only, and the datasets the
original command line named (UCF101, HMDB51, LMTD) are tens of gigabytes and
must be downloaded and have their frames extracted before any of this code can
run. That is what left the video application unrunnable rather than merely
unintegrated.

This module generates a small dataset on disk in the exact layout
:class:`video_configuration.dataset.TSNDataSet` reads - one directory of
numbered frames per video, plus ``train.txt`` / ``val.txt`` list files of
``<frame directory> <number of frames> <class index>``. So the pipeline that
runs against it is the same pipeline that runs against UCF101, and verifying
it here verifies the real path rather than a mock of it.

What it is *not* is a substitute for a real benchmark. It is four shapes on a
noisy background, which any competent design should classify easily, so
accuracy on it says the implementation works - not that it is competitive.
Point the application at a real frame-extracted dataset
(``dataset_name='custom'`` with ``train_list`` and ``val_list``) for anything
that is meant to be reported.

What separates the classes, and why
----------------------------------
Each class is a distinct *shape* following a distinct motion. The shape is
what carries the signal; the motion is there so that the segment count and the
keyframe strategy change what the network actually sees.

It is tempting to make motion direction the signal - to have a leftward and a
rightward square whose individual frames are identical - so that only
temporal information could separate them. That dataset would be unlearnable
here, and not because of any defect in this code: every consensus function TSN
offers (average, max, top-k, and a learned weighting over segments) is
*order-invariant*. It aggregates the per-segment scores with no access to
which segment came first, so a rightward and a leftward traversal of the same
path produce the same aggregate. That is a property of Temporal Segment
Networks, which trade temporal ordering for cheap long-range coverage, and a
benchmark that demanded ordering would report chance accuracy for every design
in the space.

So the classes are separable from appearance, which an order-invariant model
can learn, and the motion keeps the temporal genes consequential without being
the only route to the signal.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import json

import numpy as np
from PIL import Image


# Each class pairs a shape - the discriminative signal - with a motion, given
# as a (dx, dy) velocity in pixels per frame plus a per-frame radius change.
MOTION_CLASSES: List[Dict[str, Any]] = [
    {'name': 'disc', 'shape': 'disc', 'dx': 1.0, 'dy': 0.0, 'dr': 0.0},
    {'name': 'square', 'shape': 'square', 'dx': -1.0, 'dy': 0.0, 'dr': 0.0},
    {'name': 'ring', 'shape': 'ring', 'dx': 0.0, 'dy': 1.0, 'dr': 0.0},
    {'name': 'cross', 'shape': 'cross', 'dx': 0.0, 'dy': -1.0, 'dr': 0.0},
    {'name': 'disc_grow', 'shape': 'disc', 'dx': 0.0, 'dy': 0.0, 'dr': 0.6},
    {'name': 'square_static', 'shape': 'square', 'dx': 0.0, 'dy': 0.0, 'dr': 0.0},
]


def _shape_mask(
    shape: str,
    size: int,
    centre: Tuple[float, float],
    radius: float,
) -> np.ndarray:
    """A boolean mask of one shape, centred and sized as given.

    Args:
        shape: 'disc', 'square', 'ring' or 'cross'.
        size: Frame edge length in pixels.
        centre: Shape centre, (x, y).
        radius: Shape radius (half-width for the square and cross).
        
    Returns:
        A (size, size) boolean mask.

    Raises:
        ValueError: If the shape is not one of the four.
    """
    y_grid, x_grid = np.ogrid[:size, :size]
    dx = x_grid - centre[0]
    dy = y_grid - centre[1]

    if shape == 'disc':
        return dx ** 2 + dy ** 2 <= radius ** 2

    if shape == 'square':
        return (np.abs(dx) <= radius) & (np.abs(dy) <= radius)

    if shape == 'ring':
        distance_squared = dx ** 2 + dy ** 2
        inner = (radius * 0.55) ** 2
        return (distance_squared <= radius ** 2) & (distance_squared >= inner)

    if shape == 'cross':
        arm = max(1.0, radius * 0.35)
        return (
            ((np.abs(dx) <= radius) & (np.abs(dy) <= arm))
            | ((np.abs(dy) <= radius) & (np.abs(dx) <= arm))
        )

    raise ValueError(f"Unknown shape {shape!r}")


def _render_frame(
    shape: str,
    size: int,
    centre: Tuple[float, float],
    radius: float,
    brightness: int,
    rng: np.random.Generator,
    noise: float,
) -> Image.Image:
    """Render one frame: a bright shape on a noisy background.

    Args:
        shape: Which shape to draw.
        size: Frame edge length in pixels.
        centre: Shape centre, (x, y).
        radius: Shape radius in pixels.
        brightness: Shape intensity, 0-255.
        rng: Generator for the background noise.
        noise: Standard deviation of the background noise.

    Returns:
        The frame as an RGB image.
    """
    frame = rng.normal(40.0, noise, size=(size, size)).clip(0, 255)
    frame[_shape_mask(shape, size, centre, radius)] = brightness

    # Three identical channels: the ImageNet base models expect RGB.
    rgb = np.repeat(frame[:, :, np.newaxis].astype(np.uint8), 3, axis=2)
    return Image.fromarray(rgb, mode='RGB')


def generate(
    root: str,
    num_classes: int = 4,
    videos_per_class: int = 8,
    num_frames: int = 16,
    frame_size: int = 256,
    val_fraction: float = 0.25,
    noise: float = 12.0,
    random_seed: int = 42,
    image_tmpl: str = 'img_{:05d}.jpg',
    force: bool = False,
) -> Dict[str, object]:
    """Generate the dataset, or reuse it if it is already on disk.

    Args:
        root: Directory to generate into.
        num_classes: How many shape classes to use, up to
            ``len(MOTION_CLASSES)``.
        videos_per_class: Videos generated per class.
        num_frames: Frames per video.
        frame_size: Frame edge length. Must be at least the base model's
            input size (224 for ResNet and VGG), since the TSN augmentation
            crops rather than upscales.
        val_fraction: Fraction of each class held out for validation.
        noise: Background noise standard deviation. Some noise is needed or
            the task is trivially separable and every design scores 1.0.
        random_seed: Seed for motion jitter and noise.
        image_tmpl: Frame filename template, matching ``TSNDataSet``.
        force: Regenerate even if the dataset is already present.

    Returns:
        A manifest dictionary with the list-file paths, the class names and
        the dataset's shape.

    Raises:
        ValueError: If more classes are requested than are defined, or the
            frame size is too small for a 224-pixel crop.
    """
    if num_classes > len(MOTION_CLASSES):
        raise ValueError(
            f"Only {len(MOTION_CLASSES)} classes are defined; "
            f"{num_classes} were requested."
        )
    if frame_size < 224:
        raise ValueError(
            f"frame_size must be at least 224 (the base models' input size); "
            f"got {frame_size}."
        )

    root_path = Path(root)
    manifest_path = root_path / 'manifest.json'

    manifest = {
        'num_classes': num_classes,
        'videos_per_class': videos_per_class,
        'num_frames': num_frames,
        'frame_size': frame_size,
        'val_fraction': val_fraction,
        'noise': noise,
        'random_seed': random_seed,
        'image_tmpl': image_tmpl,
        'class_names': [MOTION_CLASSES[i]['name'] for i in range(num_classes)],
        'train_list': str(root_path / 'train.txt'),
        'val_list': str(root_path / 'val.txt'),
        'root': str(root_path),
    }

    # Reuse an identical dataset rather than regenerating thousands of frames
    # on every run.
    if manifest_path.exists() and not force:
        existing = json.loads(manifest_path.read_text())
        comparable = {k: v for k, v in manifest.items() if k not in ('root', 'train_list', 'val_list')}
        if all(existing.get(k) == v for k, v in comparable.items()):
            return existing

    rng = np.random.default_rng(random_seed)
    root_path.mkdir(parents=True, exist_ok=True)

    train_lines: List[str] = []
    val_lines: List[str] = []
    num_val = max(1, int(round(videos_per_class * val_fraction)))

    for class_index in range(num_classes):
        motion = MOTION_CLASSES[class_index]

        for video_index in range(videos_per_class):
            video_dir = root_path / f"{motion['name']}_{video_index:03d}"
            video_dir.mkdir(parents=True, exist_ok=True)

            # Start position and appearance vary per video, so a design cannot
            # separate the classes on a static cue.
            margin = frame_size * 0.25
            centre = [
                rng.uniform(margin, frame_size - margin),
                rng.uniform(margin, frame_size - margin),
            ]
            radius = rng.uniform(frame_size * 0.07, frame_size * 0.12)
            brightness = int(rng.integers(190, 256))
            speed = frame_size / float(num_frames) * rng.uniform(0.6, 1.0)

            for frame_index in range(1, num_frames + 1):
                frame = _render_frame(
                    motion['shape'],
                    frame_size,
                    (centre[0], centre[1]),
                    radius,
                    brightness,
                    rng,
                    noise,
                )
                frame.save(video_dir / image_tmpl.format(frame_index), quality=92)

                centre[0] = float(np.clip(
                    centre[0] + motion['dx'] * speed, radius, frame_size - radius
                ))
                centre[1] = float(np.clip(
                    centre[1] + motion['dy'] * speed, radius, frame_size - radius
                ))
                radius = float(np.clip(
                    radius + motion['dr'], frame_size * 0.03, frame_size * 0.2
                ))

            line = f"{video_dir} {num_frames} {class_index}"
            if video_index < num_val:
                val_lines.append(line)
            else:
                train_lines.append(line)

    (root_path / 'train.txt').write_text('\n'.join(train_lines) + '\n')
    (root_path / 'val.txt').write_text('\n'.join(val_lines) + '\n')
    manifest['num_train'] = len(train_lines)
    manifest['num_val'] = len(val_lines)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return manifest
