"""
Frame-folder video dataset for Temporal Segment Networks.

A video is a directory of extracted frames plus a line in a list file::

    <frame directory> <number of frames> <class index>

which is the layout the TSN reference implementation uses for UCF101 and
HMDB51. :mod:`video_configuration.synthetic` generates a small dataset in
exactly this layout, so the pipeline can be exercised without one of those
downloads.

Changes from the recovered implementation
-----------------------------------------
* Which frames are read is now chosen by the design's
  ``keyframe_extraction`` gene (see :mod:`video_configuration.keyframes`)
  rather than by training mode. The three hard-coded samplers remain
  reachable: strategy 1 is the old ``_sample_indices``, strategy 2 the old
  ``_get_val_indices`` / ``_get_test_indices``.
* Sampling takes a seeded generator, so a run is reproducible.
* ``_load_image`` reports the missing path when a frame cannot be read,
  instead of raising a bare ``FileNotFoundError`` from deep in PIL.
"""

import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint

from video_configuration.keyframes import sample_indices

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False,
                 keyframe_strategy=None, random_seed=42):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        # None keeps the original mode-driven behaviour; an integer selects
        # one of the five strategies regardless of mode.
        self.keyframe_strategy = keyframe_strategy
        self.rng = np.random.default_rng(random_seed)

        if self.modality == 'RGBDiff':
            self.new_length += 1# Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            path = os.path.join(directory, self.image_tmpl.format(idx))
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Frame {idx} of '{directory}' is missing (looked for "
                    f"'{path}'). Check that the list file's frame count "
                    f"matches what is on disk and that image_tmpl "
                    f"('{self.image_tmpl}') matches the filenames."
                )
            return [Image.open(path).convert('RGB')]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(directory, self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(directory, self.image_tmpl.format('y', idx))).convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """

        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):

        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]

        if self.keyframe_strategy is not None:
            segment_indices = sample_indices(
                num_frames=record.num_frames,
                num_segments=self.num_segments,
                new_length=self.new_length,
                strategy=self.keyframe_strategy,
                rng=self.rng,
            )
        elif not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)
