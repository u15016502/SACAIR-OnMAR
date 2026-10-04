"""
Dataset loaders.

Trimmed for this bundle: only the image loader is exported, because that is
all the CNN application needs and it keeps the dependency list to
requirements-min.txt. The full repository's version also exports
TextDatasetLoader (needs pandas) and SegmentationDataLoader (needs
scikit-image), for the Fuzzy ART and segmentation applications.
"""

from datasets.image_datasets import ImageDatasetLoader

__all__ = ['ImageDatasetLoader']
