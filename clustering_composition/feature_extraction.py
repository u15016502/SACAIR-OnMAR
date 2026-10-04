"""
Feature extraction for the clustering composition application.

Each ``apply_*`` function maps a batch of images to the feature vectors that
the clustering components then operate on. The choice between them is the
``feature_extraction`` gene of the design.

Two fixes were needed to make this module run:

* ``apply_pca``, ``apply_tsne``, ``apply_sift`` and ``apply_brief`` used
  ``PCA``, ``TSNE``, ``math`` and ``cv2`` without importing any of them. Each
  is wrapped in ``try/except``, so the resulting ``NameError`` was caught and
  the raw images returned unchanged - PCA and t-SNE silently did nothing at
  all. The imports are now present, and the fallback logs once instead of
  failing quietly.
* TensorFlow and OpenCV are optional (see
  :mod:`clustering_composition.optional_deps`) and are resolved per call, so a
  machine without them can still use the options that do not need them.
"""

import math
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from clustering_composition import optional_deps


def _fallback(extractor: str, exc: BaseException) -> None:
    """Warn that an extractor is returning its input unchanged."""
    warnings.warn(
        f"{extractor} failed ({type(exc).__name__}: {exc}); "
        f"returning the input unchanged, so this design clusters raw pixels.",
        RuntimeWarning,
        stacklevel=3,
    )

def _prepare_rgb_224(x_images, tf, size=(224, 224)):
    """Make a batch of images acceptable to an ImageNet-pretrained network.

    The donor networks all require three channels and a fixed input size.

    Each extractor used to widen greyscale to RGB only when an image was
    two-dimensional ``(H, W)``. The project's loaders produce ``(C, H, W)``,
    which ``run`` lays out as ``(H, W, 1)`` - three-dimensional, so the
    widening was skipped and every one of the five pretrained extractors
    failed with "The input must have 3 channels; Received
    input_shape=(224, 224, 1)". A trailing single channel is now widened too,
    and an alpha-style fourth channel is dropped.

    Args:
        x_images: The image batch.
        tf: The TensorFlow module.
        size: Target spatial size.

    Returns:
        The batch as a float array shaped (N, size[0], size[1], 3).
    """
    images = np.asarray(x_images, dtype=np.float32)

    if images.ndim == 3:
        # (N, H, W): no channel axis at all.
        images = images[..., np.newaxis]

    channels = images.shape[-1]
    if channels == 1:
        images = np.repeat(images, 3, axis=-1)
    elif channels == 2:
        # Keypoint-style features, not images; nothing sensible to resize.
        raise ValueError(
            f"Expected images, got arrays shaped {images.shape[1:]}; a "
            f"pretrained extractor cannot consume keypoint features."
        )
    elif channels > 3:
        images = images[..., :3]

    return np.asarray(tf.image.resize(images, size), dtype=np.float32)


def apply_resnet50(x_images):
    tf, keras = optional_deps.tensorflow()
    x_images = _prepare_rgb_224(x_images, tf)
    input_shape_ = x_images[0].shape

    feature_extractor = tf.keras.applications.ResNet50(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=input_shape_
        )

    preprocess_input = tf.keras.applications.resnet.preprocess_input

    inputs = keras.Input(input_shape_)
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    model = keras.Model(inputs, outputs, name="feature_extractor")

    preds = model.predict(tf.convert_to_tensor(np.array(x_images), dtype=tf.float32), batch_size= 4, verbose=0)

    return preds 


def apply_inceptionv3(x_images):
    tf, keras = optional_deps.tensorflow()
    x_images = _prepare_rgb_224(x_images, tf)
    input_shape_ = x_images[0].shape

    feature_extractor = tf.keras.applications.InceptionV3(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=input_shape_,
        )

    preprocess_input = tf.keras.applications.inception_v3.preprocess_input

    inputs = keras.Input(input_shape_)
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    model = keras.Model(inputs, outputs, name="feature_extractor")

    preds = model.predict(tf.convert_to_tensor(np.array(x_images), dtype=tf.float32), batch_size= 4, verbose=0)

    return preds 


def apply_densenet121(x_images):
    tf, keras = optional_deps.tensorflow()
    x_images = _prepare_rgb_224(x_images, tf)
    input_shape_ = x_images[0].shape

    feature_extractor = tf.keras.applications.DenseNet121(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=input_shape_,
        )

    preprocess_input = tf.keras.applications.densenet.preprocess_input

    inputs = keras.Input(input_shape_)
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    model = keras.Model(inputs, outputs, name="feature_extractor")

    preds = model.predict(tf.convert_to_tensor(np.array(x_images), dtype=tf.float32), batch_size= 4, verbose=0)

    return preds 


def apply_xception(x_images):
    tf, keras = optional_deps.tensorflow()
    x_images = _prepare_rgb_224(x_images, tf)
    input_shape_ = x_images[0].shape

    feature_extractor = tf.keras.applications.Xception(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=input_shape_,
        )

    preprocess_input = tf.keras.applications.xception.preprocess_input

    inputs = keras.Input(input_shape_)
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    model = keras.Model(inputs, outputs, name="feature_extractor")

    preds = model.predict(tf.convert_to_tensor(np.array(x_images), dtype=tf.float32), batch_size=4, verbose=0)

    return preds 


def apply_vgg16(x_images):
    tf, keras = optional_deps.tensorflow()
    x_images = _prepare_rgb_224(x_images, tf)
    input_shape_ = x_images[0].shape

    feature_extractor = tf.keras.applications.VGG16(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=input_shape_,
        )

    preprocess_input = tf.keras.applications.vgg16.preprocess_input

    inputs = keras.Input(input_shape_)
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    model = keras.Model(inputs, outputs, name="feature_extractor")

    preds = model.predict(tf.convert_to_tensor(np.array(x_images), dtype=tf.float32), batch_size=4, verbose=0)

    return preds 

def apply_tsne(x, num_components_, num_features_, learning_rate_, perplexity_, early_exaggeration_):
    try:
        x = np.nan_to_num(np.array(x), nan= 0, neginf=999999)
        nc = 2

        if len(x.shape) == 3 or len(x.shape) == 4:
            if len(x.shape) == 3 and x[0].shape[1] == 2:
                return x

            x = [v.flatten() for v in x]

        if num_components_ == -1:
            nc = len(x[0])

        if num_components_ == 0.5:
            nc = 2

        if num_components_ == 0.3:
            nc = 3

        if len(x[0]) < nc or nc == 0:
            nc = len(x[0])

        # The clamp above bounds nc by the feature count only. t-SNE is also
        # bounded by the sample count, and its default Barnes-Hut solver by 3
        # components; with num_components_ = -1 (how utils calls this) nc
        # became the full 784-dimensional feature count, the call raised, and
        # the fallback below returned raw pixels - so this option never
        # actually ran t-SNE. Bound it by both, and switch to the exact
        # solver when Barnes-Hut cannot do the job.
        x = np.asarray(x, dtype=np.float64)
        n_samples = x.shape[0]
        nc = max(1, min(nc, x.shape[1], n_samples - 1))
        method = 'barnes_hut' if nc <= 3 else 'exact'

        # Perplexity must be smaller than the sample count.
        perplexity_ = max(1.0, min(float(perplexity_), float(n_samples - 1)))

        x_trans = TSNE(n_components=nc, learning_rate=learning_rate_, init='random', perplexity=perplexity_, early_exaggeration=early_exaggeration_, method=method).fit_transform(x)
        
        if nc < num_features_:
            num_features_ = nc
        
        return x_trans[:,:num_features_]
    except Exception as exc:
        _fallback('t-SNE feature extraction', exc)
        return x

def apply_pca(x, num_components_, num_features_):
    try:
        x = np.nan_to_num(np.array(x), nan= 0, neginf=999999)
        x = np.array(x)
        nc = 2

        if len(x.shape) == 3 or len(x.shape) == 4:
            if len(x.shape) == 3 and x[0].shape[1] == 2:
                return x

            x = [v.flatten() for v in x]

        if num_components_ == -1:
            nc = len(x[0])

        if num_components_ == 0.5:
            nc = math.floor(len(x[0]) / 2)

        if num_components_ == 0.3:
            nc = math.floor(len(x[0]) / 3)

        if num_components_ == 2:
            nc = 2
            
        if len(x[0]) < nc or nc == 0:
            nc = len(x[0])

        # PCA yields at most min(n_samples, n_features) components, and the
        # clamp above only bounds nc by n_features. With num_components_ = -1
        # (how utils calls this) nc was the full feature count, PCA raised
        # "n_components=784 must be between 0 and min(n_samples,
        # n_features)=48", and the fallback returned raw pixels - so this
        # option never actually ran PCA either.
        x = np.asarray(x, dtype=np.float64)
        nc = max(1, min(nc, x.shape[0], x.shape[1]))

        pca_ = PCA(n_components=nc)
        pca_.fit(x)
        x_trans = pca_.transform(x)

        if nc < num_features_:
            num_features_ = nc

        return x_trans[:,:num_features_]
    except Exception as exc:
        _fallback('PCA feature extraction', exc)
        return x

def apply_sift(x_images):

    if len(x_images[0].shape) == 1:
        return x_images
    
    if x_images[0].shape[1] == 2:
        return x_images

    cv2 = optional_deps.opencv()

    try:
        sift = cv2.SIFT_create(80)

        def sift_func(sift, x):
            # OpenCV's SIFT takes CV_8U; the loaders hand over float32 on
            # 0-255, which made detectAndCompute reject every image ("image
            # is empty or has incorrect depth (!=CV_8U)") and the fallback
            # return raw pixels. A trailing single channel is squeezed away
            # too, since SIFT wants (H, W) or (H, W, 3).
            x = np.abs(np.asarray(x))
            if x.ndim == 3 and x.shape[-1] == 1:
                x = x[..., 0]
            x = np.clip(x, 0, 255).astype(np.uint8)
            kp, des = sift.detectAndCompute(x, None)
            keypoints = np.array(cv2.KeyPoint_convert(kp))[:30]

            if len(keypoints) == 0:
                return np.array([np.zeros((2)) for x in range(0, 30)])
            elif len(keypoints) < 30:
                buff = np.array([np.zeros((2)) for x in range(0, (30 - len(keypoints)))])
                keypoints = np.concatenate((keypoints, buff), axis=0)

            return keypoints
        
        return [sift_func(sift, x) for x in x_images]
    except Exception as exc:
        _fallback('SIFT feature extraction', exc)
        return x_images

def apply_brief(x_images):

    if len(x_images[0].shape) == 1:
        return x_images
    
    if x_images[0].shape[1] == 2:
        return x_images

    cv2 = optional_deps.opencv_contrib()

    try:
        # Initiate FAST detector
        star = cv2.xfeatures2d.StarDetector_create()
        # Initiate BRIEF extractor
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

        def brief_func(star, brief, x):
            # As for SIFT: 8-bit, and single-channel squeezed to (H, W).
            x = np.abs(np.asarray(x))
            if x.ndim == 3 and x.shape[-1] == 1:
                x = x[..., 0]
            x = np.clip(x, 0, 255).astype(np.uint8)
            # find the keypoints with STAR
            kp = star.detect(x,None)
            # compute the descriptors with BRIEF
            kp, des = brief.compute(x, kp)
            keypoints = np.array(cv2.KeyPoint_convert(kp))[:30]

            if len(keypoints) == 0:
                return np.array([np.zeros((2)) for x in range(0, 30)])
            elif len(keypoints) < 30:
                buff = np.array([np.zeros((2)) for x in range(0, (30 - len(keypoints)))])
                keypoints = np.concatenate((keypoints, buff), axis=0)

            return keypoints
        
        return [brief_func(star, brief, x) for x in x_images]
    except Exception as exc:
        _fallback('BRIEF feature extraction', exc)
        return x_images

