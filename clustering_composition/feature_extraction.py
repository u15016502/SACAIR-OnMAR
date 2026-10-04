import tensorflow as tf
import keras 
import numpy as np

def apply_resnet50(x_images):
    input_shape_ = x_images[0].shape

    if len(input_shape_) == 2:
        x_images = np.array([[[[pixel] * 3 for pixel in row]for row in img]for img in x_images])

    size = (224, 224)
    x_images = np.array([tf.image.resize(img, size) for img in x_images])
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
    input_shape_ = x_images[0].shape

    if len(input_shape_) == 2:
        x_images = np.array([[[[pixel] * 3 for pixel in row]for row in img]for img in x_images])

    size = (224, 224)
    x_images = np.array([tf.image.resize(img, size) for img in x_images])
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
    input_shape_ = x_images[0].shape

    if len(input_shape_) == 2:
        x_images = np.array([[[[pixel] * 3 for pixel in row]for row in img]for img in x_images])

    size = (224, 224)
    x_images = np.array([tf.image.resize(img, size) for img in x_images])
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
    input_shape_ = x_images[0].shape

    if len(input_shape_) == 2:
        x_images = np.array([[[[pixel] * 3 for pixel in row]for row in img]for img in x_images])
        input_shape_ = x_images[0].shape
    
    size = (224, 224)
    x_images = np.array([tf.image.resize(img, size) for img in x_images])
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
    input_shape_ = x_images[0].shape

    if len(input_shape_) == 2:
        x_images = np.array([[[[pixel] * 3 for pixel in row]for row in img]for img in x_images])

    size = (224, 224)
    x_images = np.array([tf.image.resize(img, size) for img in x_images])
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

        x_trans = TSNE(n_components=nc, learning_rate=learning_rate_, init='random', perplexity=perplexity_, early_exaggeration=early_exaggeration_).fit_transform(x)
        
        if nc < num_features_:
            num_features_ = nc
        
        return x_trans[:,:num_features_]
    except:
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

        pca_ = PCA(n_components=nc)
        pca_.fit(x)
        x_trans = pca_.transform(x)

        if nc < num_features_:
            num_features_ = nc

        return x_trans[:,:num_features_]
    except:
        return x

def apply_sift(x_images):

    if len(x_images[0].shape) == 1:
        return x_images
    
    if x_images[0].shape[1] == 2:
        return x_images

    try:
        sift = cv2.SIFT_create(80)

        def sift_func(sift, x):
            x = np.abs(x)
            kp, des = sift.detectAndCompute(x, None)
            keypoints = np.array(cv2.KeyPoint_convert(kp))[:30]

            if len(keypoints) == 0:
                return np.array([np.zeros((2)) for x in range(0, 30)])
            elif len(keypoints) < 30:
                buff = np.array([np.zeros((2)) for x in range(0, (30 - len(keypoints)))])
                keypoints = np.concatenate((keypoints, buff), axis=0)

            return keypoints
        
        return [sift_func(sift, x) for x in x_images]
    except:
        return x_images

def apply_brief(x_images):

    if len(x_images[0].shape) == 1:
        return x_images
    
    if x_images[0].shape[1] == 2:
        return x_images

    try:
        # Initiate FAST detector
        star = cv2.xfeatures2d.StarDetector_create()
        # Initiate BRIEF extractor
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

        def brief_func(star, brief, x):


            x = np.abs(x)
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
    except:
        return x_images

