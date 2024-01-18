import os
import cv2
import numpy as np
# import cupy as cp  - Import CuPy for GPU computations
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from keras.models import load_model

def extract_hog_features(images):
    features = []
    # For transferring to gpu:
    # images = cp.asarray(images)
    for img in images:
        _, hog_feature = hog(img.reshape((256, 256)), orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), block_norm='L2-Hys', visualize=True)
        features.append(hog_feature.flatten())
    return np.array(features)

def load_and_test_model(model_filename, test_images_directory):
    loaded_model = load_model(model_filename)
    test_images = []
    for file in os.listdir(test_images_directory):
        if file.endswith(('.jpg', '.png')):
            img = cv2.imread(os.path.join(test_images_directory, file), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (256, 256))
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=-1)
            img = np.expand_dims(img, axis=0)

            test_images.append(img)

    test_images = np.array(test_images)
    test_hog_features = extract_hog_features(test_images)

    scaler = StandardScaler()
    test_hog_features_scaled = scaler.fit_transform(test_hog_features)

    predictions = loaded_model.predict(test_hog_features_scaled)

    for i, prediction in enumerate(predictions):
        print(f"Image {i + 1} - Predicted Class: {'Normal' if prediction > 0.5 else 'Abnormal'}")

model_filename = 'ct_ocsvm.joblib'
#normal test = test_images_directory = 'DATASET/TEST/NORMAL'
test_images_directory = 'DATASET/TEST/ABNORMAL/HEMORRHAGE'
load_and_test_model(model_filename, test_images_directory)
