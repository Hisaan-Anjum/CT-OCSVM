import os
import numpy as np
import cv2
# import cupy as cp  - Import CuPy for GPU computations
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from keras.preprocessing.image import ImageDataGenerator
import joblib

def augment_images(directory, augmentation_factor):
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    image_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(('.jpg', '.png'))]

    augmented_images = []

    for image_file in image_files:
        img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256, 256))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)

        # Generate augmented images
        for _ in range(augmentation_factor):
            augmented_img = datagen.flow(img).next()[0]
            augmented_images.append(augmented_img)

    return np.array(augmented_images)

def extract_hog_features(images, batch_size=32):
    features = []
    for i in range(0, len(images), batch_size):
        batch_imgs = images[i:i + batch_size]
        batch_features = []
        # Transfer images to GPU memory using CuPy
        # batch_imgs = cp.asarray(batch_imgs)
        for img in batch_imgs:
            _, hog_feature = hog(img.reshape((256, 256)), orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), block_norm='L2-Hys', visualize=True)
            batch_features.append(hog_feature.flatten())

        features.extend(batch_features)

    return np.array(features)

def train_one_class_svc(X_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    y_train = np.ones(X_train_scaled.shape[0])

    svm_classifier = SVC(kernel='linear', class_weight='balanced')
    svm_classifier.fit(X_train_scaled, y_train)

    return svm_classifier

def save_classifier(classifier, filename='ct_ocsvm.joblib'):
    joblib.dump(classifier, filename)
    print(f'Trained classifier saved to {filename}')

if __name__ == "__main__":
    normal_images_directory = 'DATASET/TRAIN/NORMAL'
    augmentation_factor = 5
    batch_size = 10
    normal_augmented = augment_images(normal_images_directory, augmentation_factor)

    print("Number of augmented images:", len(normal_augmented))
    X_train, _, _, _ = train_test_split(normal_augmented, np.zeros(normal_augmented.shape[0]), test_size=0.2, random_state=42)

    X_train_hog = extract_hog_features(X_train, batch_size)

    one_class_svc_classifier = train_one_class_svc(X_train_hog)

    save_classifier(one_class_svc_classifier, filename='ct_ocsvm.joblib')

    print('Trained & saved!')
