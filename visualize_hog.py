import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog

def visualize_hog(image_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Extract HOG features
    _, hog_image = hog(img, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm='L2-Hys', visualize=True)

    # Display the original image
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')

    # Display the HOG features
    plt.subplot(1, 2, 2)
    plt.imshow(hog_image, cmap='gray')
    plt.title('HOG Features')

    plt.show()

# Example usage
image_path = 'DATASET/TEST/ABNORMAL/HEMORRHAGE/2_0_107.jpg'
visualize_hog(image_path)
