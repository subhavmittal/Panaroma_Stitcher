import numpy as np
import matplotlib.pyplot as plt
from gaussian_smoother import gaussian_smoother
def load_image(image_path):
    # Load RGB image
    rgb_image = plt.imread(image_path)
    # Convert to grayscale
    gray_image = np.dot(rgb_image[...,:3], [0.299, 0.587, 0.114])
    print("Image loaded successfully")
    print("Image shape: ", gray_image.shape)
    # Apply Gaussian smoothing with a kernel size of 5
    smooth_image = gaussian_smoother(gray_image, sigma=5)
    # Show the original and smoothed images side by side
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(gray_image, cmap='gray')
    ax[0].set_title('Grayscale Image')
    ax[1].imshow(smooth_image, cmap='gray')
    ax[1].set_title('Smoothed Image')
    plt.show()
    return smooth_image
load_image(r'C:\Users\Subhav\Desktop\Projects\COL780_Ass2\New Dataset\1\image 0.jpg')