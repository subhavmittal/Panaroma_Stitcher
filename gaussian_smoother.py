import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# Load RGB image
rgb_image = plt.imread(r'C:\Users\Subhav\Desktop\Projects\COL780_Ass2\New Dataset\1\image 0.jpg')

# Convert to grayscale
gray_image = np.dot(rgb_image[...,:3], [0.299, 0.587, 0.114])

# Apply Gaussian smoothing with a kernel size of 5
smooth_image = ndimage.gaussian_filter(gray_image, sigma=4)

# Show the original and smoothed images side by side
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(gray_image, cmap='gray')
ax[0].set_title('Grayscale Image')
ax[1].imshow(smooth_image, cmap='gray')
ax[1].set_title('Smoothed Image')
plt.show()
