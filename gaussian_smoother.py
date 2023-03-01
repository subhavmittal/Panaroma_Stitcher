# Function to smooth an image with a Gaussian filter
from scipy import ndimage
def gaussian_smoother(image, sigma=1):
    """Smooth the image with a Gaussian filter"""
    smooth_image = ndimage.gaussian_filter(image, sigma=sigma)
    return smooth_image
