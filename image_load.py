import numpy as np
import matplotlib.pyplot as plt
def load_image(image_path):
    # Load RGB image
    rgb_image = plt.imread(image_path)
    # Convert to grayscale
    gray_image = np.dot(rgb_image[...,:3], [0.299, 0.587, 0.114])
    print("Image loaded successfully")
    print("Image shape: ", gray_image.shape)
    return gray_image
load_image(r'C:\Users\Subhav\Desktop\Projects\COL780_Ass2\New Dataset\1\image 0.jpg')