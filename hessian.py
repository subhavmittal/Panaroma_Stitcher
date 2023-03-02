import numpy as np
from scipy.ndimage import convolve
from image_load import load_image
import matplotlib.pyplot as plt
class ImageHessian:
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    def __init__(self, image_path):
        # Load input image
        self.image = load_image(image_path)
        #Initialize Hessian matrix and Response matrix
        self.H = np.zeros((self.image.shape[0],self.image.shape[1],2,2),dtype=np.float32)
        self.R = np.zeros((self.image.shape[0],self.image.shape[1]),dtype=np.float32)
        self.corners = np.zeros((self.image.shape[0],self.image.shape[1]),dtype=np.uint8)
    def compute_hessian(self):
        dx = convolve(self.image,self.sobel_x)
        dy = convolve(self.image,self.sobel_y)
        dxx = convolve(dx,self.sobel_x)
        dxy = convolve(dx,self.sobel_y)
        dyy = convolve(dy,self.sobel_y)
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                self.H[i,j,0,0] = dxx[i,j]
                self.H[i,j,0,1] = dxy[i,j]
                self.H[i,j,1,0] = dxy[i,j]
                self.H[i,j,1,1] = dyy[i,j]
        print("Hessian matrix computed successfully")        
    def compute_response(self):
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                self.R[i,j] = np.linalg.det(self.H[i,j]) - 0.04*(np.trace(self.H[i,j]))**2
        print("Response matrix computed successfully")        
    def find_corners(self, threshold):
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                if self.R[i,j] > threshold:
                    self.corners[i,j] = 1
        print("Corner Candidates found successfully")
        print("Number of corner candidates: ", np.sum(self.corners))            
    def non_max_suppression(self, window_size):
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                if self.corners[i,j] == 1:
                    for k in range(-window_size,window_size+1):
                        for l in range(-window_size,window_size+1):
                            if i+k >= 0 and i+k < self.image.shape[0] and j+l >= 0 and j+l < self.image.shape[1]:
                                if self.R[i+k,j+l] > self.R[i,j]:
                                    self.corners[i,j] = 0
        print("Non-maximum suppression done successfully")
        print("Number of corners: ", np.sum(self.corners))                            
    def corner_detected_image(self):
        corner_image = np.zeros((self.image.shape[0],self.image.shape[1],3),dtype=np.uint8)
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                if self.corners[i,j] == 1:
                    for k in range(-10,11):
                        for l in range(-10,11):
                            if i+k >= 0 and i+k < self.image.shape[0] and j+l >= 0 and j+l < self.image.shape[1]:
                                corner_image[i+k,j+l,0] = 255
                else:
                    corner_image[i,j,0] = self.image[i,j]  
                    corner_image[i,j,1] = self.image[i,j]
                    corner_image[i,j,2] = self.image[i,j]              
        return corner_image
test = ImageHessian(r"4535754.jpg")
test.compute_hessian()
test.compute_response()
test.find_corners(500000)
test.non_max_suppression(5)
my_img = test.corner_detected_image()
plt.imshow(my_img)
plt.show()

                                                                
                            