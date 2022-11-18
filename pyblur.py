from matplotlib import pyplot as plt
import cv2
import math
import numpy as np

def gaussian_kernel(size, sigma, two_d=True):
    'returns a one-dimensional gaussian kernel if two_d is False, otherwise 2d'
    if two_d:
        kernel = np.fromfunction(lambda x, y: (1/(2*math.pi*sigma**2)) * math.e ** ((-1*((x-(size-1)/2)**2+(y-(size-1)/2)**2))/(2*sigma**2)), (size, size))
    else:
        kernel = np.fromfunction(lambda x: math.e ** ((-1*(x-(size-1)/2)**2) / (2*sigma**2)), (size,))
    return kernel / np.sum(kernel)

def gaussian_blur(img, k_size, k_sigma):
    'takes a greyscale image in the form of a numpy array and blurs it with a kernel of size k-size and sigma `k_sigma`'
    kernel = gaussian_kernel(k_size, k_sigma, False)
    gaus_x = np.zeros((img.shape[0], img.shape[1] - k_size + 1, 3), dtype='float64')
    for i, v in enumerate(kernel):
        gaus_x += v * img[:, i : img.shape[1] - k_size + i + 1]
    gaus_y = np.zeros((gaus_x.shape[0] - k_size + 1, gaus_x.shape[1], 3))
    for i, v in enumerate(kernel):
        gaus_y += v * gaus_x[i : img.shape[0]  - k_size + i + 1]
    return gaus_y

ksize = 91
sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
kernel = gaussian_kernel(ksize, sigma)


img = cv2.imread('/Users/ramsddc1/Documents/Projects/pii_gui/imgs/IMG_0220.JPG')
imgCopy = img.copy()

dimg = gaussian_blur(imgCopy, ksize, sigma)

cv2.imwrite("/Users/ramsddc1/Documents/Projects/pii_gui/output/output.jpg",  dimg)
