import numpy as np
import cv2 
import math
from matplotlib import pyplot as plt


def gaussian_filter(kernel_size, sigma):
    if (kernel_size % 2) == 0:
        raise ValueError("kernel_size cannot be even")

    kernel = np.ones((kernel_size,kernel_size), np.float32)

    for y in range(len(kernel)):
        for x in range(len(kernel[y])):
            exponent = (x**2 + y**2) / (2 * sigma**2)
            cell_value = (1 / (2 * np.pi * sigma**2) ) * (1 / (math.e**exponent))
            kernel[y][x] = cell_value
    
    return kernel
kernel = gaussian_filter(13, .5)

breakpoint()







img = cv2.imread('/Users/ramsddc1/Documents/Projects/pii_gui/imgs/IMG_0220.JPG')
kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(img,-1,kernel)
cv2.imwrite("/Users/ramsddc1/Documents/Projects/pii_gui/output/output.jpg", dst)
