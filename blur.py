import numpy as np
import cv2 
import math
from matplotlib import pyplot as plt


#create 1D kernel with getGaussianKernel
ksize = 90
sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8 
kernel = cv2.getGaussianKernel(ksize, sigma) 

#apply 1D kernel in both the x and y direction

#want center pixel in kernel to get closer to one and the other pixels to get closer to 0 as you fade out from the center

fade_sections = 10
section_percentage = .05 #for one side
center_pixel = kernel[math.floor(ksize/2)]

img = cv2.imread('/Users/ramsddc1/Documents/Projects/pii_gui/imgs/IMG_0220.JPG')
imgOriginal = img.copy()
imgCopy = img.copy()

hcut_sum = 0
wcut_sum = 0
next_section = imgOriginal
for i in range(fade_sections):
    dimensions = next_section.shape
    hcut = math.floor(section_percentage * dimensions[0])
    wcut = math.floor(section_percentage * dimensions[1])

    #if we are running out of pixels after making new sections
    if hcut <= ksize or wcut <= ksize:
        break
    hcut_sum += hcut
    wcut_sum += wcut

    hbuffer = hcut_sum - ksize
    wbuffer = wcut_sum - ksize

    next_section = imgCopy[hbuffer:-hbuffer, wbuffer:-wbuffer]
    dimg = cv2.sepFilter2D(next_section.copy(),-1,kernel,kernel)
    imgCopy[hcut_sum:-hcut_sum, wcut_sum:-wcut_sum] = dimg[ksize: -ksize, ksize: -ksize]

cv2.imwrite("/Users/ramsddc1/Documents/Projects/pii_gui/output/output.jpg",  imgCopy)
