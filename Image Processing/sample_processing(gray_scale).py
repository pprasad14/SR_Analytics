# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 10:13:22 2018

@author: Prem Prasad
"""

import numpy as np
#import math
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("C:\\Users\\Prem Prasad\\Pictures\\test\\test_7.png")

#converting a RGB image to GrayScale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_mask = img_gray.copy()

height = img.shape[0]
width = img.shape[1]

#threshold for black
threshold = 1 #to see image of different thresholds, load the image again

#create mask for black region
for i in np.arange(height):
    for j in np.arange(width):
        a = img_mask.item(i,j)
        if a > threshold:
            b = 255
        else: 
            b = 0
        img_mask.itemset((i,j),b)

#threshold for center_white disc
threshold = 160 #to see image of different thresholds, load the image again

#create mask for white region
for i in np.arange(height):
    for j in np.arange(width):
        a = img_mask.item(i,j)
        if a > threshold:
            b = 0
        else: 
            b = 255
        img_mask.itemset((i,j),b)

#plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.imshow(img_gray, cmap = 'gray', interpolation = 'bicubic')
plt.imshow(img_mask, cmap = 'gray', interpolation = 'bicubic')

min = 255
max = 0

#finding max and min values excluding mask positions
for i in np.arange(height):
    for j in np.arange(width):
        if img_mask.item(i,j) == 255:
            a = img_gray.item(i,j)
            if a > max:
                max = a
            if a < min:
                    min = a
        
            
for i in np.arange(height):
    for j in np.arange(width):
        if img_mask.item(i,j) == 255:
            a = img_gray.item(i,j)
            b = float(a-min)/(max-min) * 255
            img_gray.itemset((i,j),b)

#updated gray image
plt.imshow(img_gray, cmap = 'gray', interpolation = 'bicubic')

cv2.imwrite('C:\\Users\\Prem Prasad\\Pictures\\test\\test_7_masked',img_gray)



