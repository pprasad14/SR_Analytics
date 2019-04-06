# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 11:15:16 2018

@author: Prem Prasad
"""
# import the necessary packages
import numpy as np
import argparse
import cv2

img=cv2.imread("C:\\Users\\Prem Prasad\\Pictures\\test\\test_20.png")
img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

plt.imshow(img_hsv, cmap = 'gray', interpolation = 'bicubic')

# lower mask (0-10)
lower_red = np.array([0,50,50])
upper_red = np.array([10,255,255])
mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

# upper mask (170-180)
lower_red = np.array([170,50,50])
upper_red = np.array([180,255,255])
mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

# join my masks
mask = mask0+mask1

# set my output img to zero everywhere except my mask
output_img = img.copy()
output_img[np.where(mask==0)] = 0

plt.imshow(output_img, cmap = 'gray', interpolation = 'bicubic')

# or your HSV image, which I *believe* is what you want
output_hsv = img_hsv.copy()
output_hsv[np.where(mask==0)] = 0

plt.imshow(output_hsv, cmap = 'gray', interpolation = 'bicubic')
