# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:30:43 2018

@author: Prem Prasad
"""
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt

########################################
#invert

img = cv2.imread("C:\\Users\\Prem Prasad\\Pictures\\test\\test_7.png",0)
#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

height = img.shape[0]
width = img.shape[1]

max_intensity = 255

for i in np.arange(height):
    for j in np.arange(width):
        a = img.item(i,j)
        b = max_intensity - a
        img.itemset((i,j),b)
        
cv2.imwrite('C:\\Users\\Prem Prasad\\Pictures\\test\\test_7_grey.png',img)

plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

########################################
#threshold

img = cv2.imread("C:\\Users\\Prem Prasad\\Pictures\\test\\test_7.png",0)

height = img.shape[0]
width = img.shape[1]

threshold = 135 #to see image of different thresholds, load the image again

for i in np.arange(height):
    for j in np.arange(width):
        a = img.item(i,j)
        if a > threshold:
            b = 255
        else: 
            b = 0
        img.itemset((i,j),b)

        
cv2.imwrite('C:\\Users\\Prem Prasad\\Pictures\\test\\test_7_thresh135.png',img)

plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

########################################
#automatic contrast

img = cv2.imread("C:\\Users\\Prem Prasad\\Pictures\\test\\test_7.png",0)

height = img.shape[0]
width = img.shape[1]

min = 255
max = 0

for i in np.arange(height):
    for j in np.arange(width):
        a = img.item(i,j)
        if a > max:
            max = a
        if a < min:
            min = a
            
for i in np.arange(height):
    a = img.item(i,j)
    b = float(a-min)/(max-min) * 255
    img.itemset((i,j),b)
    
cv2.imwrite('C:\\Users\\Prem Prasad\\Pictures\\test\\test_7_auto_contrast.png',img)

plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')

########################################
#modified automatic contrast

# not able to import these two packages
#from numpy import histogram as h
#from numpy import cumulative_histogram as ch

img = cv2.imread("C:\\Users\\Prem Prasad\\Pictures\\test\\test_7.png",0)

height = img.shape[0]
width = img.shape[1]
pixels = height * width

# histogram code: (present in numpy)
def histogram(img):
    h = img.shape[0]
    w = img.shape[1]
    
    histo = np.zeros((256))
    
    for i in np.arange(h):
        for j in np.arange(w):
            a = img.item(i,j)
            histo[a] = histo[a] + 1
    return histo

hist = histogram(img)
plt.plot(hist) #to visualize histogram

#cumulative histogram code:
def cumulative_histogram(hist):
    c_hist = hist.copy()
    
    for i in np.arange(1,256):
        c_hist[i]=c_hist[i-1]+c_hist[i]
        
    return c_hist

cum_hist = cumulative_histogram(hist)
plt.plot(cum_hist) # to visualize cumulative histogram

p = 0.005

a_low = 0
for i in np.arange(256):
    if cum_hist[i]>= pixels * p:
        a_low = i
        break

a_high = 255
for i in np.arange(255,-1,-1):
    if cum_hist[i]<= pixels * (1 - p):
        a_high = i
        break

for i in np.arange(height):
    for j in np.arange(width):
        a = img.item(i,j)
        b = 0
        if a <= a_low:
            b = 0
        elif a >= a_high:
            b = 255
        else:
            b = float(a - a_low)/ (a_high - a_low) * 255
        img.itemset((i,j),b)

plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')

cv2.imwrite('C:\\Users\\Prem Prasad\\Pictures\\test\\test_7_mod_auto_contrast.png',img)

            
#cv2.imwrite("C:\\Users\\Prem Prasad\\Pictures\\retina\\image001_mod_automatic.png",img)

########################################
#histogram equalization

import numpy as np
import cv2
#import math

img = cv2.imread("C:\\Users\\Prem Prasad\\Pictures\\test\\test_7.png",0)

height = img.shape[0]
width = img.shape[1]
pixels = height * width

#histogram code:
def histogram(img):
    h = img.shape[0]
    w = img.shape[1]
    
    histo = np.zeros((256))
    
    for i in np.arange(h):
        for j in np.arange(w):
            a = img.item(i,j)
            histo[a] = histo[a] + 1
    return histo

#cumulative histogram code:
def cumulative_histogram(hist):
    c_hist = hist.copy()
    
    for i in np.arange(1,256):
        c_hist[i]=c_hist[i-1]+c_hist[i]
        
    return c_hist

hist = histogram(img)
cum_hist = cumulative_histogram(hist)

for i in np.arange(height):
    for j in np.arange(width):
        a = img.item(i,j)
        b = math.floor(cum_hist[a] * 255.0 / pixels)
        img.itemset((i,j), b)

plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')

cv2.imwrite('C:\\Users\\Prem Prasad\\Pictures\\test\\test_7_hist_equal.png',img)



#############################################
#histogram matching

img = cv2.imread("C:\\Users\\Prem Prasad\\Pictures\\test\\test_7.png",0)
img_ref = cv2.imread("C:\\Users\\Prem Prasad\\Pictures\\test\\test_2.jpg",0)

height = img.shape[0]
width = img.shape[1]
pixels = height*width

height_ref = img_ref.shape[0]
width_ref = img_ref.shape[1]
pixels_ref = height_ref * width_ref

#histogram code:
def histogram(img):
    h = img.shape[0]
    w = img.shape[1]
    
    histo = np.zeros((256))
    
    for i in np.arange(h):
        for j in np.arange(w):
            a = img.item(i,j)
            histo[a] = histo[a] + 1
    return histo

#cumulative histogram code:
def cumulative_histogram(hist):
    c_hist = hist.copy()
    
    for i in np.arange(1,256):
        c_hist[i]=c_hist[i-1]+c_hist[i]
    
    return c_hist

hist = histogram(img)
hist_ref = histogram(img_ref)

cum_hist = cumulative_histogram(hist)
cum_hist_ref = cumulative_histogram(hist_ref)

#normalize the cumulative histograms to compare images of different sizes/dimensions
prob_cum_hist = cum_hist / pixels
prob_cum_hist_ref = cum_hist_ref / pixels_ref

k = 256
new_values = np.zeros((k))

for a in np.arange(k):
    j = k - 1
    while True:
        new_values[a] = j
        j = j - 1
        if j < 0 or prob_cum_hist[a] > prob_cum_hist_ref[j]:
            break
        
for i in np.arange(height):
    for j in np.arange(width):
        a = img.item(i,j)
        b = new_values[a]
        img.itemset((i,j),b)
        
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    
cv2.imwrite('C:\\Users\\Prem Prasad\\Pictures\\test\\test_7_hist_match.png',img)

#########################################
#Linear blending of two images

img1 = cv2.imread("C:\\Users\\Prem Prasad\\Pictures\\test\\test_1.jpg",0)
img2 = cv2.imread("C:\\Users\\Prem Prasad\\Pictures\\test\\test_3.jpg",0)

height = img2.shape[0]
width = img2.shape[1]

#alpha is used to set the % of the image to be shown. 
#0.5 means equal blend
alpha = 0.75

for i in np.arange(height):
    for j in np.arange(width):
        a1 = img1.item(i,j)
        a2 = img2.item(i,j)
        b = a1 * alpha + a2 * (1-alpha)
        img1.itemset((i,j), b)

plt.imshow(img1, cmap = 'gray', interpolation = 'bicubic')


##########################################
#Blur with box filter
#Note: takes time to execute!!

img = cv2.imread("C:\\Users\\Prem Prasad\\Pictures\\test\\test_3.jpg",0)
img_out = img.copy()

height = img.shape[0]
width = img.shape[1]

# arange values chosen to prevent margin errors
for i in np.arange(3, height - 3):
    for j in np.arange(3, width - 3):
        sum = 0
        for k in np.arange(-3 , 4):
            for l in np.arange(-3, 4):
                a = img.item(i+k, j+l)
                sum = sum + a
        b = int(sum / 49.0)
        img_out.itemset((i,j), b)
        
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')

##########################################
#Gausian Filter for blurring *
#Note: Takes time!!

img = cv2.imread("C:\\Users\\Prem Prasad\\Pictures\\test\\test_3.jpg",0)
img_out = img.copy()

height = img.shape[0]
width = img.shape[1] 

#sum of array values is 57
gauss = (1.0/57) * np.array(
        [[0,1,2,1,0],
         [1,3,5,3,1],
         [2,5,9,5,2],
         [1,3,5,3,1],
         [0,1,2,1,0]])
    
sum(sum(gauss)) #should be 1.0 since sum of probabilities

#arange values chosen to prevent margin errors
for i in np.arange(2, height-2):
    for j in np.arange(2, width-2):
        sum = 0
        for k in np.arange(-2,3):
            for l in np.arange(-2,3):
                a = img.item(i+k , j+l)
                #offset of 2 to accomodate k and l values
                p = gauss[2+k , 2+l]
                sum = sum + (p * a)
        b = sum
        img_out.itemset((i,j), b)
        
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')

##########################################
#Laplace difference Filter *
#Note: Takes time!!

img = cv2.imread("C:\\Users\\Prem Prasad\\Pictures\\test\\test_3.jpg",0)
img_out = img.copy()

height = img.shape[0]
width = img.shape[1] 

#sum of array values is 0
laplace = (1.0/16) * np.array(
        [[0,0,-1,0,0],
         [0,-1,-2,-1,0],
         [-1,-2,16,-2,-1],
         [0,-1,-2,-1,0],
         [0,0,-1,0,0]])
    
sum(sum(laplace)) #should be 0.0 

#arange values chosen to prevent margin errors
for i in np.arange(2, height-2):
    for j in np.arange(2, width-2):
        sum = 0
        for k in np.arange(-2,3):
            for l in np.arange(-2,3):
                a = img.item(i+k , j+l)
                #offset of 2 to accomodate k and l values
                w = laplace[2+k , 2+l]
                sum = sum + (w * a)
        b = sum
        img_out.itemset((i,j), b)
        
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')

###############################################
# Non-linear filter * : Max
# Takes time

img = cv2.imread("C:\\Users\\Prem Prasad\\Pictures\\test\\test_3.jpg",0)
img_out = img.copy()

height = img.shape[0]
width = img.shape[1]

# arange values chosen to prevent margin errors
for i in np.arange(3, height - 3):
    for j in np.arange(3, width - 3):
        max_val = 0
        for k in np.arange(-3 , 4):
            for l in np.arange(-3, 4):
                a = img.item(i+k, j+l)
                if a > max_val:
                    max_val = a
        b = max_val
        img_out.itemset((i,j), b)
        
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')

###############################################
# Non-linear filter * : Min
# Takes time

img = cv2.imread("C:\\Users\\Prem Prasad\\Pictures\\test\\test_3.jpg",0)
img_out = img.copy()

height = img.shape[0]
width = img.shape[1]

# arange values chosen to prevent margin errors
for i in np.arange(3, height - 3):
    for j in np.arange(3, width - 3):
        min_val = 255
        for k in np.arange(-3 , 4):
            for l in np.arange(-3, 4):
                a = img.item(i+k, j+l)
                if a < min_val:
                    min_val = a
        b = min_val
        img_out.itemset((i,j), b)
        
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')

##############################################
# Non-Linear filter: Median Filter (equal weights)
# used when paper noise is present
# Takes time

img = cv2.imread("C:\\Users\\Prem Prasad\\Pictures\\test\\test_3.jpg",0)
img_out = img.copy()

height = img.shape[0]
width = img.shape[1]

for i in np.arange(3, height - 3):
    for j in np.arange(3, width - 3):
        neighbors = []
        for k in np.arange(-3 , 4):
            for l in np.arange(-3, 4):
                a = img.item(i+k , j+l)
                neighbors.append(a)
        neighbors.sort()
        median = neighbors[24]
        b = median
        img_out.itemset((i,j), b)
        
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')

cv2.imwrite('C:\\Users\\Prem Prasad\\Pictures\\test\\test_3_med_filter.jpg',img)

##############################################
# Non-Linear filter: Weighted Median Filter (different weights)
# used when paper noise is present
# Takes time

img = cv2.imread("C:\\Users\\Prem Prasad\\Pictures\\test\\test_3.jpg",0)
img_out = img.copy()

weights = np.array(
        [[0,0,1,2,1,0,0],
         [0,1,2,3,2,1,0],
         [1,2,3,4,3,2,1],
         [2,3,4,5,4,3,2],
         [1,2,3,4,3,2,1],
         [0,1,2,3,2,1,0],
         [0,0,1,2,1,0,0]])
    
height = img.shape[0]
width = img.shape[1]

M = int((sum(sum(weights)) - 1) / 2)

for i in np.arange(3, height - 3):
    for j in np.arange(3, width - 3):
        neighbors = []
        for k in np.arange(-3 , 4):
            for l in np.arange(-3, 4):
                a = img.item(i+k , j+l)
                w = weights[k+3 , l+3]
                for _ in np.arange(w):
                    neighbors.append(a)
        neighbors.sort()
        median = neighbors[M]
        b = median
        img_out.itemset((i,j), b)
        
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')

cv2.imwrite('C:\\Users\\Prem Prasad\\Pictures\\test\\test_3_med_filter.jpg',img)

################################################
#Sobel Edge Operator: shows the edges within the image
#uses change in intensity to identify

# X and F are numpy matrices: 
# X is the image, F is the filter
def convolve_np(X, F):
    X_height = X.shape[0]
    X_width = X.shape[1]
    
    F_height = F.shape[0]
    F_width = F.shape[1]
    
    H = (F_height - 1) / 2
    W = (F_width - 1) / 2
    
    out = np.zeros((X_height, X_width))
    
    for i in np.arange(H , X_height - H):
        for j in np.arange(W , X_width - W):
            sum = 0
            for k in np.arange(-H , H + 1):
                for l in np.arange(-W , W + 1):
                    a = X[int(i+k) , int(j+l)]
                    w = F[int(H+k) ,int(W+l)]
                    sum = sum + (w * a)
            out[int(i),int(j)] = sum
    return out

img = cv2.imread("C:\\Users\\Prem Prasad\\Pictures\\test\\test_3.jpg",0)

height = img.shape[0]
width = img.shape[1]

Hx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]])
    
Hy = np.array([[-1, -2, -1],
               [0, 0, 0],
               [1, 2, 1]])
    
img_x = convolve_np(img, Hx) / 8.0
img_y = convolve_np(img, Hy) / 8.0

img_out = np.sqrt(np.power(img_x, 2) + np.power(img_y, 2))

img_out = (img_out / np.max(img_out)) * 255

plt.imshow(img_out, cmap = 'gray', interpolation = 'bicubic')        
    
################################################
#Prewitt Edge Operator: shows the edges within the image
#uses change in intensity to identify

# X and F are numpy matrices: 
# X is the image, F is the filter
def convolve_np(X, F):
    X_height = X.shape[0]
    X_width = X.shape[1]
    
    F_height = F.shape[0]
    F_width = F.shape[1]
    
    H = (F_height - 1) / 2
    W = (F_width - 1) / 2
    
    out = np.zeros((X_height, X_width))
    
    for i in np.arange(H , X_height - H):
        for j in np.arange(W , X_width - W):
            sum = 0
            for k in np.arange(-H , H + 1):
                for l in np.arange(-W , W + 1):
                    a = X[int(i+k) , int(j+l)]
                    w = F[int(H+k) ,int(W+l)]
                    sum = sum + (w * a)
            out[int(i),int(j)] = sum
    return out

img = cv2.imread("C:\\Users\\Prem Prasad\\Pictures\\test\\test_3.jpg",0)

height = img.shape[0]
width = img.shape[1]

#only difference w.r.t Sobel is the following matrices:
Hx = np.array([[-1, 0, 1],
               [-1, 0, 1],
               [-1, 0, 1]])
    
Hy = np.array([[-1, -1, -1],
               [0, 0, 0],
               [1, 1, 1]])
    
img_x = convolve_np(img, Hx) / 6.0
img_y = convolve_np(img, Hy) / 6.0

img_out = np.sqrt(np.power(img_x, 2) + np.power(img_y, 2))

img_out = (img_out / np.max(img_out)) * 255

plt.imshow(img_out, cmap = 'gray', interpolation = 'bicubic')    
    
    