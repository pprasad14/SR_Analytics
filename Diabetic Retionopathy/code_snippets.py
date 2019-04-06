# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 11:35:41 2018

@author: Prem Prasad
"""
##############################################################################
#to return the position of one element in an array
def index(array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx
        
x = np.arange(16).reshape(4,4)

ar = np.arange(8,12)
[idx for idx, el in enumerate(x) if np.array_equal(el, ar)]


#to get postions of elements >10
 x[x>10]
 np.nonzero(x>10)
 
##############################################################################
 #getting coordinates of pixels above given threshold
 
 #create an empty list to store coordinates
points = [] #type is list

#to get the NO. OF PIXELS having a value greater than a threshold value
new_image[new_image>.811].shape

#retrieve all x and y coordinates of image whose values above threshold
coor = np.nonzero(new_image>.4)

#append the x and y coordinates for each point
for i in np.arange(len(coor[0])):
    points.append(coor[0][i])
    points.append(coor[1][i])
    
#to reshape, convert to numpy array
points = np.array(points)
points = points.reshape(len(coor[0]), 2)

#to see the reshaped points array:
points

#check shape of points:
points.shape
        
img_plot = plt.imshow(new_image,cmap=plt.cm.Greens_r)

plt.scatter(x = coor[0], y = coor[1], c = 'r', s = 40)

#############################################################################
#Canny edge detection

img = cv2.imread("C:\\Users\\Prem Prasad\\Pictures\\test\\test_3.jpg",0)

edges = cv2.Canny(img,50,100) # find best min and max values 

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()

############################################################################
#outlines

import numpy as np
import cv2 #this is the main openCV class, the python binding file should be in /pythonXX/Lib/site-packages
from matplotlib import pyplot as plt

gwashBW = cv2.imread("C:\\Users\\Prem Prasad\\Pictures\\test\\test_60.png",0) #import image
 #change to grayscale
 
ret,thresh1 = cv2.threshold(gwashBW,66,255,cv2.THRESH_BINARY) 
 
kernel = np.ones((5,5),np.uint8) #square image kernel used for erosion
erosion = cv2.erode(thresh1, kernel,iterations = 1) #refines all edges in the binary image

opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel) #this is for further removing small noises and holes in the image

plt.imshow(closing, 'gray') #Figure 2
plt.xticks([]), plt.yticks([])
plt.show()
 
closing,contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #find contours with simple approximation

plt.imshow(closing, cmap = 'gray', interpolation = 'bicubic')

areas = [] #list to hold all areas

for contour in contours:
  ar = cv2.contourArea(contour)
  areas.append(ar)
  
max_area = max(areas)
max_area_index = areas.index(max_area) #index of the list element with largest area

cnt = contours[max_area_index] #largest area contour

cv2.drawContours(closing, [cnt], 0, (255, 255, 255), 3, maxLevel = 0)

plt.imshow(closing, cmap = 'gray', interpolation = 'bicubic')
################################

 im = cv2.imread('C:\\Users\\Prem Prasad\\Pictures\\test\\test_60.png')
 imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
 ret,thresh = cv2.threshold(imgray,69,255,0)
 im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
 cv2.drawContours(im2, contours, 4, (0,255,0), 3)
 
 plt.imshow(im2, cmap = 'gray', interpolation = 'bicubic')

#########################Create the Center Mask:##############################
#using Red Channel
center_mask = image_slice_red.copy()

#threshold for center_white disc
threshold = 150/255 #to see image of different thresholds, load the image again
 
#create mask for white region
for i in np.arange(height):
    for j in np.arange(width):
        a = center_mask.item(i,j)
        if a > threshold:
            b = 255
        else: 
            b = 0
        center_mask.itemset((i,j),b)

#to show in grey scale
plt.imshow(center_mask, cmap = 'gray', interpolation = 'bicubic')
'''
'''
#with respect to red 
plt.imshow(center_mask, cmap=plt.cm.Reds_r) 
#with respect to green 
plt.imshow(center_mask, cmap=plt.cm.Greens_r)  
#with respect to blue 
plt.imshow(center_mask, cmap=plt.cm.Blues_r) 

##############################################################################
#to see grayscale in different thresholds
fig, ax = try_all_threshold(image_data_gray, figsize=(10, 8), verbose=False)
############################################################################

im = cv2.imread( "C:\\Users\\Prem Prasad\\Pictures\\test\\blob.jpg", cv2.IMREAD_GRAYSCALE)

detector = cv2.SimpleBlobDetector_create()

params = cv2.SimpleBlobDetector_Params()
 
# filter by thresholds
params.minThreshold = 0;
params.maxThreshold = 75;

#filter by area
params.filterByArea = True
params.minArea = 500

#Filter by Circularity
params.filterByCircularity = True;
params.minCircularity = 0.8;

#Filter by Convexity
params.filterByConvexity = True;
params.minConvexity = 0.87;

# Filter by Inertia
params.filterByInertia = True;
params.minInertiaRatio = 0.50;

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
    detector = cv2.SimpleBlobDetector(params)
else :
    detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(im)

im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.imshow(im_with_keypoints, cmap = 'gray', interpolation = 'bicubic')

cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)
#######################



import cv2
import numpy as np;
from matplotlib import pyplot as plt

# Read image
img = cv2.imread("C:\\Users\\Prem Prasad\\Pictures\\test\\blob2.png",0)
########################################
#threshold

height = img.shape[0]
width = img.shape[1]

threshold = 240 #to see image of different thresholds, load the image again

for i in np.arange(height):
    for j in np.arange(width):
        a = img.item(i,j)
        if a > 240:
            b = 255
        if a < 100:
            b = 255
        else:
            b = 0
        img.itemset((i,j),b)

plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 0
params.maxThreshold = 255

# Filter by Area.
params.filterByArea = True
params.minArea = 500

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.9

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.1

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.1

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs.
keypoints = detector.detect(img)

im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.imshow(im_with_keypoints, cmap = 'gray', interpolation = 'bicubic')


titles = ['Blobs Detected']
images = [im_with_keypoints]


for i in xrange(1):    
    plt.subplot(1,1,i+1), plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis    
plt.show()

###########################################
# range of threshold values

height = img.shape[0]
width = img.shape[1]

for i in np.arange(height):
    for j in np.arange(width):
        a = img.item(i,j)
        if a > 0 and a < 50:
            b = 50
        elif a > 50 and a < 100:
            b = 100
        elif a > 100 and a < 150:
            b = 150
        elif a > 150 and a < 200:
            b = 200
        elif a > 100 and a < 150:
            b = 255 
        img.itemset((i,j),b)

plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')

################################################
blobs_log = blob_log(image_gray_green, max_sigma=30, num_sigma=10, threshold=.3)
blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)# Compute radii in the 3rd column.

blobs_dog = blob_dog(image_gray_green, max_sigma=30, threshold=.5)
blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

blobs_doh = blob_doh(image_gray_green,min_sigma = 1, overlap = 0.5,
                     max_sigma=50,num_sigma=15, threshold=.0007)

blobs_list = [blobs_log, blobs_dog, blobs_doh]
colors = ['red', 'lime', 'yellow']
titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
          'Determinant of Hessian']
sequence = zip(blobs_list, colors, titles)

fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True,
                         subplot_kw={'adjustable': 'box-forced'})
ax = axes.ravel()

for idx, (blobs, color, title) in enumerate(sequence):
    ax[idx].set_title(title)
    ax[idx].imshow(image_gray_green, interpolation='nearest')
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
        ax[idx].add_patch(c)
    ax[idx].set_axis_off()

plt.tight_layout()
plt.show()
















