import numpy as np
from scipy.misc import imread#, imsave
import pylab as plt
import math
import cv2
#from skimage.filters import try_all_threshold, threshold_minimum

image_data = imread("C:\\Users\\Prem Prasad\\Pictures\\test\\test_19.png")
#image_data_gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)

#RGB and Grayscale
plt.imshow(image_data, cmap = 'gray', interpolation = 'bicubic')
#plt.imshow(image_data_gray, cmap = 'gray', interpolation = 'bicubic')

#fig, ax = try_all_threshold(image_data_gray, figsize=(10, 8), verbose=False)

#scaling image, results in values of matrix in range of 0 to 1
scaled_image_data = image_data #/ 255.
#scaled_image_data_gray = image_data_gray / 255.
#plt.imshow(scaled_image_data, cmap = 'gray', interpolation = 'bicubic')
#plt.imshow(scaled_image_data_gray, cmap = 'gray', interpolation = 'bicubic')

########################create the Edge Mask:################################

height = image_data.shape[0]
width = image_data.shape[1]

edge_mask = image_data[:,:,1].copy() #used green channel, since cleaner than red and blue

#threshold for black region: Edges
#taken 4 since 1 or 2 or 3 have some outliers
threshold = 4 #to see image of different thresholds, load the image again

#create mask for black region
for i in np.arange(height):
    for j in np.arange(width):
        a = edge_mask.item(i,j)
        if a > threshold:
            b = 255
        else: 
            b = 0
        edge_mask.itemset((i,j),b)

plt.imshow(edge_mask, cmap = 'gray', interpolation = 'bicubic')#in gray scale
plt.imshow(edge_mask, cmap=plt.cm.Greens_r)#in green scale

################################################################


######################## separating channels#####################

#getting separate slices of the image ie Red, Green, Blue channels
image_slice_red =  scaled_image_data[:,:,0]
image_slice_green =  scaled_image_data[:,:,1]
image_slice_blue =  scaled_image_data[:,:,2]

#to see the different color components of an image
plt.subplot(221)
plt.imshow(image_slice_red, cmap=plt.cm.Reds_r)

plt.subplot(222)
plt.imshow(image_slice_green, cmap=plt.cm.Greens_r)

plt.subplot(223)
plt.imshow(image_slice_blue, cmap=plt.cm.Blues_r)  

plt.subplot(224)
plt.imshow(scaled_image_data)  

plt.show()

#set the Green Channel as the new image
new_image = image_slice_green

################################################################
#blob detection

# Read red image 
img = image_slice_green.copy() #stores as grayscale

plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')

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




##########################################
#img = imread("C:\\Users\\Prem Prasad\\Pictures\\test\\blob2.png")
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#threshold
height = img.shape[0]
width = img.shape[1]

threshold = 130 #to see image of different thresholds, load the image again

for i in np.arange(height):
    for j in np.arange(width):
        a = img.item(i,j)
        if a > threshold:
            b = 255
        #if a < 100:
         #   b = 255
        else:
            b = 0
        img.itemset((i,j),b)

plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')

#invert image: 

max_intensity = 255

for i in np.arange(height):
    for j in np.arange(width):
        a = img.item(i,j)
        b = max_intensity - a
        img.itemset((i,j),b)

plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 0
params.maxThreshold = 255 

# Filter by Area.
params.filterByArea = True
params.minArea = 100

# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.5

# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.75

# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.1

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs.
keypoints = detector.detect(img)

len(keypoints)

im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.imshow(im_with_keypoints, cmap = 'gray', interpolation = 'bicubic')


#################### Center Mask #########################################
#using Red Channel
center_mask = image_slice_red.copy()

#threshold for center_white disc
threshold = 150/255 #to see image of different thresholds, load the image again
 
#create mask for white region
for i in np.arange(height):
    for j in np.arange(width):
        a = center_mask.item(i,j)
        if a > threshold:
            b = 0
        else: 
            b = 255
        center_mask.itemset((i,j),b)

#to show in grey scale
plt.imshow(center_mask, cmap = 'gray', interpolation = 'bicubic')


####################Combine both Masks:#######################################

final_mask = edge_mask + center_mask
#to show in grey scale
plt.imshow(final_mask, cmap = 'gray', interpolation = 'bicubic')

#################################################

