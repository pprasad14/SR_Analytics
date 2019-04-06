from math import sqrt
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd

white = []
black = []

for a in np.arange(1,90):
    if a < 10:
        image = cv2.imread("C:\\Users\\Prem Prasad\\Pictures\\retina\\image00%d.png"%a)    
    else:
        image = cv2.imread("C:\\Users\\Prem Prasad\\Pictures\\retina\\image0%d.png"%a)    

    image_slice_green =  image[:,:,1]
    image_gray_green = rgb2gray(image_slice_green)
    plt.imshow(image_gray_green, cmap = 'gray', interpolation = 'bicubic')

    ############## finding white count#################
    '''
    blobs_doh = blob_doh(image_gray_green,min_sigma = 1,
                         max_sigma=50,num_sigma=20, threshold=.0008)

    white.append(len(blobs_doh))
    '''
    
    params = cv2.SimpleBlobDetector_Params()
    params.blobColor = 255
    # Change thresholds
    params.minThreshold = 0
    params.maxThreshold = 255 
    
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 10
    
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.5
    
    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.75
    
    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.1
    
    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    
    # Detect blobs.
    keypoints = detector.detect(image_gray_green)
    
    len(keypoints)
    
    white.append(len(keypoints))
    #########################Finding the black spots ##########

    params = cv2.SimpleBlobDetector_Params()
    params.blobColor = 0
    # Change thresholds
    params.minThreshold = 0
    params.maxThreshold = 255 
    
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 10
    
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.5
    
    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.75
    
    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.1
    
    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    
    # Detect blobs.
    keypoints = detector.detect(image_gray_green)
    
    len(keypoints)
    
    black.append(len(keypoints))
    ###################################
d = {'White spots': white, 'Black spots': black}
index_ = np.arange(1,90)
df = pd.DataFrame(data = d)
df= df.set_index(index_)
    
writer = pd.ExcelWriter('C:\\Users\\Prem Prasad\\Pictures\\retina\\output_2.xlsx')
df.to_excel(writer,'Sheet1')
writer.save()

#df = pd.read_excel('C:\\Users\\Prem Prasad\\Pictures\\retina\\output.xlsx', sheetname='Sheet1')
'''
for a in np.arange(1,90):
    b = df.loc[a][0]
    w = df.loc[a][1]
    
    if a < 10:
        image = cv2.imread("C:\\Users\\Prem Prasad\\Pictures\\retina\\image00%d.png"%a)    
    else:
        image = cv2.imread("C:\\Users\\Prem Prasad\\Pictures\\retina\\image0%d.png"%a)
        
    if w <=3 and b <=5:
        cv2.imwrite('C:\\Users\\Prem Prasad\\Pictures\\retina(classified)\\diabetes(no)\\%d.png'%a, image)
    else:
        cv2.imwrite('C:\\Users\\Prem Prasad\\Pictures\\retina(classified)\\diabetes(yes)\\%d.png'%a, image)
        
'''