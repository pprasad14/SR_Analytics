import numpy as np
import cv2
from matplotlib import pyplot as plt

########################################
#read and show image

img = cv2.imread('C:\\Users\\Prem Prasad\\Pictures\\test\\test_3.jpg',0)
cv2.imshow('image',img)
k = cv2.waitKey(0) & 0xFF
if k == 27: # wait for ESC key to exit
    print('k value is: ',k)
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    print('k value is: ',k)
    cv2.imwrite('C:\\Users\\Prem Prasad\\Pictures\\test\\gray3.jpg',img)
    cv2.destroyAllWindows()

img = cv2.imread('C:\\Users\\Prem Prasad\\Pictures\\test\\test_3.jpg',0)
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
plt.show()

########################################
#video capture, q to quit

cap = cv2.VideoCapture(0)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

########################################
