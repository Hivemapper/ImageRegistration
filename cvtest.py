import cv2
import numpy as np
#from matplotlib import pyplot as plt

#filename = 'overhead.png'
#img = cv2.imread(filename)
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#gray = np.float32(gray)
#dst = cv2.cornerHarris(gray,2,3,0.04)

##result is dilated for marking the corners, not important
#dst = cv2.dilate(dst,None)

## Threshold for an optimal value, it may vary depending on the image.
#img[dst>0.01*dst.max()]=[0,0,255]

#cv2.imshow('dst',img)
#if cv2.waitKey(0) & 0xff == 27:
    #cv2.destroyAllWindows()

img = cv2.imread('google_cropped.png')
img = cv2.imread('overhead_cropped3.png')
#img = cv2.imread('home.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)

cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imwrite('sift_keypoints.jpg',img)
#cv2.imshow('dst',img)


img = cv2.imread('overhead_cropped3.png')
# Initiate STAR detector
orb = cv2.ORB_create(nfeatures=2000)

# find the keypoints with ORB
kp = orb.detect(img,None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)

# draw only keypoints location,not size and orientation
cv2.drawKeypoints(img,kp,img,color=(0,255,0), flags=0)
#cv2.imshow('dst',img)
cv2.imwrite('ORB_keypoints.jpg',img)
