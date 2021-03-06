import numpy as np
import cv2
import time

from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10
start = time.time() 

#img1 = cv2.imread('box.png',0)          # queryImage
#img2 = cv2.imread('box_in_scene.png',0) # trainImage
#img1 = cv2.imread('overhead.png',0)          # queryImage
img1 = cv2.imread('middle_school.png',0)          # queryImage
#img1 = cv2.imread('overhead_cropped3.png',0)          # queryImage
#img2 = cv2.imread('giant_cropped.jpg',0) # trainImage
#img2 = cv2.imread('medium.png',0) # trainImage
#img2 = cv2.imread('USGSHRO1.jpg',0) # trainImage
img2 = cv2.imread('USGSHRO1_half.jpg',0) # trainImage
#img2 = cv2.imread('USGSHRO1_fifth.jpg',0) # trainImage
#img2 = cv2.imread('google_cropped.png',0) # trainImage


mask = np.zeros(img1.shape, np.uint8)
mask[img1 != 0] = 255
cv2.imwrite('mask.png',mask)

#for i in range(0, mask.shape[0]):
  #for j in range(0, mask.shape[1]):
    #if img1[i,j]!= 0:
      #mask[i,j] = 255


# Initiate SIFT detector
#sift = cv2.xfeatures2d.SIFT_create(nfeatures=100000)
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp2, des2 = sift.detectAndCompute(img2,None)
mid = time.time() 
kp1, des1 = sift.detectAndCompute(img1,None)





FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.8*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    #M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,10.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    matchesMask = None
    
end = time.time()
print("Total Time:",end - start)
print("Query Time:",end - mid)

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

#plt.imshow(img3, 'gray'),plt.show()
cv2.imwrite('matching_result.jpg',img3)
