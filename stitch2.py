from pyimagesearch.panorama import Stitcher
from optparse import OptionParser
import imutils
import numpy as np
import cv2

num_frames = 2000

ratio=0.5
reprojThresh=2.0
#single size
width = 1280
#aggregate size
combined_height = 2000
combined_width = 2000

norm_threshold = 1000

blurriness_threshold = 150

def mask_text(img):
  mask = np.zeros(img.shape[0:2], np.uint8)
  mask[np.average(img,2) < 15] = 255
  #mask[imgcolor[:,:,0] != imgcolor[:,:,1]] = 0
  #print "I am masking"
  mask[1:-2,1:-2] = np.maximum(mask[1:-2,1:-2], mask[0:-3,1:-2])
  mask[1:-2,1:-2] = np.maximum(mask[1:-2,1:-2], mask[2:-1,1:-2])
  mask[1:-2,1:-2] = np.maximum(mask[1:-2,1:-2], mask[1:-2,0:-3])
  mask[1:-2,1:-2] = np.maximum(mask[1:-2,1:-2], mask[1:-2,2:-1])
  mask[1:-2,1:-2] = np.maximum(mask[1:-2,1:-2], mask[0:-3,1:-2])
  mask[1:-2,1:-2] = np.maximum(mask[1:-2,1:-2], mask[2:-1,1:-2])
  mask[1:-2,1:-2] = np.maximum(mask[1:-2,1:-2], mask[1:-2,0:-3])
  mask[1:-2,1:-2] = np.maximum(mask[1:-2,1:-2], mask[1:-2,2:-1])
  mask[1:-2,1:-2] = np.maximum(mask[1:-2,1:-2], mask[0:-3,1:-2])
  mask[1:-2,1:-2] = np.maximum(mask[1:-2,1:-2], mask[2:-1,1:-2])
  mask[1:-2,1:-2] = np.maximum(mask[1:-2,1:-2], mask[1:-2,0:-3])
  mask[1:-2,1:-2] = np.maximum(mask[1:-2,1:-2], mask[1:-2,2:-1])
  s = np.sum(mask)/(255.0*img.shape[0]*img.shape[1])
  #print s
  print s
  if s > .25:
    print "To Much Text: Skipping"
    raise Exception("Too much Text!")
  dst = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)
  return dst

def masked_merge(image1, image2):
  assert(image1.shape == image2.shape)
  w = image1.shape[0]
  h = image1.shape[1]
  div = np.ones([w,h], np.uint8)
  div[(image1[:,:,0] != 0) & (image2[:,:,0] != 0)] = 2
  #cv2.imwrite("stitch_results/mask.png", div)
  result = np.zeros(image1.shape, np.uint8)
  result[:,:,0] = image1[:,:,0]/div+image2[:,:,0]/div
  result[:,:,1] = image1[:,:,1]/div+image2[:,:,1]/div
  result[:,:,2] = image1[:,:,2]/div+image2[:,:,2]/div
  return result

def detectAndDescribe(image):
  # convert the image to grayscale
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # detect and extract features from the image
  descriptor = cv2.xfeatures2d.SIFT_create()
  (kps, features) = descriptor.detectAndCompute(image, None)

  # convert the keypoints from KeyPoint objects to NumPy
  # arrays
  kps = np.float32([kp.pt for kp in kps])

  # return a tuple of keypoints and features
  return (kps, features)

def matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
  # compute the raw matches and initialize the list of actual
  # matches
  matcher = cv2.DescriptorMatcher_create("BruteForce")
  rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
  
  #FLANN_INDEX_KDTREE = 0
  #index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
  #search_params = dict(checks = 50)
  #flann = cv2.FlannBasedMatcher(index_params, search_params)
  #rawMatches = flann.knnMatch(featuresA,featuresB,k=2)

  matches = []

  # loop over the raw matches
  for m in rawMatches:
    # ensure the distance is within a certain ratio of each
    # other (i.e. Lowe's ratio test)
    if len(m) == 2 and m[0].distance < m[1].distance * ratio:
      matches.append((m[0].trainIdx, m[0].queryIdx))

  # computing a homography requires at least 4 matches
  if len(matches) > 4:
    # construct the two sets of points
    ptsA = np.float32([kpsA[i] for (_, i) in matches])
    ptsB = np.float32([kpsB[i] for (i, _) in matches])

    # compute the homography between the two sets of points
    (homography, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
      reprojThresh)

    # return the matches along with the homograpy matrix
    # and status of each matched point
    return (matches, homography, status)

  # otherwise, no homograpy could be computed
  return None

if __name__ == "__main__":
  # construct the argument parse and parse the arguments
  #parser = OptionParser(usage = "usage: gridplot.py filename")
  #(options,args) = parser.parse_args()
  
  
  args = ["frames/%d.png"%i for i in range(0,num_frames,2)]
  
  imageA = imutils.resize(mask_text(cv2.imread(args[1])), width=width)
  
  
  Acenter = np.zeros((combined_height,combined_width,3), np.uint8)
  Ashape = imageA.shape
  print Ashape
  Astart = [Acenter.shape[0]/2-Ashape[0]/2, Acenter.shape[1]/2-Ashape[1]/2]
  Acenter[Astart[0]:Astart[0]+Ashape[0], Astart[1]:Astart[1]+Ashape[1]] = imageA
  imageA = Acenter
  (kpsA, featuresA) = detectAndDescribe(imageA)
  combo = imageA.copy()
  batch_i = 0;
  current_transform = np.identity(3)
  
  for i in range(2,len(args)):
    print "Image", i
    imageB = cv2.imread(args[i])
    try:
      imageB = mask_text(imageB)
    except:
      continue
    #blurriness = cv2.Laplacian(imageB, cv2.CV_64F).var()
    #print blurriness
    #if blurriness < blurriness_threshold:
      #print "Image Too Blurry!"
      #continue
    imageB = imutils.resize(imageB, width=width)
    #cv2.imwrite("stitch_results/tmp%d.png"%i, imageB)
    Bcenter = np.zeros((combined_height,combined_width,3), np.uint8)
    Bcenter[Astart[0]:Astart[0]+Ashape[0], Astart[1]:Astart[1]+Ashape[1]] = imageB
    imageB = Bcenter
    (kpsB, featuresB) = detectAndDescribe(imageB)

    # match features between the two images
    M = None
    try:
      M = matchKeypoints(kpsA, kpsB,
        featuresA, featuresB, ratio, reprojThresh)
    except:
      pass
    # if the match is None, then there aren't enough matched
    # keypoints to create a panorama
    #print M[1]
    
    if M is None or M[1] is None or np.linalg.norm(M[1]) > norm_threshold:
      print "Failed to Match: Starting new batch"
      if not (M is None) and not (M[1] is None):
        print "\tBig Norm:", np.linalg.norm(M[1])
      #cv2.imwrite("stitch_results/batch%d_result.png"%batch_i, combo)
      batch_i += 1
      combo = imageB
      current_transform = np.identity(3)
      
    else:
      # otherwise, apply a perspective warp to stitch the images
      # together
      (matches, homography, status) = M
      print np.linalg.norm(homography)
      current_transform = np.dot(homography, current_transform)
      Bwarp = cv2.warpPerspective(imageB, np.linalg.inv(current_transform),
        (combined_width, combined_height), flags=cv2.INTER_NEAREST)
      combo = masked_merge(combo,Bwarp)
    #update imageA
    imageA = imageB
    kpsA = kpsB
    featuresA = featuresB
    cv2.imwrite("stitch_results/batch%d_%d.png"%(batch_i,i), combo)
  print "Stitched"
  # show the images
  #cv2.imwrite("stitch_results/A.png", imageA)
  #cv2.imwrite("stitch_results/B.png", Bwarp)
  #cv2.imwrite("stitch_results/matches.png", vis)
  cv2.imwrite("stitch_results/batch%d_result.png"%batch_i, combo)
