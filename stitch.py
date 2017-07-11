from pyimagesearch.panorama import Stitcher
from optparse import OptionParser
import imutils
import numpy as np
import cv2
import pytesseract
from PIL import Image
import os

num_frames = 2100
frame_step = 2

ratio=0.5
reprojThresh=4.0
#single size
#width = 1280
width = 600
ghosts = 6
#aggregate size
combined_height = 2000
combined_width = 2000

norm_threshold = 1000

blurriness_threshold = 15
text_overlay = True

source_folder = "frames-ChinaLake-2fps"

def mask_text(img):
  #mask = np.zeros(img.shape[0:2], np.uint8)
  #mask[np.average(img,2) < 15] = 255
  #cv2.imwrite("ocrin.png", mask)
  #text = pytesseract.image_to_string(Image.open("ocrin.png"))
  ##os.remove("ocrin.png")
  #print text
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
  #print s
  if s > .25:
    print "\'Text\' Coverage"
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

def assessHomography(H):
  #print "Transform:"
  #print H
  ev =  np.linalg.eigvals(H)
  print "\t",ev
  if abs(np.real(ev[1])) < abs(np.imag(ev[1])) or np.real(ev[0]) < 0.25:
    #print "\n\nBAD TRANSFORM\n\n"
    return False  
  return True
  
  


if __name__ == "__main__":
  # construct the argument parse and parse the arguments
  #parser = OptionParser(usage = "usage: BOOO!")
  #(options,args) = parser.parse_args()
  
  
  args = ["%s/%d.png"%(source_folder,i) for i in range(0,num_frames,frame_step)]
  imageA = cv2.imread(args[1])
  imageA = imageA[ghosts:-ghosts,ghosts:-ghosts]
  if text_overlay:
    imageA = mask_text(imageA)
  imageA = imutils.resize(imageA, width=width)
  
  
  Acenter = np.zeros((combined_height,combined_width,3), np.uint8)
  Ashape = imageA.shape
  
  
  
  print "Frame Shape:"
  print Ashape
  Astart = [Acenter.shape[0]/2-Ashape[0]/2, Acenter.shape[1]/2-Ashape[1]/2]
  #transform the scaled image to the center of the combo
  center_transform = np.array([[1.0,0.0,Astart[1]],[0,1,Astart[0]],[0,0,1]])
  #center_transform = np.array([[1.0,0.0,Acenter.shape[0]/2],[0,1,Acenter.shape[1]/2],[0,0,1]])
  
  Acenter[Astart[0]:Astart[0]+Ashape[0], Astart[1]:Astart[1]+Ashape[1]] = imageA
  #imageA = Acenter
  (kpsA, featuresA) = detectAndDescribe(imageA)
  combo = Acenter
  batch_i = 0;
  current_transform = np.identity(3)
  
  for i in range(2,len(args)):
    print "Image", i
    imageB = cv2.imread(args[i])
    imageB = imageB[ghosts:-ghosts,ghosts:-ghosts]
    if text_overlay:
      try:
        imageB = mask_text(imageB)
      except:
        continue
    blurriness = cv2.Laplacian(imageB, cv2.CV_64F).var()
    print "\tBlur:", blurriness
    if blurriness < blurriness_threshold:
      print "Image Too Blurry!"
      cv2.imwrite("stitch_results/blurred_%d.png"%i, imageB)
      continue
    imageB = imutils.resize(imageB, width=width)
    #cv2.imwrite("stitch_results/tmp%d.png"%i, imageB)
    Bcenter = np.zeros((combined_height,combined_width,3), np.uint8)
    Bcenter[Astart[0]:Astart[0]+Ashape[0], Astart[1]:Astart[1]+Ashape[1]] = imageB
    #imageB = Bcenter
    (kpsB, featuresB) = detectAndDescribe(imageB)
    
    #Bwarp = cv2.warpPerspective(imageB, center_transform,
        #(combined_width, combined_height), flags=cv2.INTER_NEAREST)
    #cv2.imwrite("stitch_results/BC.png", Bcenter)
    #cv2.imwrite("stitch_results/BW.png", Bwarp)

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
    #if not (M is None) and not (M[1] is None):
        #assessHomography(M[1])
    #if M is None or M[1] is None or np.linalg.norm(M[1]) > norm_threshold:
    if M is None or M[1] is None or not assessHomography(M[1]):
      print "Failed to Match: Starting new batch"
      if not (M is None) and not (M[1] is None):
        print "\tNorm:", np.linalg.norm(M[1])
      #cv2.imwrite("stitch_results/batch%d_result.png"%batch_i, combo)
      batch_i += 1
      #combo = imageB
      combo = Bcenter
      current_transform = np.identity(3)
      
    else:
      # otherwise, apply a perspective warp to stitch the images
      # together
      (matches, homography, status) = M
      print "\tNorm:",np.linalg.norm(homography)
      current_transform = np.dot(homography, current_transform)
      #Bwarp = cv2.warpPerspective(imageB, np.linalg.inv(current_transform),
        #(combined_width, combined_height), flags=cv2.INTER_NEAREST)
      #combo = masked_merge(combo,Bwarp)
      Cwarp = cv2.warpPerspective(combo, np.dot(np.dot(center_transform,homography),np.linalg.inv(center_transform)),
      #Cwarp = cv2.warpPerspective(combo, np.dot(homography,np.linalg.inv(center_transform)),
      #Cwarp = cv2.warpPerspective(combo, np.linalg.inv(homography),
      #Cwarp = cv2.warpPerspective(combo, np.dot(homography,center_transform),
      #Cwarp = cv2.warpPerspective(combo, np.dot(center_transform,homography),
        (combined_width, combined_height), flags=cv2.INTER_NEAREST)
      #cv2.imwrite("stitch_results/batch_%d_%dc.png"%(batch_i,i), Cwarp)
      #combo = masked_merge(Cwarp,imageB)
      combo = masked_merge(Cwarp,Bcenter)
    #update imageA
    imageA = imageB
    Acenter = Bcenter
    kpsA = kpsB
    featuresA = featuresB
    cv2.imwrite("stitch_results/batch_%d_%d.png"%(batch_i,i), combo)
  print "Stitched"
  # show the images
  #cv2.imwrite("stitch_results/A.png", imageA)
  #cv2.imwrite("stitch_results/B.png", Bwarp)
  #cv2.imwrite("stitch_results/matches.png", vis)
  #cv2.imwrite("stitch_results/batch%d_result.png"%batch_i, combo)
