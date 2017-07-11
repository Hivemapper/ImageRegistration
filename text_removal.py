import numpy as np
import cv2
img = cv2.imread('vlcshot.png', 0)
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
img = cv2.filter2D(img, -4, kernel)
imgcolor = cv2.imread('vlcshot.png')
mask = np.zeros(img.shape, np.uint8)
tmp = np.zeros(img.shape, np.uint8)
mask[img < 5] = 255
#mask[imgcolor[:,:,0] != imgcolor[:,:,1]] = 0
cv2.imwrite('thinmask.png',mask)
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

cv2.imwrite('textmask.png',mask)

mask = cv2.imread('textmask.png',0)

dst = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)
cv2.imwrite('detext.png',dst)
cv2.imwrite('bw.png',img)
#cv2.imshow('dst',dst)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
