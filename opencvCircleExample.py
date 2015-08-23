import cv2 
import numpy as np

img = cv2.imread('tennis_balls.jpg')

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.imshow('original', img) 
cv2.imshow('hsv', hsv)

mask = cv2.inRange(hsv, (0.11*256, 0.60*256, 0.20*256), (0.14*256, 1.00*256, 1.00*256) ) #### mask out yellow

###### These steps should clean up some of smaller noise in the image 
se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 10) ) #### create kernel for the morphological operations
se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 5) ) #### create kernel for the morphological operations
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se2) #### close = dilation followed by erosion 
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se1) #### open = erosion followed by dilation 

###### smooth the image 
mask2 = cv2.GaussianBlur(mask, (15, 15), sigmaX=0, sigmaY=0)

circles = cv2.HoughCircles( mask2, cv2.cv.CV_HOUGH_GRADIENT, 1, 5, param1 = 200, param2 = 30, minRadius=20, maxRadius= 200)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('detected circles',img)

cv2.waitKey(0)
cv2.destroyAllWindows()


