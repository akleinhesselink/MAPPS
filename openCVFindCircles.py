import cv2 
import numpy as np

def maskColor( img, loColor, hiColor):     
    mask = cv2.inRange(img, loColor, hiColor) 
    return( mask )

def cleanNoise( img, smallKernal, bigKernal):     
    ###### These steps should clean up some of smaller noise in the image 
    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, smallKernal ) #### create kernel for the morphological operations
    se2 = cv2.getStructuringElement(cv2.MORPH_RECT, bigKernal ) 
    mask = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se2) #### close = dilation followed by erosion 
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se1) #### open = erosion followed by dilation 
    return( mask )

def prepareImage( img, loColor, hiColor, smallKernal, bigKernal, blurSize ): 
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #### convert to hsv 
    mask = maskColor( hsv, loColor, hiColor)
    mask = cleanNoise( mask, smallKernal= smallKernal, bigKernal= bigKernal )
    mask = cv2.GaussianBlur(mask, blurSize, sigmaX=0, sigmaY=0)    
    return(mask)

def checkDetection( img, circles ): 
    
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
    cv2.imshow('detected circles',img)

#### Define Variables: ####
imgFile = 'tennis_balls.jpg' #### detect yellow tennis balls 

loColor = (0.11*256, 0.60*256, 0.20*256) #### low yellow
hiColor = (0.14*256, 1.00*256, 1.00*256) #### high yellow 
smallKernal = (11, 5)
bigKernal = (21, 10)
blurSize = (15, 15)
###########################

img = cv2.imread(imgFile)

mask = prepareImage(img, loColor, hiColor, smallKernal, bigKernal, blurSize) 

circles = cv2.HoughCircles( mask, cv2.cv.CV_HOUGH_GRADIENT, 
                            1, 5, param1 = 200, param2 = 30, minRadius=20, maxRadius= 200)

circles = np.uint16(np.around(circles))

checkDetection(img, circles)

cv2.waitKey(0)
cv2.destroyAllWindows()
