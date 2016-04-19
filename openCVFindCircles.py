import cv2 
import numpy as np
import matplotlib.mlab as mlab
from matplotlib import pyplot as plt

def maskColor( img, colorMasks ):     
    mask = []
    for colors in colorMasks:         
        mask.append(cv2.inRange(img, colors[0], colors[1]))
    mask = sum(mask)
    return(mask)

def cleanNoise( img, smallKernal, bigKernal):     
    ###### These steps should clean up some of smaller noise in the image 
    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, smallKernal ) #### create kernel for the morphological operations
    se2 = cv2.getStructuringElement(cv2.MORPH_RECT, bigKernal ) 
    mask = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se2) #### close = dilation followed by erosion 
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se1) #### open = erosion followed by dilation 
    return( mask )

def prepareImage( img, colorRange, smallKernal, bigKernal, blurSize ): 
    mask = maskColor( hsv, colorRange)
    mask = cleanNoise( mask, smallKernal= smallKernal, bigKernal= bigKernal )
    mask = cv2.GaussianBlur(mask, blurSize, sigmaX=0, sigmaY=0)    
    return(mask)

def checkDetection( img, circles, color  ): 
    if (img is None) : 
        print ("no image")
    if (circles is None): 
        print("no circles")
        
    for i in circles[0,:]:
        cv2.circle(img,tuple(i[0:2]),i[2],color,1)
        cv2.circle(img,tuple(i[0:2]),2,color,1)
    
    centroid = np.mean(circles, axis = 1)
    cv2.circle(img, tuple(centroid.astype(int)[0][0:2]), centroid.astype(int)[0][2], (0,0,0), 2)
    cv2.circle(img, tuple(centroid.astype(int)[0][0:2]), 2, (0, 0, 0), 3) 
    cv2.imshow('detected circles',img)

#### Define Variables: ####
imgFile = 'tennis_balls.jpg' #### detect yellow tennis balls 
imgFile2 = 'test_images/colors/IMG_1937.JPG'

loColor = np.array([24, 0.60*255, 0.2*255]) #### low yellow
hiColor = np.array([38, 1.00*255, 1.00*255]) #### high yellow
loBlue = np.array([95, 70, 150])
hiBlue = np.array([110, 256, 256])
loRed1 = np.array([173, 130, 150])
hiRed1 = np.array([180, 256, 256])
loRed2 = np.array([0, 120, 150])
hiRed2 = np.array([7, 256, 256])

redRange = [[loRed1, hiRed1], [loRed2, hiRed2]]
blueRange = [[loBlue, hiBlue]]

smallKernal = (6, 3)
bigKernal = (11, 5)
blurSize = (5, 5)
###########################

colors = [ 'green', 'blue', 'red']
maximums = [ 180, 256, 256]
img = cv2.imread(imgFile2)
img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25) 
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #### convert to hsv 

chans = cv2.split(hsv)

#for (i , chan, color) in zip(range(len(chans)), chans, colors ) :        
#    plt.figure(i)
#    n, bins, patches = plt.hist(chan.ravel(), maximums[i], [0, maximums[i]], facecolor=color, alpha=0.5)
#    plt.show()

redMask = prepareImage(hsv, redRange, smallKernal, bigKernal, blurSize)
blueMask = prepareImage(hsv, blueRange, smallKernal, bigKernal, blurSize)

redCircles = cv2.HoughCircles( redMask, cv2.cv.CV_HOUGH_GRADIENT, 
                            1, 5, param1 = 100, param2 = 20, minRadius=10, maxRadius= 200)

redCircles = np.uint16(np.around(redCircles))
checkDetection(img, redCircles, color = (0,0,255))

blueCircles = cv2.HoughCircles( blueMask, cv2.cv.CV_HOUGH_GRADIENT, 
                            1, 5, param1 = 100, param2 = 20, minRadius=10, maxRadius= 200)

redCircles = np.uint16(np.around(redCircles))
checkDetection(img, blueCircles, color = (255, 0, 0))

cv2.imshow('maskBlue', blueMask)
cv2.imshow('maskRed', redMask)

cv2.waitKey(0)
cv2.destroyAllWindows()

