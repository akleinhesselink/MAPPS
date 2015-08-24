import cv2 
import numpy as np

#### Define Variables: ####
imgFile = 'tennis_balls.jpg' #### detect yellow tennis balls 
imgFile = 'single_tennis_ball.png'

loColor = (0, 70, 70) #### low yellow
hiColor = (80, 256, 256) #### high yellow 

blurSize = (15, 15)
###########################

img = cv2.imread(imgFile)

blur_img = cv2.GaussianBlur(img, blurSize, sigmaX=0, sigmaY=0)

hsv = cv2.cvtColor(blur_img, cv2.COLOR_BGR2HSV) #### convert to hsv

mask = cv2.inRange( hsv, loColor, hiColor)

cv2.imshow('test', mask )

contours, hierarchy  = cv2.findContours (mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

max_area = 0
largest_contour = None

for idx, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if area > max_area:
        max_area = area
        largest_contour = contour
    
if not largest_contour == None:
    M = cv2.moments(largest_contour)
    
x = int(M[ 'm10']/M[ 'm00'])
y = int(M[ 'm01']/M[ 'm00'])

print( x, y )

cv2.circle(img, (x, y), 40, (180,0,0), 3)

cv2.imshow('detected color', img )

cv2.waitKey(0)
cv2.destroyAllWindows()
