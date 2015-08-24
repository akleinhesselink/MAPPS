import cv2 
import numpy as np

def prepareImage( img, loColor, hiColor, blurSize ): 
    blur_img = cv2.GaussianBlur(img, blurSize, sigmaX=0, sigmaY=0)
    hsv = cv2.cvtColor(blur_img, cv2.COLOR_BGR2HSV) #### convert to hsv     
    mask = cv2.inRange( hsv, loColor, hiColor)
    return(mask)

#### Define Variables: ####
imgFile = 'tennis_balls.jpg' #### detect yellow tennis balls 
imgFile = 'single_tennis_ball.png'

loColor = (0, 70, 70) #### low yellow
hiColor = (80, 256, 256) #### high yellow 
smallKernal = (11, 5)
bigKernal = (21, 10)
blurSize = (15, 15)
###########################

img = cv2.imread(imgFile)

mask = prepareImage( img, loColor, hiColor, blurSize )

cv2.imshow('test', mask )

M_all  = cv2.moments(mask , 0 )

area = M_all[ 'm00']

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
    
print(area, M['m00'] )

x = int(M[ 'm10']/M[ 'm00'])
y = int(M[ 'm01']/M[ 'm00'])

print( x, y )

cv2.circle(img, (x, y), 40, (180,0,0), 3)

cv2.imshow('detected color', img )

cv2.waitKey(0)
cv2.destroyAllWindows()
