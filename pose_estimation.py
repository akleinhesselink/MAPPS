# pose estimation 

import cv2
import numpy as np
import glob

# functions 

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img


# load previously saved camera matrix and distortion matrix 
mtx = np.load('calibration/mtx.npy')
dist = np.load('calibration/dist.npy')

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
 
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

src = 'test_images/checkerboard/MVI_1952.MOV'

cap = cv2.VideoCapture(src)
ret, img = cap.read()

fshape = img.shape[::-1][1:3]

ind = 0 
skip = 1

while(cap.isOpened()):
    
    ind += 1
    ret, img = cap.read()
    
    if ind % skip == 0 : 
        if ret==True:            
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            _, bw = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
            ret, corners = cv2.findChessboardCorners(bw, (7,6), None)

            if ret == True:

                # Find the rotation and translation vectors.
                ret, rvecs, tvecs = cv2.solvePnP(objp, corners, mtx, dist)

                # project 3D points to image plane
                imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

                img = draw(img,corners,imgpts)
                cv2.imshow('img', img)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break  
            
            else: 
                cv2.imshow('img', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            break

cap.release()
cv2.destroyAllWindows()