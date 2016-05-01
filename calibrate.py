import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

square_size = 3
objp = np.zeros((6*7, 3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# read video 
src = 'test_images/checkerboard/MVI_1952.MOV'

cap = cv2.VideoCapture(src)
ret, frame = cap.read()

fshape = frame.shape[::-1][1:3]

ind = 0 
skip = 1

while(cap.isOpened()):
    ind += 1
    ret, frame = cap.read()
    if ind % skip == 0 : 
        if ret==True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bw = cv2.threshold(frame, 100, 255, cv2.THRESH_BINARY)[1]
        
            ret, corners = cv2.findChessboardCorners(bw, (7,6),None)
            
            if ret == True:
                objpoints.append(objp)
        
                imgpoints.append(corners)
        
                # Draw and display the corners
                # cv2.drawChessboardCorners(frame, (7,6), corners, ret)
                
                cv2.imshow('frame', frame)     
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            break


# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, fshape)

# undistort a new image 
img = cv2.imread('test_images/checkerboard/still.0984.jpg')
h, w = img.shape[:2]

newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
#x,y,w,h = roi
#dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibration/test_calibration.png',dst)

# reprojection error 
tot_error = 0

for i in xrange(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    tot_error += error

print "total error: ", tot_error/len(objpoints)

text_file = open("calibration/calibration_error.txt", "w")
text_file.write("Total error =: %s; n =: %s;" % 
                (tot_error/len(objpoints), len(objpoints)))
text_file.close()

np.save('calibration/mtx.npy', mtx)
np.save('calibration/dist.npy', dist)
np.save('calibration/rvecs.npy', rvecs)
np.save('calibration/tvecs.npy', tvecs)

if ret: 
    print "calibration successful"
else:
    print "calibration unsuccessful"