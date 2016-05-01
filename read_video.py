import numpy as np
import cv2
import glob

# read video 
src = 'test_images/checkerboard/MVI_1952.MOV'

cap = cv2.VideoCapture(src)
ret, frame = cap.read()

while(cap.isOpened()):
     ret, frame = cap.read()
     if ret==True:
          frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
          frame = cv2.threshold(frame, 100, 255, cv2.THRESH_BINARY)[1]
          cv2.imshow('frame', frame)

          #cv2.imshow('frame',frame)
          if cv2.waitKey(2) & 0xFF == ord('q'):
               break
     else:
          break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()