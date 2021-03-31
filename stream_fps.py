import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)
prev_frame = 0
new_frame = 0
cap.set(3, 608)
cap.set(4, 400)
cap.set(10, 60)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    new_frame = time.time()
    fps = int(1/(new_frame-prev_frame))
    prev_frame = new_frame
    cv2.putText(frame ,str(fps),(0,80),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,255),2)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()