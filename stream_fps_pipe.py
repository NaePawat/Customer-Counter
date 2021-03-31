import imutils
from PiVideoStream import PiVideoStream
import time
import cv2

vs = PiVideoStream().start()
prev_frame = 0
new_frame = 0
time.sleep(2.0)

while(True):
    frame = vs.read()
    frame = imutils.resize(frame, width=608)
    
    new_frame = time.time()
    fps = int(1/(new_frame-prev_frame))
    prev_frame = new_frame
    cv2.putText(frame ,str(fps),(0,80),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,255),2)

    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()