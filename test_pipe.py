# import the necessary packages
from __future__ import print_function
from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS
from picamera.array import PiRGBArray
from picamera import PiCamera
import argparse
import imutils
import time
import cv2

from threading import Thread

prev_frame = 0
new_frame = 0

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=100,
	help="# of frames to loop over for FPS test")
ap.add_argument("-d", "--display", type=int, default=-1,
	help="Whether or not frames should be displayed")
args = vars(ap.parse_args())
# initialize the camera and stream
camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(320, 240))
stream = camera.capture_continuous(rawCapture, format="bgr",
	use_video_port=True)

# allow the camera to warmup and start the FPS counter
print("[INFO] sampling frames from `picamera` module...")
fps = FPS().start()
# loop over some frames
for (i, f) in enumerate(stream):
	# grab the frame from the stream and resize it to have a maximum
	# width of 400 pixels
	frame = f.array
	frame = imutils.resize(frame, width=400)
	
	new_frame = time.time()
	fps = int(1/(new_frame-prev_frame))
	prev_frame = new_frame
	cv2.putText(frame ,str(fps),(0,80),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,255),2)
	# check to see if the frame should be displayed to our screen
	cv2.imshow("Frame", frame)
	if cv2.waitKey(1) & 0xFF == ord('q'): break
	# clear the stream in preparation for the next frame and update
	# the FPS counter
	rawCapture.truncate(0)
	fps.update()
	# check to see if the desired number of frames have been reached
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
stream.close()
rawCapture.close()
camera.close()

print("[INFO] sampling THREADED frames from `picamera` module...")
vs = PiVideoStream().start()
fps = FPS().start()
# loop over some frames...this time using the threaded stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	# check to see if the frame should be displayed to our screen
	new_frame = time.time()
	fps = int(1/(new_frame-prev_frame))
	prev_frame = new_frame
	cv2.putText(frame ,str(fps),(0,80),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,255),2)
	
	cv2.imshow("Frame", frame)
	if cv2.waitKey(1) & 0xFF == ord('q'): break
	# update the FPS counter
	fps.update()
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()