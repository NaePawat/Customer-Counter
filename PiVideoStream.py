# import the necessary packages
from picamera.array import PiRGBArray
from picamera.array import PiYUVArray
from picamera import PiCamera
from threading import Thread
import cv2

class PiVideoStream:
	def __init__(self, resolution=(608, 400), framerate=32, **kwargs):
		self.camera = PiCamera()
		self.camera.resolution = resolution
		self.camera.framerate = framerate
		self.camera.brightness = 60
		for (arg, value) in kwargs.items():
			setattr(self.camera, arg, value)
		self.rawCapture = PiRGBArray(self.camera, size=resolution)
		self.stream = self.camera.capture_continuous(self.rawCapture,
			format="bgr", use_video_port=True)
		
		self.frame = None
		self.stopped = False

	def start(self):
		t = Thread(target=self.update, args=())
		t.daemon = True
		t.start()
		return self

	def update(self):
		for f in self.stream:
			self.frame = f.array
			self.rawCapture.truncate(0)
			if self.stopped:
				self.stream.close()
				self.rawCapture.close()
				self.camera.close()
				return

	def read(self):
		return self.frame

	def stop(self):
		self.stopped = True