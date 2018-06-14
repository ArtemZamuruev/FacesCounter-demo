import cv2
import utils
from pyfacesdk import Image8U
import config


class CamToolkit:

	def __init__(self, source):
		self.capture = cv2.VideoCapture(source)
		self.fps = int(self.capture.get(cv2.CAP_PROP_FPS))
		if self.fps == 0:
			utils.log("No video found at this source: %s" % str(source))
		self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
		self.scale_coef = config.video_scale

		self.video_dict = {
			"w": int(self.width*self.scale_coef),
			"h": int(self.height*self.scale_coef),
			"fps": self.fps,
			"roi": None
		}


	def get_frame(self, scaled=True):
		status, frame = self.capture.read()
		if float(self.scale_coef) != 1.0 and scaled:
			frame = cv2.resize(frame, (0, 0), fx=self.scale_coef, fy=self.scale_coef)
		return frame

	def get_frame_8u(self, scaled=True):
		return Image8U(self.get_frame())


