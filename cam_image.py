import cv2
import os
import numpy as np


class frames(object):
	def __init__(self):
		self.cam = cv2.VideoCapture(0)
		self.frame = np.zeros((480, 640, 3), np.uint8)
		# self.processed_frame = np.zeros(())

	def capture_frames(self):
		stat, self.frame = self.cam.read()
		return self.frame
		
	def show_image(self, frame, name):
		cv2.imshow(name, frame)

	def check(self):
		a = cv2.waitKey(1)
		if(a == 27):
			return 0
		if(a == 32):
			return 2
		return 1

	def close(self):
		cv2.destroyAllWindows()
		self.cam.release()

	def save_image(self, no):
		cv2.imwrite(str(no)+'_frame.jpg', self.frame)
		# cv2.imwrite(str(no)+'_pr_frame.jpg', self.processed_frame)

	def process(self):
		grey = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)
		blur_gray = cv2.GaussianBlur(grey, (35,35), 0)
		stat, self.processed_frame = cv2.threshold(blur_gray, 255,127, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
		return self.processed_frame


