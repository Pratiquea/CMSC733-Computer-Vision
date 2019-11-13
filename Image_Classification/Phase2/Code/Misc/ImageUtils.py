import numpy as np
# import cv2
# import random
# import skimage
# import PIL
import sys

def StandardizeInputs(IMG):
	IMG /= 255.0
	IMG -= 0.5
	IMG *= 2.0
	return IMG