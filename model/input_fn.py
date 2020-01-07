import cv2
import os
import numpy as np



calib_image_dir = "./calib/"
calib_image_list = "./calib/calib.txt"
calib_batch_size = 1
def calib_input(iter):
  images = []
  line = open(calib_image_list).readlines()
  for index in range(0, calib_batch_size):
    curline = line[iter * calib_batch_size + index]
    calib_image_name = curline.strip()

    # read image as grayscale, returns numpy array (28,28)
    image = cv2.imread(calib_image_dir + calib_image_name)

    # scale the pixel values to range 0 to 1.0
    image = image/255.0
    # print(image.shape)
    # reshape numpy array to be (28,28,1)
    image = image.reshape((image.shape[0], image.shape[1], 3))
    images.append(image)
  return {"x": images}