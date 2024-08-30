import numpy as np 
import cv2

depth_img = cv2.imread('output_depth_image.png')
rgb_img = cv2.imread('sensor_data/zed/calib_zed_08_23_13_17.png')

print(depth_img.shape)
print(rgb_img.shape)

cv2.imshow('Tiefenbild', depth_img)
cv2.imshow('Farbbild', rgb_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
