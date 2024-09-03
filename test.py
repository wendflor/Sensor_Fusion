import numpy as np
import cv2
import matplotlib.pyplot as plt


from read_calibration_file import read_yaml_file

img_rgb_path = 'sensor_data/zed/images/zed_calib_0000.png'

img_bgr = cv2.imread(img_rgb_path)

    # convert BGR to RGB
rgb_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


calibration_data = read_yaml_file(path='sensor_data/zed/left_camera_calibration_parameters.yaml')

 
rgb_image_undist = cv2.undistort(
    src=rgb_image, 
    cameraMatrix=calibration_data['camera_matrix'],
    distCoeffs=calibration_data['distortion_coefficients']
        )

plt.figure(num='Dist - undist')
plt.subplot(1, 2, 1)
plt.title('raw image')
plt.imshow(rgb_image)
plt.subplot(1, 2, 2)
plt.title('undist image')
plt.imshow(rgb_image_undist)
plt.show()