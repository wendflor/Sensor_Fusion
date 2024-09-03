import numpy as np
import os

def convert_bin_to_npy(input_file_path):
    point_cloud = np.fromfile(input_file_path, dtype=np.float32)

    points = point_cloud[1:].reshape(-1, 3)  # Expecting  XYZ-Pointcloud

    file_name_with_extension = os.path.basename(input_file_path)


    output_file_path = f'sensor_data/Visionerf/point_clouds/{os.path.splitext(file_name_with_extension)[0]}.npy'

    np.save(output_file_path, points)

    print(f"Point Clouds saved as .npy-arrays in'{output_file_path}'.") 

    return points

if __name__ == '__main__':
    #  PATH to .bin-file
    input_file_path = 'sensor_data/Visionerf/XYZcloud_13_17_03.bin'
    points = convert_bin_to_npy(input_file_path)
    print(points)
    