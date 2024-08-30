import open3d as o3d
import numpy as np
import yaml
import matplotlib.pyplot as plt

from convert_bin_to_npy import convert_bin_to_npy
from read_calibration_file import read_yaml_file

def transform_point_cloud_to_depth_image(rgb_image_legacy, point_cloud_npy, calibration_data,R,t):
    # Convert o3d Image to o3d Tensor Image
    rgb_image = o3d.t.geometry.Image.from_legacy(rgb_image_legacy)

    # Define camera intrinsic parameters
    width, height = calibration_data['img_col_px'], calibration_data['img_row_px']  # Image resolution
    camera_matrix = calibration_data['camera_matrix']
    # Load extrinsic parameters R and t of transformation from Visionerf to Zed
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t
    intrinsic = o3d.core.Tensor(camera_matrix)
    extrinsic = o3d.core.Tensor(T)
    # Convert point cloud to o3d tensor
    pcd = o3d.t.geometry.PointCloud(point_cloud_npy)
 
    # Convert the point cloud to a depth image
    depth_image = pcd.project_to_depth_image(width,height,intrinsic,extrinsic,depth_scale=1.0,depth_max=600.0)
    #depth_image_legacy = depth_image.to_legacy()
    # Convert open3d image to numpy array
    depth_np_float32 = np.asarray(depth_image)
    print(depth_np_float32)
    depth_np_uint16 = depth_np_float32.astype(np.uint16)
    print(depth_np_uint16)
    depth_image = o3d.t.geometry.Image(depth_np_uint16)





    if rgb_image.rows != depth_image.rows or rgb_image.columns != depth_image.columns:
        raise ValueError("Die Auflösungen des Farb- und Tiefenbildes stimmen nicht überein. Sie müssen gleich sein.")
    
    # RGB Image: Image[size={1242,2208}, channels=3, UInt8, CPU:0], Depth Image: Image[size={1242,2208}, channels=1, Float32, CPU:0]
    rgbd_image = o3d.t.geometry.RGBDImage(color=rgb_image, depth=depth_image)

    
    # Save or visualize the depth image
    #o3d.io.write_image("output_depth_image.jpg", depth_image.to_legacy())
    #o3d.visualization.draw_geometries([depth_image.to_legacy()])

    return depth_image, rgb_image, rgbd_image





if __name__== '__main__':
    # Load rgb-image
    rgb_image = o3d.io.read_image('sensor_data/zed/calib_zed_08_23_13_17_yellow_edge.png')

    # Load the point cloud
    point_cloud_path = 'sensor_data/Visionerf/XYZcloud_13_17_03.bin'
    point_cloud_npy = convert_bin_to_npy(point_cloud_path)
    #print(point_cloud_npy)

    # Load camera intrinsic parameters
    intrinsics_path = 'sensor_data/zed/left_camera_calibration_parameters.yaml'
    calibration_data = read_yaml_file(path=intrinsics_path)
    
    camera_matrix=calibration_data['camera_matrix']
    distortion_coefficients=calibration_data['distortion_coefficients']

    # Load extrinsic parameters R and t of transformation from Visionerf to Zed
    R = np.load('sensor_data/extrinisic_calibration_data_visio2zed/29-08-2024-17-39-22/r.npy')
    t = np.load('sensor_data/extrinisic_calibration_data_visio2zed/29-08-2024-17-39-22/t.npy').reshape(-1)

    # Get different images as open3D tensor
    depth_image, rgb_image, rgbd_image = transform_point_cloud_to_depth_image(rgb_image,point_cloud_npy, calibration_data, R, t)
    rgbd_image_legacy = rgbd_image.to_legacy()
    print(rgbd_image)
    #print(rgbd_image_legacy)

    
    
    plt.subplot(1, 2, 1)
    plt.title('color image')
    plt.imshow(rgbd_image_legacy.color)
    plt.subplot(1, 2, 2)
    plt.title('depth image')
    plt.imshow(rgbd_image_legacy.depth)
    plt.show()
    
    # Define camera intrinsic parameters
    width, height = calibration_data['img_col_px'], calibration_data['img_row_px']  # Image resolution
    camera_matrix = calibration_data['camera_matrix']


    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t
    #extrinsic = T
    #intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2])
    #colored_pcd= o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image_legacy, intrinsic, extrinsic)

    intrinsic = o3d.core.Tensor(camera_matrix)
    extrinsic = o3d.core.Tensor(T)

    colored_pcd= o3d.t.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic, extrinsic, depth_scale=1,depth_max = 600)
    colored_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    print(colored_pcd)
    o3d.visualization.draw_geometries([colored_pcd.to_legacy()])

