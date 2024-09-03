import open3d as o3d
import numpy as np
import yaml
import matplotlib.pyplot as plt
import os

from convert_bin_to_npy import convert_bin_to_npy
from read_calibration_file import read_yaml_file

def transform_point_cloud_to_depth_image(rgb_image_legacy, point_cloud_npy, calibration_data,R,t, display = False):
    '''
    Input: 
    Output: -rgb image
            -depth image
            -RGBD Image pair [Aligned]
                Color [size=(2208,1242), channels=4, format=UInt8, device=CPU:0]
                Depth [size=(2208,1242), channels=1, format=UInt16, device=CPU:0]
    '''
    
    
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
    #print(depth_np_float32)
    depth_np_uint16 = depth_np_float32.astype(np.uint16)
    #print(depth_np_uint16)
    depth_image = o3d.t.geometry.Image(depth_np_uint16)

    if rgb_image.rows != depth_image.rows or rgb_image.columns != depth_image.columns:
        raise ValueError("Die Auflösungen des Farb- und Tiefenbildes stimmen nicht überein. Sie müssen gleich sein.")
    
    # RGB Image: Image[size={1242,2208}, channels=3, UInt8, CPU:0], Depth Image: Image[size={1242,2208}, channels=1, Float32, CPU:0]
    rgbd_image = o3d.t.geometry.RGBDImage(color=rgb_image, depth=depth_image)
    rgbd_image_legacy = rgbd_image.to_legacy()

    if display == True:
        plt.figure(num='RGBD image pair')
        plt.subplot(1, 2, 1)
        plt.title('color image')
        plt.imshow(rgbd_image_legacy.color)
        plt.subplot(1, 2, 2)
        plt.title('depth image')
        plt.imshow(rgbd_image_legacy.depth)
        plt.show()

    return depth_image, rgb_image, rgbd_image

def transform_RGBD_to_colored_point_cloud(rgbd_image, calibration_data, R, t, display= False):
    '''
    Input:  -rgbd_image in o3d tensor form
            -calibration_data from yaml file
            -Rotation matrix R as npy array
            -Translation vector t as npy array
            -display parameter (shows 3D point cloud if True)

    Output: -colored point cloud as o3d tensor geometry object    
    '''
    
    # Define camera intrinsic parameters
    # width, height = calibration_data['img_col_px'], calibration_data['img_row_px']  # Image resolution
    camera_matrix = calibration_data['camera_matrix']

    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t
    #extrinsic = T
    #intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2])
    #colored_pcd= o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image_legacy, intrinsic, extrinsic)

    intrinsic = o3d.core.Tensor(camera_matrix)
    extrinsic = o3d.core.Tensor(T)

    colored_pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic, extrinsic, depth_scale=1,depth_max = 600)

    if display == True:
        display_point_cloud = colored_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        o3d.visualization.draw_geometries([display_point_cloud.to_legacy()])  

    backprojected_RGBD_image = colored_pcd.project_to_rgbd_image(width=2208 , height=1242, intrinsics = intrinsic, extrinsics=extrinsic, depth_scale=1, depth_max=600)


    return colored_pcd, backprojected_RGBD_image
 
if __name__== '__main__':

    # Load camera intrinsic parameters
    intrinsics_path = 'sensor_data/zed/left_camera_calibration_parameters.yaml'
    calibration_data = read_yaml_file(path=intrinsics_path)
    
    camera_matrix=calibration_data['camera_matrix']
    distortion_coefficients=calibration_data['distortion_coefficients']

    # Load extrinsic parameters R and t of transformation from Visionerf to Zed
    R = np.load('sensor_data/extrinisic_calibration_data_visio2zed/03-09-2024-11-08-59/r.npy')
    t = np.load('sensor_data/extrinisic_calibration_data_visio2zed/03-09-2024-11-08-59/t.npy').reshape(-1)

    # Define directories
    image_dir = 'sensor_data/zed/images/'
    pointcloud_dir = 'sensor_data/Visionerf/point_clouds/'

    # Get a list of all image and point cloud files
    image_files = sorted(os.listdir(image_dir))
    pointcloud_files = sorted(os.listdir(pointcloud_dir))

    # Loop through all image files
    for image_file in image_files:
        # Check if the image file is in the correct format
        if image_file.startswith('zed_') and image_file.endswith('.png'):
            if 'calib' in image_file:
                # Handle calibration images zed_calib_0000.png
                base_name = 'zed_calib_'
                number = image_file.split('_')[-1].split('.')[0]  # Extracts 0000 from zed_calib_0000.png
                pointcloud_file = f'visionerf_calib_{number}.bin'
                calib = True
            else:
                # Handle regular images zed_0001.png
                base_name = 'zed_'
                number = image_file.split('_')[-1].split('.')[0]  # Extracts 0001 from zed_0001.png
                pointcloud_file = f'visionerf_{number}.bin'
                calib=False
            # Check if the corresponding point cloud file exists
            if pointcloud_file in pointcloud_files:
                # Full paths to the files
                image_path = os.path.join(image_dir, image_file)
                pointcloud_path = os.path.join(pointcloud_dir, pointcloud_file)

                # Call the processing code for the image and point cloud
                print(f"Processing image: {image_path} and point cloud: {pointcloud_path}")

                # Load rgb-image
                rgb_image = o3d.io.read_image(image_path)

                # Load the point cloud
                point_cloud_npy = convert_bin_to_npy(pointcloud_path)
                #print(point_cloud_npy)

                # Get different images in o3d tensor form
                depth_image, rgb_image, rgbd_image = transform_point_cloud_to_depth_image(rgb_image,point_cloud_npy, calibration_data, R, t, display=True)

                #Get colored point cloud as o3d tensor geometry object
                colored_point_cloud, backprojected_RGBD_image = transform_RGBD_to_colored_point_cloud(rgbd_image, calibration_data, R, t, display=True)
                '''
                # Create a new figure with a specific window title
                plt.figure(num='RGBD image pair')

                # First row, first column: Color image
                plt.subplot(2, 2, 1)  # 2x2 grid, position 1
                plt.title('Color Image')
                plt.imshow(rgbd_image.to_legacy().color)

                # First row, second column: Depth image
                plt.subplot(2, 2, 2)  # 2x2 grid, position 2
                plt.title('Depth Image')
                plt.imshow(rgbd_image.to_legacy().depth)

                # Second row, first column: Additional image 1
                plt.subplot(2, 2, 3)  # 2x2 grid, position 3
                plt.title('Color Image projected')
                plt.imshow(backprojected_RGBD_image.to_legacy().color)  # Replace with your additional image data

                # Second row, second column: Additional image 2
                plt.subplot(2, 2, 4)  # 2x2 grid, position 4
                plt.title('Depth Image projected')
                plt.imshow(backprojected_RGBD_image.to_legacy().depth)  # Replace with your additional image data

                # Display the plot
                plt.show()
                '''


                if calib == True:
                    # Save the color image
                    o3d.t.io.write_image(f'output_data/RGBD/color_calib_{number}.png', rgbd_image.color)

                    # Save the depth image
                    o3d.t.io.write_image(f'output_data/RGBD/depth_calib_{number}.png', rgbd_image.depth)    

                    # Save the tensor point cloud to different formats
                    o3d.t.io.write_point_cloud(f'output_data/colored_point_cloud/colored_point_cloud_calib_{number}.ply', colored_point_cloud)    # Save as PLY
                    o3d.t.io.write_point_cloud(f'output_data/colored_point_cloud/colored_point_cloud_calib_{number}.pcd', colored_point_cloud)    # Save as PCD
                    o3d.t.io.write_point_cloud(f'output_data/colored_point_cloud/colored_point_cloud_calib_{number}.xyzrgb', colored_point_cloud) # Save as XYZRGB with normalized rgb values                
                    
                    # Extract point positions as npy array
                    points = colored_point_cloud.point.positions.numpy()
                    # Extract point color as npy array
                    colors = colored_point_cloud.point.colors.numpy() # normalized color values
                    # Combine positions and colors into a single array
                    point_cloud_data = np.hstack((points, colors))
                    # Save the point cloud data as an NPY file
                    np.save(f'output_data/colored_point_cloud/colored_point_cloud_calib_{number}.npy', point_cloud_data)

                else:
                    # Save the color image
                    o3d.t.io.write_image(f'output_data/RGBD/color_{number}.png', rgbd_image.color)

                    # Save the depth image
                    o3d.t.io.write_image(f'output_data/RGBD/depth_{number}.png', rgbd_image.depth)

                    # Save the tensor point cloud to different formats
                    o3d.t.io.write_point_cloud(f'output_data/colored_point_cloud/colored_point_cloud_{number}.ply', colored_point_cloud)    # Save as PLY
                    o3d.t.io.write_point_cloud(f'output_data/colored_point_cloud/colored_point_cloud_{number}.pcd', colored_point_cloud)    # Save as PCD
                    o3d.t.io.write_point_cloud(f'output_data/colored_point_cloud/colored_point_cloud_{number}.xyzrgb', colored_point_cloud) # Save as XYZRGB with normalized rgb values

                    # Extract point positions as npy array
                    points = colored_point_cloud.point.positions.numpy()
                    # Extract point color as npy array
                    colors = colored_point_cloud.point.colors.numpy() # normalized color values
                    # Combine positions and colors into a single array
                    point_cloud_data = np.hstack((points, colors))
                    # Save the point cloud data as an NPY file
                    np.save(f'output_data/colored_point_cloud/colored_point_cloud_{number}.npy', point_cloud_data)

                

                print("RGBD image pair saved in: output_data/RGBD/. Colored point cloud saved in: output_data/colored_point_cloud/")



                
            else:
                print(f"No corresponding point cloud file found for {image_file}.")
        






