import pyrealsense2.pyrealsense2 as rs
from datetime import datetime as dt
import numpy as np
import pickle
import cv2
import os

# Directories
SIDER_CAM_RGB = 'C:\\Users\\DUDU\\Desktop\\RealSense\\data collection\\sensor captures\\data\\rgb_sider_cam'
SIDER_CAM_DEPTH = 'C:\\Users\\DUDU\\Desktop\\RealSense\\data collection\\sensor captures\\data\\depth_sider_cam'

SIDEL_CAM_RGB = 'C:\\Users\\DUDU\\Desktop\\RealSense\\data collection\\sensor captures\\data\\rgb_sidel_cam'
SIDEL_CAM_DEPTH = 'C:\\Users\\DUDU\\Desktop\\RealSense\\data collection\\sensor captures\\data\\depth_sidel_cam'

FRONT_CAM_RGB = 'C:\\Users\\DUDU\\Desktop\\RealSense\\data collection\\sensor captures\\data\\rgb_front_cam'
FRONT_CAM_DEPTH = 'C:\\Users\\DUDU\\Desktop\\RealSense\\data collection\\sensor captures\\data\\depth_front_cam'

# Cameras and their serial numbers
CAMERAS = {117122250194: 'SIDER',
           941322072432: 'FRONT',
           105322251893: 'SIDEL'}

class RealSense:
    def __init__(self):
        try:
            self.ctx = rs.context()
            self.devices = self.ctx.query_devices()
            self.pipelines = []
            self.last_side_rgb = None
            self.last_side_depth = None
            self.last_front_rgb = None
            self.last_front_deoth = None
            self.setup()

        except Exception as e:
            print('RealSense setup error: ', e)

    def setup(self):
        for device in self.devices:
            pipeline = rs.pipeline()
            config = rs.config()

            try:
                config.enable_device(device.get_info(rs.camera_info.serial_number))
                config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            except Exception as e:
                print('Error enabling RealSense streams: ', e)

            try:
                pipeline.start(config)
                self.pipelines.append(pipeline)
            except Exception as e:
                print('Error starting pipeline: ', e)

    def save_capture(self, device, date, image_type, image, frame):
        serial_number = int(device.get_info(rs.camera_info.serial_number))
        cam_direction = CAMERAS[serial_number]

        if cam_direction == 'SIDER':
            if image_type == 'rgb':
                self.last_side_rgb = f'{SIDER_CAM_RGB}/side_rgb_{date}.jpg'
                cv2.imwrite(self.last_side_rgb, image)
                print('Saved image to side RGB-r directory.')

            elif image_type == 'depth':
                self.last_side_depth = f'{SIDER_CAM_DEPTH}/side_depth_{date}.jpg'
                cv2.imwrite(self.last_side_depth, image)
                self.save_depth_data(image, f'{SIDER_CAM_DEPTH}/matrices/side_depth_matrix_{date}.pkl')
                self.save_point_cloud(frame, f'{SIDER_CAM_DEPTH}/matrices/side_point_cloud_{date}.pkl')
                print('Saved image to side depth-r directory.')

        elif cam_direction == 'SIDEL':
            if image_type == 'rgb':
                self.last_side_rgb = f'{SIDEL_CAM_RGB}/side_rgb_{date}.jpg'
                cv2.imwrite(self.last_side_rgb, image)
                print('Saved image to side RGB-l directory.')

            elif image_type == 'depth':
                self.last_side_depth = f'{SIDEL_CAM_DEPTH}/side_depth_{date}.jpg'
                cv2.imwrite(self.last_side_depth, image)
                self.save_depth_data(image, f'{SIDEL_CAM_DEPTH}/matrices/side_depth_matrix_{date}.pkl')
                self.save_point_cloud(frame, f'{SIDEL_CAM_DEPTH}/matrices/side_point_cloud_{date}.pkl')
                print('Saved image to side depth-l directory.')

        elif cam_direction == 'FRONT':
            if image_type == 'rgb':
                self.last_front_rgb = f'{FRONT_CAM_RGB}/front_rgb_{date}.jpg'
                cv2.imwrite(self.last_front_rgb, image)
                print('Saved image to front RGB directory.')

            elif image_type == 'depth':
                self.last_front_depth = f'{FRONT_CAM_DEPTH}/front_depth_{date}.jpg'
                cv2.imwrite(self.last_front_depth, image)
                self.save_depth_data(image, f'{FRONT_CAM_DEPTH}/matrices/front_depth_matrix_{date}.pkl')
                self.save_point_cloud(frame, f'{FRONT_CAM_DEPTH}/matrices/front_point_cloud_{date}.pkl')
                print('Saved image to front depth directory.')
        return image

    def save_depth_data(self, depth_frame, path):
        depth_matrix = np.asanyarray(depth_frame)

        with open(f'{path}', 'wb') as m:
            pickle.dump(depth_matrix, m)

    def save_point_cloud(self, depth_frame, path):
        pc = rs.pointcloud()
        points = pc.calculate(depth_frame)
        v = points.get_vertices()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)
        with open(f'{path}', 'wb') as m:
            pickle.dump(verts, m)

    def capture(self, date):
        #        assert(len(self.pipelines) == 2), 'Less than two RealSense cameras connected'
        images = []
        for pipeline in self.pipelines:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            depth_image = np.asanyarray(rs.colorizer().colorize(depth_frame).get_data())
            color_image = np.asanyarray(color_frame.get_data())

            device = pipeline.get_active_profile().get_device()

            # Save RGB capture
            image = self.save_capture(device=device, date=date, image_type='rgb', image=color_image, frame=depth_frame)
            images.append(image)

            # Save depth capture
            self.save_capture(device=device, date=date, image_type='depth', image=depth_image, frame=depth_frame)

        # Disp
        image = images[0]
        for i in range(1,len(images)) :
            image = np.concatenate((image, images[i]), axis = 1 )
        cv2.imshow('Image', image)
        cv2.waitKey(1)

    def delete_last(self):
        os.remove(self.last_side_rgb)
        os.remove(self.last_side_depth)
        os.remove(self.last_front_rgb)
        os.remove(self.last_front_depth)


if __name__ == '__main__':
    capturer = RealSense()
    print('Initializing...')

    # Capture with dummy date
    capturer.capture('2021-01-01-0123456789')