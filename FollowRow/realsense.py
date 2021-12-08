import pyrealsense2.pyrealsense2 as rs
from datetime import datetime as dt
import numpy as np
import pickle
import cv2
import os

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
                print( str(int(device.get_info(rs.camera_info.serial_number))) + ' connected' )
            except Exception as e:
                print('Error enabling RealSense streams: ', e)

            try:
                pipeline.start(config)
                self.pipelines.append(pipeline)
            except Exception as e:
                print('Error starting pipeline: ', e)

    def disp(self, k):        
        for i in range(k) : 
            images = []
            for pipeline in self.pipelines:
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                depth_image = np.asanyarray(rs.colorizer().colorize(depth_frame).get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                images.append(color_image) 
                
            # Disp
            image = images[0]
            for i in range(1,len(images)) :
                image = np.concatenate((image, images[i]), axis = 1 )
            cv2.imshow('Image', image)
            cv2.waitKey(1)
    

if __name__ == '__main__':
    capturer = RealSense()
    print('Initializing...')
    capturer.disp(10)
    
