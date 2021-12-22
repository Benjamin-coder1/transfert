import pyrealsense2 as rs
import numpy as np
import cv2, sys 
sys.path.append("..")
import Tools

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
            Tools.disp('Error connecting RealSense cameras: ' + str(e), 0)
            

    def setup(self):
        for device in self.devices:
            pipeline = rs.pipeline()
            config = rs.config()
            id_device = device.get_info(rs.camera_info.serial_number)
            
            try:                
                config.enable_device(id_device)
                config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)                
            except Exception as e:
                Tools.disp('Error enabling RealSense streams: ' + str(e), 0)

            try:
                pipeline.start(config)
                self.pipelines.append(pipeline)
                if CAMERAS[int(id_device)] == "FRONT" :
                    Tools.disp('FRONT connected', 1)
                    self.pipelineF = pipeline
                elif CAMERAS[int(id_device)] == "SIDER" :                
                    self.pipelineR = pipeline
                    Tools.disp('RIGHT connected', 1)
                elif CAMERAS[int(id_device)] == "SIDEL" : 
                    self.pipelineL = pipeline
                    Tools.disp('LEF connected', 1)
            except Exception as e:
                Tools.disp('Error starting pipeline: ' + str(e) , 0)

    def disp(self, k):        
        for i in range(k) : 
            images = []
            for pipeline in [self.pipelineF, self.pipelineL] :
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                depth_image = np.asanyarray(rs.colorizer().colorize(depth_frame).get_data())
                color_image = np.asanyarray(color_frame.get_data())                
                images.append(np.concatenate((color_image,depth_image), axis=0)) 
                
            # Disp
            image = images[0]
            for i in range(1,len(images)) :
                image = np.concatenate((image, images[i]), axis = 1 )
            cv2.imshow('Image', image)
            cv2.waitKey(1)
    


    
