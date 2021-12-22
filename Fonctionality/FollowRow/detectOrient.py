import matplotlib.pyplot as plt 
import numpy as np 
import pyrealsense2 as rs
import matplotlib.patches as patches
from statistics import mode
import math, json, cv2, torch, sys, time
from PIL import Image
sys.path.append("../StateMachine")
import realsense
   

def detectOrient(pipelineF, modelOrient ) : 
    """ This function allow to detect the position of the trunc on the side cameras
            pipelineF - pipeline for the front camera 
            modelOrient - Model for detecting orientation """

    # Get data from cameras  
    frameF = pipelineF.wait_for_frames().get_depth_frame()
    imageF = np.asanyarray(rs.colorizer().colorize(frameF).get_data())

    # Save value 
    beta = False 
    boxes = []
   
    # Make prediction 
    results = modelOrient(Image.fromarray(imageF, 'RGB'))            
    results = json.loads(results.pandas().xyxy[0].to_json(orient="records"))  
    for res in results : 
        if res['name'] == 'forward' and res["confidence"] > 0.1 : 
            xmin, xmax, ymin, ymax = int( res['xmin']),  int(res['xmax']), int( res['ymin']),  int(res['ymax'])
            beta  = (87/640) * (320 - (xmin + (xmax - xmin)/2 )  )
            boxes.append( [res['confidence'], beta, xmin, ymin, xmax, ymax ] )
    
    # Display
    cv2.imshow('image', imageF)
    cv2.waitKey(1)

    # No result 
    if len(boxes) == 0 : return False

    # Give value back 
    boxes = np.array(boxes)
    ind = np.argmax(boxes, axis=0)[0]  
    imageF = cv2.rectangle(imageF, (int(boxes[ind,2]), int(boxes[ind,3])), (int(boxes[ind,4]), int(boxes[ind,5])), (0, 0, 0), 2)
      
    
    
    return boxes[ind, 1]

        
 


# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best_v03.pt')
# Load camera 
camera = realsense.RealSense()
# Launch code
while True : 
    value = detectOrient(camera.pipelineF,  model ) 
    print( "VALLLLLLLLLLUUUUUUUUUUEEEEEEEE " + str(value) )
    time.sleep(1e-2)
