import matplotlib.pyplot as plt 
import numpy as np 
import pyrealsense2 as rs
import matplotlib.patches as patches
from statistics import mode
import math, json, cv2, torch, sys, time
from PIL import Image
sys.path.append("../StateMachine")
import realsense


_, ax = plt.subplots(nrows=1, ncols=2)


def detectTrunc(pipelineR, pipelineL, threshold, modelTrunc ) : 
    """ This function allow to detect the position of the trunc on the side cameras
            pipelineR - pipeline for the righ side 
            pepelineL - pipeline for the Left side 
            threshold - threshold for the computation of depth 
            modelTrunc - Model for detecting trunc """

    # Get data from cameras  
    frameR = pipelineR.wait_for_frames().get_depth_frame()
    frameL = pipelineL.wait_for_frames().get_depth_frame()
    imageR = np.asanyarray(rs.colorizer().colorize(frameR).get_data())
    imageL = np.asanyarray(rs.colorizer().colorize(frameL).get_data())

    # Concatenate 
    nbLine = np.shape(imageR)[0]
    image = np.concatenate((imageR[int(nbLine/2):, :,:], imageL[:int(nbLine/2), :,:]), axis=0)

    # Make prediction 
    results = modelTrunc(Image.fromarray(image, 'RGB'))            
    results = json.loads(results.pandas().xyxy[0].to_json(orient="records"))
    boxes = []
    for res in results : 
        if res["confidence"] > 0.1 : 
            boxes.append( [int(res["xmin"]), int(res["ymin"]), int(res["xmax"]), int(res["ymax"]) ])
    
    # Save data 
    zR, xR = [], []
    zL, xL = [], []  


    if len(boxes) >= 1:

        # Compute point cloud 
        pc = rs.pointcloud()    
        w, h = rs.video_frame(frameR).width, rs.video_frame(frameR).height
        point_cloudR =  np.asanyarray(pc.calculate(frameR).get_vertices()).view(np.float32).reshape(h, w, 3)
        point_cloudL =  np.asanyarray(pc.calculate(frameL).get_vertices()).view(np.float32).reshape(h, w, 3)
        
        for box in boxes : 
            # Get bounding boxes
            x0, y0, x1, y1 = box
            
            # draw bounding boxes on image
            image = cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 0), 2)

            if x0 > nbLine/2 : 
                # Get the position for the RIGHT camera
                x0, x1 = x0 + int(nbLine/2), x1 + int(nbLine/2)
                hist_data = (point_cloudR[y0:y1, x0:x1, 2]).flatten()
                n, bins, _ = ax[0].hist(hist_data, bins=100, range=[0, 4], histtype='step')

                # pick peak in range threshold
                start_ind, end_ind = 0, 0
                for ind in range(len(bins)):
                    if round(bins[ind], 1) <= threshold[0]:
                        start_ind = ind
                    elif round(bins[ind], 1) >= threshold[1]:
                        end_ind = ind
                        break

                max_ind = np.argmax(np.array([n[ind] for ind in range(start_ind, end_ind)])) + start_ind
                try : zR.append( mode([i for i in hist_data if (i >= bins[max_ind]) and (i <= bins[max_ind + 1])])  )
                except: continue

                # find point with closest depth to average trunk depth
                depth = abs(point_cloudR[y0:y1, x0:x1, 2] - zR[-1])
                closest = np.array(np.unravel_index(np.argmin(depth, axis=None), depth.shape))
                closest[0] += y0
                closest[1] += x0

                # Save data 
                xR.append(point_cloudR[closest[0]][closest[1]][0])

            else : 
                # Get the position for the LEFT camera
                hist_data = (point_cloudL[y0:y1, x0:x1, 2]).flatten()
                n, bins, _ = ax[0].hist(hist_data, bins=100, range=[0, 4], histtype='step')

                # pick peak in range threshold
                start_ind, end_ind = 0, 0
                for ind in range(len(bins)):
                    if round(bins[ind], 1) <= threshold[0]:
                        start_ind = ind
                    elif round(bins[ind], 1) >= threshold[1]:
                        end_ind = ind
                        break

                max_ind = np.argmax(np.array([n[ind] for ind in range(start_ind, end_ind)])) + start_ind
                try : zL.append( mode([i for i in hist_data if (i >= bins[max_ind]) and (i <= bins[max_ind + 1])])  )
                except: continue

                # find point with closest depth to average trunk depth
                depth = abs(point_cloudR[y0:y1, x0:x1, 2] - zL[-1])
                closest = np.array(np.unravel_index(np.argmin(depth, axis=None), depth.shape))
                closest[0] += y0
                closest[1] += x0

                # Save data 
                xL.append(point_cloudR[closest[1]][closest[0]][0])
        
    cv2.imshow('image', image)
    cv2.waitKey(1)
    return (zR, xR) if len(zR) != 0 else False, (zL, xL) if len(zL) != 0 else False

        
 


# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best_v03.pt')
# Load camera 
camera = realsense.RealSense()
# Launch code
while True : 
    value = detectTrunc(camera.pipelineF, camera.pipelineF, (0.2, 1.5), model ) 
    print( value )
    time.sleep(1e-2)
