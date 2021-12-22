import sys, cv2, time, threading, pickle                   
import numpy as np                       
import pyrealsense2 as rs                 
sys.path.append("..")
import ConfigParameters as p 
import Tools

def getPointCloud(depth_frame) : 
    """ This function compute the point cloud from the depth frame """
    pc = rs.pointcloud()    
    points = pc.calculate(depth_frame)
    w = rs.video_frame(depth_frame).width
    h = rs.video_frame(depth_frame).height
    return np.asanyarray(points.get_vertices()).view(np.float32).reshape(h, w, 3)


def getFramset(pipe) : 
    """ This function allows to get the framset without waiting time """
    global frameset
    try : 
        frameset = pipe.wait_for_frames()
    except Exception as e :  
        Tools.disp( str(e.args[0]) , 0, p.EventLogger )     

    
def detect_obstacle(SaveLogFile = False) : 
    """  This function is made to detect obstacle from the front camera, it is using
    AI to detect obstacle from RGB camera then we use point cloud to localize obstacle
    The camera used is a Realsense DepthCamera D435  """

    ## Initialisation 
    global pipe, frameset
    Nbcsvt = 0    # Number of consecutive frames to stop the process
    kernelSize = 5       # size of middle kernel for 3D localization 
    frequency = 0      # frequency of processing 
    timeList = []    # list for frequency computation 
    inScaleFactor, meanVal , expected = [0.007843 , 127.53,  300]  # parameters for CNN
    NbFrame = 0   # Number of frames
    fileColor, filePointCloud, fileResult = [open(p.fileNameColor, "wb"),open(p.fileNamePointCloud, "wb"), open(p.fileNameResult, "wb")]  # files to save frames 

    ## Connection to the Camera
    pipe, align = Tools.ConnectCamera()

    ## Configuration of the CNN
    net = cv2.dnn.readNetFromONNX('best.onnx')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)    
    classNames = ("background", "aeroplane", "bicycle", "bird", "boat",
                "bottle", "bus", "car", "cat", "chair",
                "cow", "diningtable", "dog", "horse",
                "motorbike", "person", "pottedplant",
                "sheep", "sofa", "train", "tvmonitor")    

    ## Get first framset 
    Tools.disp( msg="Launch of the first framset", logger=p.EventLogger )
    GetFrameset = threading.Thread(target=getFramset, args=(pipe,))
    GetFrameset.start()    
    
    ## LOOP of detection 
    while not p.stop :

        # Get time for computing frequency 
        t1 = time.time() 
 
        # Get data from camera 
        GetFrameset.join()   
        color_frame, depth_frame = [frameset.get_color_frame(), align.process(frameset).get_depth_frame()]

        ## Launch framset for next image  
        GetFrameset = threading.Thread(target=getFramset, args=(pipe,))
        GetFrameset.start()

        # Point cloud computation 
        verts = getPointCloud(depth_frame)

        # Image processing  
        color = np.asanyarray(color_frame.get_data())
        height, width = color.shape[:2]        
        aspect = width / height
        resized_image = cv2.resize(color, (round(expected * aspect), expected))
        crop_start = round(expected * (aspect - 1) / 2)
        crop_img = resized_image[0:expected, crop_start:crop_start+expected]
        blob = cv2.dnn.blobFromImage(crop_img, inScaleFactor, (expected, expected), meanVal, False)
        net.setInput(blob, "data")
        detections = net.forward("detection_out")

        # Nb of detected object and saving 
        boudingList, nbObjct = [[], 0]

        # Extraction of the 3D localisation 
        while detections[0,0,nbObjct,2] > p.conf : 

            # Work only for bottles            
            label = detections[0,0,nbObjct,1]
            if classNames[int(label)] != "bottle" : 
                nbObjct += 1
                continue       

            # Get box bounding color             
            xmin  = int(expected*detections[0,0,nbObjct,3])
            ymin  = int(expected*detections[0,0,nbObjct,4])
            xmax  = int(expected*detections[0,0,nbObjct,5])
            ymax  = int(expected*detections[0,0,nbObjct,6]) 
           
            # Get box bounding depth 
            scale = height / expected
            xminD = int((expected*detections[0,0,nbObjct,3] + crop_start)*scale )
            xmaxD = int((expected*detections[0,0,nbObjct,5] + crop_start)*scale )
            yminD = int((expected*detections[0,0,nbObjct,4])*scale )
            ymaxD = int((expected*detections[0,0,nbObjct,6])*scale ) 

            # Compute the middle
            xmil = int( (xmax - xmin)/2 + xmin)
            ymil = int( (ymax - ymin)/2 + ymin)
            xmilD = int((xmil + crop_start)*scale )
            ymilD = int((ymil)*scale ) 
            
            # Localisation of the middle point 
            try : 
                # Looking the - Z axis 
                Tab = verts[yminD:ymaxD, xminD:xmaxD,:]
                Tab = np.reshape( Tab , (np.prod( np.shape(Tab)[0:2] ) , 3) )[:,2]
                z = min( Tab[Tab > 0] )

                # Look for the - X axis and - Y axis      
                done = False              
                for yD in range(ymilD - kernelSize, ymilD + kernelSize ) : 
                    for xD in range(xmilD - kernelSize, xmilD + kernelSize ) :            
                        if verts[yD, xD, 2] != 0 :
                            milyD, milxD = [yD, xD]      
                            y , x = [ verts[ milyD, milxD , 0 ] , -verts[ milyD, milxD , 1 ]]
                            milx, mily = np.array([milxD/scale - crop_start, milyD/scale]).astype(int)
                            boudingList.append( [xmin, ymin, xmax, ymax, milx, mily, x, y, z, frequency ])
                            done = True
                            break 
                    if done : break 

                # Decide if we stop or no
                if z < p.distMin and Nbcsvt < p.Nbcsvtmax :   Nbcsvt += 1 
                if z < p.distMin and Nbcsvt >= p.Nbcsvtmax :   p.stop = True                 

                # GUI dusplay 
                pos = str(round(x,2))+ ' ' + str(round(y,2)) + ' ' + str(round(z,2))
                cv2.circle(crop_img, ( milx , mily ), 3, (255,0,0), -1)
                cv2.rectangle(crop_img, (xmin, ymin), (xmax, ymax), (0, 0, 0), 2)
                cv2.putText(crop_img, pos, (xmin - 30, ymin - 10), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0) )

            except Exception as e :
                pass
                        
            # Next Object detected  
            nbObjct += 1
        
        # Saving of frames 
        if SaveLogFile and NbFrame%10 == 0 : 
            pickle.dump( boudingList, fileResult)
            pickle.dump( verts, filePointCloud)
            pickle.dump( color, fileColor)
        NbFrame += 1
         
        # Compute frequency 
        timeList.append( time.time() - t1)
        if len(timeList) > 2 : 
            frequency = round(1/np.mean(timeList[max(0,len(timeList)  - 15):len(timeList) - 1 ]),1)
            print(str(frequency) + " Hz")

        # diplay image 
        cv2.putText(crop_img, str( frequency ) + "Hz", (250,290), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0) )
        cv2.imshow('image', crop_img)
        cv2.waitKey(1)

    Tools.disp( "--***--  OBSTACLE DETECTED / STOP  --***-", 2)     
     


detect_obstacle()

