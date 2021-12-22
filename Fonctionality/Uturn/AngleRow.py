import cv2
import numpy as np                       
import pyrealsense2 as rs   
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Clustering
dbscan = DBSCAN(eps=0.01, min_samples=15)

def clustering(x,y) : 
    X = np.concatenate((np.reshape(x, (-1,1)), np.reshape(y, (-1,1))), 1)
    ind = (np.random.rand(1, np.size(x)) > 0.95)
    Xlimited = X[ind[0],:]
    label = dbscan.fit(Xlimited).labels_  
    Cluster = []
    for lab in np.unique(label) : 
        x = Xlimited[label == lab, 0]
        y = Xlimited[label == lab, 1]  
        Cluster.append( [x,y])
    return Cluster

def getPointCloud(depth_frame) : 
    pc = rs.pointcloud()    
    points = pc.calculate(depth_frame)
    w = rs.video_frame(depth_frame).width
    h = rs.video_frame(depth_frame).height
    return np.asanyarray(points.get_vertices()).view(np.float32).reshape(h, w, 3)

def fittage(x,y) : 
    parametres = np.polyfit(x, y, 1) 
    angle = round(np.arctan(parametres[0]) * 180/np.pi, 2)
    return parametres[0], parametres[1], round(angle,1)   


# Connexion camera
print('Connexion realsense Camera ... ')
pipe, align, cfg = [rs.pipeline() , rs.align(rs.stream.color), rs.config()]
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipe.start(cfg)
print('Aligned')  

# display 
plt.ion()
fig, ax = plt.subplots()
ax.grid(True)
lstScatter,lstPlot = [[],[]]
for i in range(10) : 
    lstScatter.append(ax.scatter([],[], marker='x'))
    lstPlot.append(ax.plot([],[], color='k')[0])
plt.xlim(-1/2,1/2)
plt.ylim(-1/2,1/2)
info = plt.text(0.5, 1.05, "" , ha="center", va="center", transform=ax.transAxes)
plt.draw()

while True : 
    ## Connection to the Camera
    frameset = pipe.wait_for_frames()
    depth_frame = frameset.get_depth_frame()
    verts = np.reshape( getPointCloud(depth_frame), (480*640, 3))

    ## Isolation of points
    cond1 = np.prod(verts, 1) != 0 
    cond2 = np.abs(verts[:,0]) < 1/2
    cond3 = np.abs(verts[:,2]) < 1/2
    cond4 = verts[:,1] < 0 
    ind = cond1 & cond2 & cond3 & cond4

    # Get the points 
    x = verts[ind,0]
    y = verts[ind,2]
    z = verts[ind,1] 

    # Clustering 
    listLegend1, listLegend2 = [(), ()]
    Cluster = clustering(x,y)
    for i in range(10) : 
        lstScatter[i].set_offsets(np.c_[[],[]])
        lstPlot[i].set_xdata([]) 
        lstPlot[i].set_ydata([])  
        
    i = 0
    for cluster in Cluster :   
        if np.size(cluster[0]) > 100 :     
            center, size, angle = cv2.minAreaRect(np.concatenate((np.reshape(cluster[0], (-1,1)), np.reshape(cluster[1], (-1,1))), 1))
            angle = round(angle,1)
            lstScatter[i].set_offsets(np.c_[cluster[0],cluster[1]])
            vertices= cv2.boxPoints((center, size, angle))
            lstPlot[i].set_xdata(vertices[:,0]) 
            lstPlot[i].set_ydata(vertices[:,1]) 
            listLegend1 += (lstScatter[i],)
            listLegend2 += (str(np.size(cluster[0])) + " / " + str(angle) + "°",)
            i = i + 1
      

    plt.legend(listLegend1, listLegend2)
    fig.canvas.draw_idle()
    plt.pause(0.1)
    
     

    



