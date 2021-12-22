import time, cmath, threading, cv2
import numpy as np
import matplotlib.pyplot as plt
from Simu import *
from sklearn.cluster import DBSCAN

# This module is used to control the simulator, it simulate a car mooving and turning in a field
# The attribute used to control the car are the following 
#       rec.angle - set the steering angle of the car
#       rec.speed  - set the speed of the car 
#       rec.on -    set True to start vehicle / False to stop it 
# The attribute used to see what the car can see is 
#       rec.pc - return the point cloud seen by the camera in top view (2D)

#############  PARAMETERS OF THE SIMULATOR #################
class Variable() : 
    def __init__(self) : 
        self.l0 = 3     #  space between the row 
        self.L, self.l, self.C0, self.initialAngle = 3.3, 1.4, (6.2,-6), 1   # 16.7, 6), 230 Initial position and shape of the vehicle 
        self.jump = 4  # number of rows to jump
        self.sleepingTime = 2e-2  
        self.alpha = 45*np.pi/180    # steering angle for 90 degres
        self.maxSpeed = 2     # max speed of the vehicle  m/s
        self.deltX, self.deltY = 0.1, 0.5  # security on the side
        self.state = -1  # state of the U turn 0/1/2/3
        self.nbRow = 6
var = Variable()

#############  BASIC FUNCTIONS #################
# Fitage 1D line 
def fittage(x,y) : 
    """ return the line fitted under the format : parametres[0]*x parametres[1]"""
    parametres = np.polyfit(x, y, 1) 
    angle = round(np.arctan(parametres[0]) * 180/np.pi, 2)
    return parametres[0], parametres[1], round(angle,1)   

# Clustering
dbscan = DBSCAN(eps=1.2, min_samples=2)
def clustering(x,y) : 
    """ Clusterize the point cloud and return a list of the clusters"""
    if np.size(x)*np.size(y) == 0 : 
        return [] 
    X = np.concatenate((np.reshape(x, (-1,1)), np.reshape(y, (-1,1))), 1)
    label = dbscan.fit(X).labels_      
    Cluster = []
    for lab in np.unique(label) : 
        x = X[label == lab, 0]
        y = X[label == lab, 1]  
        Cluster.append( [x,y])
    return Cluster

# Displaying
# fig2, ax = plt.subplots()
# ax.grid(True)
# lstScatter,lstPlot = [[],[]]
# for i in range(10) : 
#     lstScatter.append(ax.scatter([],[], marker='x'))
#     lstPlot.append(ax.plot([],[], color='k')[0])
# plt.xlim(-10,10)
# plt.ylim(0,10)
# plt.draw()

#############  FUNCTION BY STEP #################

# Global variable for exchange between step 
a = 0

def step1(rec) : 
    """ First step for when we leave the row """
    deltX = 0   # distance betewwen the vehicle and the right row
    d0 = 0  # distance to the end of the row
    global a   # save distance to the right
    Rint = var.L / np.tan(var.alpha)  
    Rext = np.sqrt( var.L**2 + (var.l + Rint)**2 ) 
  

    def computePosition() : 
        while not hasattr(rec, 'pcF') : 
            time.sleep(1)
        # Compute cluster 
        pcF = rec.pcF
        clusterF = clustering(pcF[0], pcF[1])  
        # Do we see something ? 
        if np.size(pcF) == 0 : 
            print("We are blind !")
            return False
        # Extract information of the side and the front 
        maxValY = []
        for cluster in clusterF :
            meanValX = np.mean(cluster[0])
            maxValY.append(np.max(cluster[1]))
            if meanValX > 0 and meanValX < var.l0 : 
                deltX = meanValX 
        d0 = np.mean(maxValY)       
        return [d0, deltX]


    while True : 
        # Measure data we need
        val = computePosition()
        if val == False : 
            print("We are blind now.")
            break
        d0, deltX = val
        if d0 > 5 : 
            rec.speed = var.maxSpeed
            time.sleep(1e-2)
            continue
        
        print('Close enought of the end of the row. ')
        # LEAVING ROW
        wx = var.l/2 + Rint       
        wy1 = var.L + d0 + var.deltY - np.sqrt(Rint**2 - (deltX - wx)**2)   
        a = deltX - var.l/2
        # ENTERING ROW
        x = (Rext + var.deltX) - var.l0
        wy2 = var.deltY - np.sqrt(Rint**2 - x**2)
        wy2 = var.L + d0 + wy2
        
        wy = np.max([wy1,wy2])
        print('Distance before stopping : ' + str(wy) + 'm') 

        rec.speed = var.maxSpeed
        time.sleep(wy/var.maxSpeed)
        rec.speed = 1e-3
        rec.angle = var.alpha
        break
    print('Step 1 : done ') 

def step2(rec) : 
    """ In this step we do the first 90 degrees turn """
    def computeAngle() : 
        while not hasattr(rec, 'pcS')  : 
            time.sleep(1)
        # Compute cluster 
        pcS = rec.pcS
        clusterS = clustering(pcS[0], pcS[1])  
        # Do we see something ? 
        if np.size(pcS) == 0 : 
            print("We are blind !")
            return False
        # Extract information of the angle  
        angleList = []
        for cluster in clusterS :
            if np.size(cluster[0]) > 20 : 
                _, _, angle = fittage(cluster[0], cluster[1])
                angleList.append(angle)
        return angleList

    while True : 
        # We do turn when we can see 
        angleList = computeAngle() 
        if angleList == [] or np.mean(angleList) < 80 : 
            rec.speed = var.maxSpeed/4
            rec.angle = var.alpha
            time.sleep(1e-2)
            continue
        
        # we achieve the missing degrees we can't see        
        Rint = var.L / np.tan(var.alpha)  
        missingAngle = 90 - np.mean(angleList)
        print('Achieve : ' + str(missingAngle) + "°")
        d0 = (var.l/2 + Rint)*(missingAngle*np.pi/180)
        rec.speed = var.maxSpeed/4
        time.sleep( d0/rec.speed)
        rec.speed = 1e-3
        rec.angle = var.alpha
        break

    print('Step 2 : done ') 

def step3(rec) : 
    """ In this step we just go straight ahead"""
    global a   # get distance from the right 
    Rint = var.L / np.tan(var.alpha) 
    Rext = np.sqrt( var.L**2 + (var.l + Rint)**2 ) + var.deltX
    # We compute if we have space 
    minJump = int((Rext + Rint - a)/var.l0) + 1
    if var.jump < minJump : 
        print('Min jump : ' + str(minJump))
        var.jump = minJump   
    d0 = var.jump*var.l0 - (Rint + Rext - a)
    print('Straight ahead : ' + str(d0) + 'm')
    rec.angle = 1e-3
    rec.speed = var.maxSpeed
    time.sleep(d0/rec.speed)
    rec.speed = 1e-3
    print('Step 3 : done')

def step4(rec) : 
    """ In this step we do the second 90 degrees turn """
    def computeAngle() : 
        while not hasattr(rec, 'pcF')  : 
            time.sleep(1)
        # Compute cluster 
        pcF = rec.pcF
        clusterF = clustering(pcF[0], pcF[1])  
        # Do we see something ? 
        if np.size(pcF) == 0 : 
            print("We are blind !")
            return False
        # Extract information of the angle  
        angleList = []
        for cluster in clusterF :
            if np.size(cluster[0]) > 20 : 
                _, _, angle = fittage(cluster[0], cluster[1])
                angleList.append(angle)
        return angleList


    rec.angle = var.alpha
    while True : 
        # We do turn when we can see 
        angleList = computeAngle() 
        if np.size(angleList) == 0 or np.mean(angleList) < 80 : 
            rec.speed = var.maxSpeed/4            
            time.sleep(1e-2)
            continue
        rec.speed = 1e-3
        rec.angle = 1e-3
        time.sleep(1e-2)
        break
   
def step5(rec) : 
    """In this step we stay align straight ahead"""
    def computeDistRight() : 
        while not hasattr(rec, 'pcS')  : 
            time.sleep(1)
        # Compute cluster 
        pcS = rec.pcS
        clusterS = clustering(pcS[0], pcS[1])  
        # Do we see something ? 
        if np.size(pcS) == 0 : 
            print("We are blind !")
            return False
        # Extract information of the angle  
        distList = []
        for cluster in clusterS :
            distList.append(np.mean(cluster[0]))
        print( distList)
        return np.min(distList)

    K = -1 # (var.alpha*180/np.pi)/((var.l0 - var.l)/2)
    rec.speed = var.maxSpeed/2
    while True : 
        x3, _ = rec.vertices[3,:]
        d0 = (7 - x3) + var.l/2
        err = var.l0/2 - d0
        rec.angle = K*err
        time.sleep(5e-1)


#############  THE WHOLE FUNCTION #################
def computeAngle(lstScatter) : 
    while not hasattr(rec, 'pcF')  : 
        time.sleep(1)
    # Compute cluster     
    pcF = rec.pcF
    clusterF = clustering(pcF[0], pcF[1])  
    # Do we see something ? 
    if np.size(pcF) == 0 : 
        print("We are blind !")
        return False
    # Extract information of the angle  
    angleList = []
    for i in range(10) : 
        lstScatter[i].set_offsets(np.c_[[],[]] ) 
    i=0
    for cluster in clusterF :
        if np.size(cluster[0]) > 20 : 
            _, _, angle = fittage(cluster[0], cluster[1])
            angleList.append(angle)
            lstScatter[i].set_offsets(np.c_[cluster[0],cluster[1]])
            i += 1
    return angleList

def control(rec) : 
    """ This function is used to control the vehicle using the attributes described earlier """
    rec.speed = 1e-3
    rec.angle = 1e-3
    # Stopping for leaving the row 
    # step1(rec)
    # # Turning for the first 90 degrees turn 
    # step2(rec)
    # # Go straight ahead 
    # step3(rec)
    # # Second turn 
    # step4(rec)
    # Stay aligned in the row 
    step5(rec)


    
################################################
###########     START SIMULATION     ###########   
################################################

pcInitial = initializePC(var.l0, var.nbRow)
rec = rectangle(pcInitial, var.C0 , L=var.L, l=var.l, sleepingTime=var.sleepingTime, initialAngle=var.initialAngle , l0=var.l0)
threading.Thread( target=control, args=(rec,) ).start()
# Main loop in which you should write your code using the previous described attribute of the vehicle 
while True :     
    # fig2.canvas.draw_idle()
    fig.canvas.draw_idle()
    plt.pause(var.sleepingTime)


        
 