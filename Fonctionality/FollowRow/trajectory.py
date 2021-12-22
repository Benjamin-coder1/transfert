import numpy as np 
import cmath, time, annexes, sys
import matplotlib.pyplot as plt 
sys.path.append("..")
import ConfigParameters as p 
import Tools


"""
This script allow to compute the trajectory of the vehicle depending the position 
    we save the current position in vertices 
        vertices[0,:] - Position of front camera 
        vertices[1,:] - Position of 1 
        vertices[2,:] - Position of 2  
"""

figure , ax = plt.subplots(nrows=1, ncols=1)
figureR, axsR = plt.subplots(nrows=1, ncols=2)
figureL, axsL = plt.subplots(nrows=1, ncols=2)
plt.ion()
plt.show()

def display(l0, dl, beta, L, l, lst) :
    # Display sides    
    figureR.canvas.draw()
    figureL.canvas.draw()
    figureR.canvas.flush_events()
    figureL.canvas.flush_events()
    
    # Display trajectory 
    ax.clear()
    plt.grid(True)        
    ax.scatter( [0]*10, np.linspace(-2,5, 10))
    ax.scatter( [l0]*10, np.linspace(-2,5, 10))
    ax.scatter( [l0/2]*15, np.linspace(-2,10, 15), marker='|', color='grey', alpha = 1/2) 
    ax.set_ylim([-1,4])
    ax.set_xlim([-1,2])
    annexes.add_ComplexeRectangle(dl, beta, ax, L, l) 
    ax.scatter( lst[:,0], lst[:,1], color='k', marker='x', s= 20)
    figure.canvas.draw()    
    figure.canvas.flush_events()

def moove(alpha, deltaT, vertices, v0, vehicle ) : 
    """ Update vertices with : 
            alpha - angle of the wheels (radian)
            deltaT - Time of mooving 
            v0 - constant speed of mooving 
            vehicle - (L,l)
            pos - Save position of front camera in the list
            """    
    # Size of the vehicle    
    L, l = vehicle 
    # Compute matrix of rotation 
    R2 = L/np.tan(alpha)
    w0 = v0/R2
    deltaAngle = w0*deltaT 
    M = np.array([[np.cos(deltaAngle), np.sin(deltaAngle)],[-np.sin(deltaAngle), np.cos(deltaAngle)]])
    # Compute the position of rotation center 
    x0, y0 = vertices[1,:]
    x1, y1 = vertices[2,:]
    eps = (x1 - x0)/abs(x1 - x0)
    if x1 != x0 :             
        r =  (y1 - y0)/(x1 - x0) 
        x = x1 + eps*L / (np.tan(alpha) * np.sqrt( 1 + r**2 ) )
        y = y1 + eps*r*L / (np.tan(alpha) * np.sqrt( 1 + r**2 ) ) 
    else : 
        y = y0  - (l + L/np.tan(alpha) ) 
    C = np.array([[x],[y]])  
    # Change the position 
    for i in range(3) : 
        X = np.dot(M, np.reshape( vertices[i,:], (-1,1) ) - C ) + C
        vertices[i,0], vertices[i,1] = X[0], X[1]
    # Save it in the list 
    return vertices[0,0] , vertices[0,1]
   
def computeTrajectory(v0, dl , beta, alpha, vehicle , nbTemps, K, computTime= False ) : 
    """ This function compute a list with the command to send to the vehicle
            v0 - speed    (m/s)
            dl - distance to the left  (m)  
            alpha - initial angle of the steering wheels (radian)
            beta - angle of the vehicle (radian)
            vehicle - (L, l, l0)   (m,m,m)
            computTime -  Time of computation 
    """  
    
    # Initial position of the front camera
    L, l, l0 = vehicle
    x = dl + (l/2)*np.cos(beta)
    y = L*np.sin(np.pi/2 - beta) + (l/2)*np.sin(beta)    
    # Coordinate of the bounding boxes of the vehicle    
    vertices = [[x,y]]  
    vertices.append( [L*np.cos(np.pi/2 - beta) + dl  ,0])
    vertices.append( [vertices[-1][0] + l*np.cos(beta) ,l*np.sin(beta) ])
    vertices = np.array(vertices) 
    # output 
    output = np.zeros((nbTemps, 6))    
    # Moove during the time of computation     
    if computTime != False  : moove(alpha = alpha, deltaT = computTime, vertices = vertices, v0 = v0, vehicle = (l, L))

    for k in range(nbTemps) : 
        # Compute the alpha we need 
        x, _ = vertices[0,:]
        e = x - (l0/2)
        newAlpha = -K*(e - beta/(1 + abs(e)))
        deltalpha = (min( newAlpha, p.maxSteeringAngle*np.pi/180 ) if newAlpha > 0 else max(newAlpha, -p.maxSteeringAngle*np.pi/180 )) - alpha                       
        timeAlpha = abs(deltalpha/p.wr)     # Take care when it's not completed             
        # Step 1 --> Achieve good steering angle 
        nbStep = 10
        for i in range(nbStep) :
            alpha += deltalpha/nbStep   
            moove(alpha = alpha, deltaT = timeAlpha/nbStep, vertices = vertices, v0 = v0, vehicle = (l, L))
        # Step 2 --> Finish turning 
        position = moove(alpha = alpha, deltaT = p.deltaT - timeAlpha, vertices = vertices, v0 = v0, vehicle = (l, L))
        # Save curent position 
        x1, y1, x2, y2 = vertices[1,0], vertices[1,1], vertices[2,0], vertices[2,1]
        beta = cmath.phase( complex(x2 - x1, y2 - y1))
        # Save values 
        output[k,:] = [        
            k*p.deltaT,    # Time (t)
            alpha,    # Alpha (t)
            beta,   # Beta   (t)
            vertices[0,0] - (l/2)*np.cos(beta),   # dl (t + 1)
            position[0],  # X  (t+1)
            position[1]   # Y  (t+1)
        ]
        # if p.Debug : Tools.disp( "alpha : %f°  Take : %f  beta : %f° dl : %fm" %(180*alpha/np.pi, 100*(timeAlpha)/p.deltaT, 180*output[k,2]/np.pi, output[k,3]), 4)
    
    return output

def followRow(v0, vehicle , cameras, l1, vehicleControler,  nbTemps , K, logger=False) : 
    """ This function compute a list with the command to send to the vehicle
            v0 - speed             
            vehicle - (L, l, l0)
            l1 - distance betwem the two side cameras
            nbTemps - number of point
            vehicleControler - object to control the vehicle 
    """

    ################################################################
    #############   PART 0 - Initialisation  #######################
    ################################################################ 

    # Save information 
    if p.Save : logger.info("START - autonomous mode : following row ")

    # Basic parameters
    L, l, l0 = vehicle    
    sleepingTime = p.deltaT
    lst, lstNew, blind = False, False, 0
    k = 0
    
    while p.Autonomousmode == 1 : 
        ################################################################
        #############   PART 1 - Take measurment #######################
        ################################################################   
        axsR[0].clear()
        axsL[0].clear()
        axsR[1].clear()
        axsL[1].clear()   

        # Counter for computing time 
        start = time.time()  

        # From the side cameras 
        Coordr =  False #detectTrunc.detectTrunc(pipeline=cameras.pipelineR, threshold=(0.2, 0.6) )
        Coordl = ([0.35, 0], [0.4, 0.1]) if k == 0 else False #detectTrunc.detectTrunc(pipeline=cameras.pipelineL, threshold=(0.2, 0.6) )
        # From the front cameras 
        beta = 0
        k += 1
        # Save and display 
        if p.Debug : Tools.disp( 'Measurements made  : {Beta : %f° / Coordr : %s / Coordl : %s } '%(180*beta/np.pi , str(Coordr), str(Coordl)) , 2, logger)                  
                
        ################################################################
        #########   PART 2 - Compute new trajectory ####################
        ################################################################ 

        if (Coordr == False) and (Coordl != False) : 
            ###### ----------------### CASE 1 ###---------------- ###### 
            # Compute parameters 
            dl_lst = []
            for i in range(len(Coordl[0])) : 
                zl, xl = Coordl[0][i], Coordl[1][i]
                dl = abs(zl*np.cos(beta) - xl*np.sin(beta)) - np.cos(beta)*(l - l1)/2 
                dl_lst.append( dl )
            dl = np.mean(dl_lst)
            # Compute command
            Tools.disp("1/ CALCUL COMMANDE : " + "beta : " + str(180*beta/np.pi) +  " / dl : " + str(dl) +  " / sigm dl : " + str(np.std(dl_lst)), 6, logger)
            computTime = (time.time() - start) + p.computingTrajectoryTime
            lstNew = computeTrajectory(v0=v0, dl=dl , beta=beta, alpha=1e-3*np.pi/180, vehicle=vehicle, K=K, nbTemps=nbTemps, computTime=computTime) 
            
        elif (Coordr != False) and (Coordl == False) : 
            ###### ----------------### CASE 2 ###---------------- ###### 
            # Compute parameters 
            dl_lst = []
            for i in range(len(Coordr[0])) : 
                zr, xr = Coordr[0][i], Coordr[1][i]
                dr = abs(zr*np.cos(beta) - xr*np.sin(beta)) -  np.cos(beta)*(l - l1)/2 
                dl_lst.append( l0 - (dr + l*np.cos(beta)) )
            dl = np.mean(dl_lst)
            # Compute command
            Tools.disp("2/ CALCUL COMMANDE : " + "beta : " + str(180*beta/np.pi) + " / dl : " + str(dl) + " / sigm dl : " + str(np.std(dl_lst)), 6, logger)
            computTime = (time.time() - start) + p.computingTrajectoryTime
            lstNew = computeTrajectory(v0=v0, dl=dl , beta=beta, alpha=1e-3*np.pi/180, vehicle=vehicle,  K=K, nbTemps=nbTemps, computTime=computTime) 

        elif (Coordr != False) and (Coordl != False) : 
            ###### ----------------### CASE 3 ###---------------- ###### 
            # Compute parameters 
            dr_lst, dl_lst = [], []

            for i in range(len(Coordr[0])) : 
                zr, xr = Coordr[0][i], Coordr[1][i]                                
                dr = abs(zr*np.cos(beta) - xr*np.sin(beta)) - np.cos(beta)*(l - l1)/2 
                dr_lst.append( dr )            
            
            for i in range(len(Coordl[0])) : 
                zl, xl = Coordl[0][i], Coordl[1][i]                           
                dl = abs(zl*np.cos(beta) - xl*np.sin(beta)) - np.cos(beta)*(l - l1)/2         
                dl_lst.append( dl)

            dr, dl = np.mean(dr_lst), np.mean(dl_lst)            
            dlr = l0 - (dr + l*np.cos(beta))
            Dl = np.abs(dl - dlr)
            dl = (dl + dlr)/2 

            # Compute command
            Tools.disp("3/ CALCUL COMMANDE : " +  "beta : " + str(180*beta/np.pi) + " / dl : " + str(dl) + " / sigm dr : " + str(np.std(dr_lst)) + " / sigm dl : " + str(np.std(dl_lst)) + " / Dl : " + str(Dl), 6, logger)
            computTime = (time.time() - start) + p.computingTrajectoryTime
            lstNew = computeTrajectory(v0=v0, dl=dl , beta=beta, alpha=0*np.pi/180, vehicle=vehicle,  K=K, nbTemps=nbTemps, computTime=computTime) 

        elif (Coordr == False) and (Coordl == False) :
            ###### ----------------### CASE 4 ###---------------- ######
            blind += 1
            if blind == p.nbBlindStop : 
               Tools.disp('End of command !', 0)
               # vehicleControler.go(0)
               return
            alpha, beta, dl = lst[1,1:4]
            lstNew = computeTrajectory(v0=v0 + i/50, dl=dl , beta=beta, alpha=alpha*np.pi/180, vehicle=vehicle,  K=K, nbTemps=nbTemps) 
           
           
        ################################################################
        #######   PART 3 - Choice for the new trajectory ###############
        ################################################################
        
        # Begining of the row  /  no new trajectory computed 
        if np.size(lst) == 1 or np.shape(lst)[0] != nbTemps :
            lst = lstNew
            display(l0, lst[0,3], lst[0,2], L, l, lst[:,-2:] )
            # vehicleControler.turn(180*lst[0,1]/np.pi)
            # vehicleControler.go(0.09)
            print(180*lst[0,1] /np.pi)
            time.sleep(sleepingTime)
            continue
        
        # New trajectory computed we decide what we have to do  
        lst = lstNew
        display(l0, lst[0,3], lst[0,2], L, l, lst[:,-2:] )
        # vehicleControler.turn(180*lst[0,1]/np.pi)
        print(180*lst[0,1] /np.pi)
        # vehicleControler.go(0.09)
        time.sleep(sleepingTime)
        
    # Save information 
    if p.Save : logger.info("END - autonomous mode : following row ")
    
        
        
        
        
 
        
