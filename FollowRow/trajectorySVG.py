import numpy as np 
import cmath, time, annexes
import matplotlib.pyplot as plt 
"""
This script allow to compute the trajectory of the vehicle depending the position 
    we save the current position in vertices 
        vertices[0,:] - Position of front camera 
        vertices[1,:] - Position of 1 
        vertices[2,:] - Position of 2  
"""

plt.ion()
figure , ax = plt.subplots(nrows=1, ncols=1)
def display(l0, dl, beta, L, l, lst) : 
    ax.clear()
    plt.grid(True)        
    ax.scatter( [0]*10, np.linspace(-2,5, 10))
    ax.scatter( [l0]*10, np.linspace(-2,5, 10))
    ax.scatter( [l0/2]*15, np.linspace(-2,10, 15), marker='|', color='grey', alpha = 1/2) 
    plt.ylim([-1,7])
    plt.xlim([-1,2])
    # Display trajectory 
    annexes.add_ComplexeRectangle(dl, beta, ax, L, l) 
    ax.scatter( lst[:,0], lst[:,1], color='k', marker='x', s= 20) 
    figure.canvas.draw()    
    figure.canvas.flush_events()

def moove(alpha, deltaT, vertices, v0, vehicle, pos = False) : 
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
    if pos != False : 
        pos[0].append(vertices[0,0])
        pos[1].append(vertices[0,1])
        return pos

def computeTrajectory(v0, dl , beta, alpha, vehicle , nbTemps = 15, K = 1) : 
    """ This function compute a list with the command to send to the vehicle
            v0 - speed    (m/s)
            dl - distance to the left  (m)  
            alpha - initial angle of the steering wheels (radian)
            beta - angle of the vehicle (radian)
            vehicle - (L, l, l0)   (m,m,m)
    """  
    
    # Initial position of the front camera
    L, l, l0 = vehicle
    x = dl + (l/2)*np.cos(beta)
    y = L*np.sin(np.pi/2 - beta) + (l/2)*np.sin(beta)
    # Data dictionnary 
    position_lst, alpha_lst, beta_lst, dl_lst = [[x], [y]], [[0, alpha]], [[beta]], [[dl]]
    # Coordinate of the bounding boxes of the vehicle    
    vertices = [[x,y]]  
    vertices.append( [L*np.cos(np.pi/2 - beta) + dl  ,0])
    vertices.append( [vertices[-1][0] + l*np.cos(beta) ,l*np.sin(beta) ])
    vertices = np.array(vertices) 
    # Parameters of trajectory
    deltaT = 0.3    # Interval for sending command 
    wr = (1/1.1e-2)*np.pi/180   # Speed of rotation of the wheels   rad/s  

    for k in range(nbTemps) : 
        # Compute the alpha we need 
        x, y = vertices[0,:]
        e = x - (l0/2)
        deltalpha = (min( -K*e, np.pi/4 ) if e < 0 else max(-K*e, -np.pi/4 )) - alpha                       
        timeAlpha = abs(deltalpha/wr)     # Take care when it's not completed             
        # Step 1 --> Achieve good steering angle 
        nbStep = 10
        for i in range(nbStep) :
            alpha += deltalpha/nbStep   
            moove(alpha = alpha, deltaT = timeAlpha/nbStep, vertices = vertices, v0 = v0, vehicle = (l, L))
        # Step 2 --> Finish turning 
        position_lst = moove(alpha = alpha, deltaT = deltaT - timeAlpha, vertices = vertices, v0 = v0, vehicle = (l, L), pos = position_lst)
        # Save curent position 
        x1, y1, x2, y2 = vertices[1,0], vertices[1,1], vertices[2,0], vertices[2,1]
        beta_lst.append( [cmath.phase( complex(x2 - x1, y2 - y1)) ])
        alpha_lst.append([(k+1)*deltaT,alpha])
        dl_lst.append( list(vertices[0,0] - (l/2)*np.cos(beta_lst[-1])))
        print( "alpha : %f°  Take : %f  beta : %f° dl : %fm" %(180*alpha/np.pi, 100*(timeAlpha)/deltaT, 180*beta_lst[-1][0]/np.pi, dl_lst[-1][0]))
    
    return np.concatenate( (alpha_lst, beta_lst, dl_lst, np.transpose(position_lst)), axis = 1)


def followRow(v0, vehicle , nbTemps = 10, K = 1) : 
    """ This function compute a list with the command to send to the vehicle
            v0 - speed 
            Coordr - (xr, zr) 
            Coordl - (xl, zl) 
            beta - angle of the vehicle (degrees)
            vehicle - (L, l, l0)
            nbTemps - number of point 
    """

    ################################################################
    #############   PART 0 - Initialisation  #######################
    ################################################################ 
    
    # Basic parameters
    L, l, l0 = vehicle    
    sleepingTime = 2
    # Initial state 
    lst = False
    i = 0   
    
    while True : 
        ################################################################
        #############   PART 1 - Take measurment #######################
        ################################################################ 
        # From the historic 
        # state = lstNew[0,1:]
        # From the side cameras
        Coordl = (1, 0.4) if i%5 == 0 else False
        Coordr = False
        # From the front camera 
        beta = 0.1 if i%5 == 0 else False 
        i += 1

        print('\nMESURES PHYSIQUES : {Beta : %f° / Coordr : %s / Coordl : %s } '%(180*beta/np.pi , str(Coordr), str(Coordl)))
                
        ################################################################
        #########   PART 2 - Compute new trajectory ####################
        ################################################################      
        lstNew = False 

        if (Coordr == False) and (Coordl != False) and (beta != False) : 
            ###### ----------------### CASE 3 ###---------------- ###### 
            # Compute parameters 
            xl, zl = Coordl
            dl = abs(zl*np.cos(beta) - xl*np.sin(beta))            
            # Compute command
            print("CALCUL COMMANDE :  { beta : %f, dl : %f}"%(180*beta/np.pi, dl) )
            lstNew = computeTrajectory(v0=v0, dl=dl , beta=beta, alpha=-10*np.pi/180, vehicle=vehicle ) 
            
        elif (Coordr != False) and (Coordl == False) and (beta != False) : 
            ###### ----------------### CASE 5 ###---------------- ###### 
            # Compute parameters 
            xr, zr = Coordr
            dr = abs(zr*np.cos(beta) - xr*np.sin(beta))
            dl = l0 - (dr + l*np.cos(beta))
            # Compute command
            print("CALCUL COMMANDE :  { beta : %f, dl : %f}"%(180*beta/np.pi, dl) )
            lstNew = computeTrajectory(v0=v0, dl=dl , beta=beta, alpha=0, vehicle=vehicle)

        elif (Coordr != False) and (Coordl != False) and (beta != False) : 
            ###### ----------------### CASE 2 ###---------------- ###### 
            # Compute parameters 
            xr, zr = Coordr
            xl, zl = Coordl
            dr = abs(zr*np.cos(beta) + xr*np.sin(beta))
            dl = abs(zl*np.cos(beta) - xl*np.sin(beta))
            dlr = l0 - (dr + l*np.cos(beta))
            dl = (dl + dlr)/2   # improve efficiency using both sides
            # Compute command
            print("CALCUL COMMANDE :  { beta : %f, dl : %f, Ddl : %f}"%(180*beta/np.pi, dl, abs(dl - dlr)) )
            lstNew = computeTrajectory(v0=v0, dl=dl , beta=beta, alpha=0, vehicle=vehicle)

        elif (Coordr != False) and (Coordl != False) and (beta == False) : 
            ###### ----------------### CASE 2 ###---------------- ###### 
            # Compute parameters 
            xr, zr = Coordr
            xl, zl = Coordl
            r = np.sqrt( (zr + zl + l)**2 + (xr - xl)**2)
            phi = cmath.phase( complex(zr + zl + l, xr - xl) ) 
            beta = phi + np.arccos(l0/r)            
            dr = abs(zr*np.cos(beta) + xr*np.sin(beta))
            dl = abs(zl*np.cos(beta) - xl*np.sin(beta))            
            # Compute command
            print("CALCUL COMMANDE :  { beta : %f, dl : %f}"%(180*beta/np.pi, dl) )
            lstNew = computeTrajectory(v0=v0, dl=dl , beta=beta, alpha=0, vehicle=vehicle)

       
        ################################################################
        #######   PART 3 - Choice for the new trajectory ###############
        ################################################################

        # Begining of the row
        if np.size(lst) == 1 : 
            lst = lstNew
            display(l0, lst[0,3], lst[0,2], L, l, lst[:,-2:] )
            time.sleep(sleepingTime)
            continue
        
        # No new trajectory computed 
        if np.size(lstNew) == 1  : 
            # End of the list of the computed trajectory 
            if np.shape(lst)[0] == 1 : 
                print('End of command !')
                return 
            lst[:,-1] -= (lst[1, -1] - lst[0, -1] )           
            lst = lst[1:,:]
            display(l0, lst[0,3], lst[0,2], L, l, lst[:,-2:] )
            time.sleep(sleepingTime)
            continue
        
        # New trajectory computed we decide what we have to do  
        n = np.shape(lst)[0]
        comp = annexes.compareTrajectory(lstNew[:n, 3], lst[:, 3])
        print( comp  )
        if np.max(comp) < 1000 : 
            lst = lstNew
        
        # Send the commands to the vehicle 
        time.sleep(sleepingTime)
        
 
        
