import matplotlib.pyplot as plt 
import numpy as np
import time, cv2, threading, cmath, keyboard

fig, ax = plt.subplots()
ax.grid(True)
plt.xlim(2.5,22.5)
plt.ylim(-7.5,12.5)

verrou = threading.Lock()
##############################################
###########     DEFINE SHAPES      ###########     
##############################################

class rectangle() :
    def __init__(self, pcGlob, center, L, l, sleepingTime = 1e-2, initialAngle=0) : 
        self.L = L   # Longueur
        self.l = l   # Largeur 
        self.angle = np.pi/6    # Angle roue avant         
        self.speed =  1e-1           # Loop
        self.on = True     # vehicle on 
        self.touched = False   # know if simulation is failed
        self.pcGlob = pcGlob    # Point cloud environement 
        self.sleepingTime = sleepingTime
        self.vertices = cv2.boxPoints(( center , (self.l, self.L), initialAngle)) 
        threading.Thread( target=self.dispXY ).start()
        threading.Thread( target=self.moove ).start()
        threading.Thread( target=self.attachCircle ).start()
        threading.Thread( target=self.detectColision ).start()
        threading.Thread( target=self.dispText ).start()
        threading.Thread( target=self.visionFieldFace ).start()
        threading.Thread( target=self.manualControl ).start()

    def manualControl(self): 
        print(('Manual Control : OK'))
        while True :
            if keyboard.read_key() == "right" : 
                self.angle += 1e-2
            elif keyboard.read_key() == "left" : 
                self.angle -= 1e-2
            elif keyboard.read_key() == "up" : 
                self.speed = min( 5, self.speed + 5e-1)              
            elif keyboard.read_key() == "down" : 
                self.speed = max( 1e-1, self.speed - 5e-1)  
            if self.speed == 1e-1 : 
                self.on = False
                continue
            self.on = True
            time.sleep(self.sleepingTime)
            
    def attachCircle(self) : 
        print(('Display rotation circle : OK'))
        ### DEFINE CIRCLE
        while not hasattr(self, 'C') : 
            time.sleep(self.sleepingTime)
            continue 
        circInt = circle(self.L/np.tan(self.angle), self.C, self.sleepingTime)
        circExt = circle(np.sqrt( self.L**2 + (self.l + self.L/np.tan(self.angle))**2 ), self.C, self.sleepingTime)
        while True : 
            circInt.centre = self.C
            circExt.centre = self.C
            circInt.rayon = self.L/np.tan(self.angle)
            circExt.rayon = np.sqrt( self.L**2 + (self.l + self.L/np.tan(self.angle))**2 )
            time.sleep(self.sleepingTime)

    def moove(self) :
        print(('Mooving : OK'))  
        traceScatter = ax.scatter([],[], color='green', marker='x')       
        pos0, pos1 = [[],[]], [[],[]]
        deb = time.time()        
        while True :           
            d = self.speed * (time.time() - deb)  
            deb = time.time()  
            self.deltaAngle = np.sin(self.angle)*d/self.L     # angle of mooving at constant speed at eah loop 
            M = np.array([[np.cos(self.deltaAngle), np.sin(self.deltaAngle)],[-np.sin(self.deltaAngle), np.cos(self.deltaAngle)]])
            x0, y0 = self.vertices[1,:]
            x1, y1 = self.vertices[2,:]
            eps = (x1 - x0)/abs(x1 - x0)
            if x1 != x0 :             
                r =  (y1 - y0)/(x1 - x0) 
                x = x1 + eps*self.L / (np.tan(self.angle) * np.sqrt( 1 + r**2 ) )
                y = y1 + eps*r*self.L / (np.tan(self.angle) * np.sqrt( 1 + r**2 ) ) 
            else : 
                y = y0  - (self.l + self.L/np.tan(self.angle) )  

            self.C = np.array([[x],[y]])     
            if not self.on : 
                time.sleep(self.sleepingTime)
                continue    
            for i in range(4) : 
                X = np.dot(M, np.reshape( self.vertices[i,:], (-1,1) ) - self.C ) + self.C
                self.vertices[i,0], self.vertices[i,1] = X[0], X[1]

            if self.touched : 
                traceScatter.set_color('#850415')
            pos0[0].append(self.vertices[0,0])
            pos0[1].append(self.vertices[0,1])
            pos1[0].append(x1)
            pos1[1].append(y1)
            traceScatter.set_offsets(np.c_[pos0[0]+pos1[0],pos0[1]+pos1[1] ])            
            time.sleep(self.sleepingTime)
        
    def detectColision(self) :     
        print(('Detection colision : OK'))    
        self.touched = False
        ah, av = 0,0
        while True : 
            x0, y0 = self.vertices[0,:]
            x1, y1 = self.vertices[1,:]
            x2, y2 = self.vertices[2,:]
            x3, y3 = self.vertices[3,:]

            if abs(x0 - x1) > 1e-3 : ah = (y0 - y1) / (x0 - x1)
            else : ah = 1e8
            bh1 = y0 - ah*x0
            bh2 = y3 - ah*x3

            if abs(x0 - x1) > 1e-3  : av = (y0 - y1) / (x0 - x1)
            else : av = 1e8 
            av = (y2 - y1) / (x2 - x1)
            bv1 = y3 - av*x3
            bv2 = y2 - av*x2

            xtouched, ytouched = [], []
        
            if ah > 0 and bh1 > bh2: 
                for ind in range(np.size(self.pcGlob[0])) : 
                    x = self.pcGlob[0][ind]
                    y = self.pcGlob[1][ind]
                    if y > ah*x + bh2 and y < ah*x + bh1 and y < av*x + bv1 and y > av*x + bv2 : 
                        xtouched.append(x)
                        ytouched.append(y)
            elif ah < 0 and bh1 > bh2 : 
                for ind in range(np.size(self.pcGlob[0])) : 
                    x = self.pcGlob[0][ind]
                    y = self.pcGlob[1][ind]
                    if y > ah*x + bh2 and y < ah*x + bh1 and y > av*x + bv1 and y < av*x + bv2 : 
                        xtouched.append(x)
                        ytouched.append(y)
            elif ah > 0 and bh1 < bh2 : 
                for ind in range(np.size(self.pcGlob[0])) : 
                    x = self.pcGlob[0][ind]
                    y = self.pcGlob[1][ind]
                    if y < ah*x + bh2 and y > ah*x + bh1 and y > av*x + bv1 and y < av*x + bv2 : 
                        xtouched.append(x)
                        ytouched.append(y)
            elif ah < 0 and bh1 < bh2 : 
                for ind in range(np.size(self.pcGlob[0])) : 
                    x = self.pcGlob[0][ind]
                    y = self.pcGlob[1][ind]
                    if y < ah*x + bh2 and y > ah*x + bh1 and y < av*x + bv1 and y > av*x + bv2 : 
                        xtouched.append(x)
                        ytouched.append(y)

            if np.size(xtouched) != 0 : 
                self.figPlot.set_color('red')
                self.touched = True
            else :  self.figPlot.set_color('green')

            time.sleep(self.sleepingTime)

    def visionFieldFace(self) :
        print(('Field of vision : OK')) 
        visionScatter = ax.scatter([],[], color='orange', alpha = 1/4)
        visionScatter2 = ax.scatter([],[], color='b', alpha = 1/4)
        self.fieldFace = 86*np.pi/180
        t = np.linspace(0,10, 100)
        fieldplot1 = ax.plot([],[], color='blue')[0]
        fieldplot2 = ax.plot([],[], color='blue')[0]
        fieldplot3 = ax.plot([],[], color='blue')[0]
        fieldplot4 = ax.plot([],[], color='blue')[0]
        alphaF = (np.pi - self.fieldFace) / 2
        M1F = np.array([[np.cos(alphaF), -np.sin(alphaF)],[np.sin(alphaF), np.cos(alphaF)]])
        M2F = np.array([[np.cos(self.fieldFace), -np.sin(self.fieldFace)],[np.sin(self.fieldFace), np.cos(self.fieldFace)]])
        self.fieldSide = 60*np.pi/180
        alphaS = (np.pi - self.fieldFace) / 2
        M1S = np.array([[np.cos(alphaS), np.sin(alphaS)],[-np.sin(alphaS), np.cos(alphaS)]])
        M2S = np.array([[np.cos(self.fieldFace), np.sin(self.fieldFace)],[-np.sin(self.fieldFace), np.cos(self.fieldFace)]])
        
        r, eps = 0,0
        a3, b3 = 0,0
        l1 = 0.1  # for random
        while True : 
            
            ####### FRONT VIEW  ########
            x0, y0 = self.vertices[0,:]
            x1, y1 = self.vertices[1,:]
            x2, y2 = self.vertices[2,:]
            x3, y3 = self.vertices[3,:]
            xc , yc = (x0 + x3)/2, (y0 + y3)/2

            if x3 != x0 : 
                r = (y3 - y0) / (x3 - x0)
                eps = (x3 - x0) / abs(x3 - x0)
            else : 
                r = 1e8
                eps = 1
            
            u = np.array([[1], [r]])*eps/np.sqrt(1 + r**2)

            # Droite num 1
            v = np.dot(M1F,u)
            x = v[0]*t + xc
            y = v[1]*t + yc
            a1, b1 = v[1]/v[0] , yc - v[1]/v[0]*xc            
            fieldplot1.set_xdata(x)
            fieldplot1.set_ydata(y)            

            # Droite num 2 
            v = np.dot(M2F, v )
            x = v[0]*t + xc
            y = v[1]*t + yc
            a2, b2 = v[1]/v[0] , yc - v[1]/v[0]*xc
            fieldplot2.set_xdata(x)
            fieldplot2.set_ydata(y)

            # Droite num 3
            if x3 != x0 : 
                a3 = (y0 - y3)/(x0 - x3) 
            else : 
                a3 = 1e8
            b3 = y3 - x3*a3

            # Detect points we should see FRONT
            xseen, yseen = [], []
            phase = cmath.phase(complex(x0 - x1, y0 - y1))
            if  phase >= np.pi/2 :  
                for ind in range(np.size(self.pcGlob[0])) : 
                    x = self.pcGlob[0][ind]
                    y = self.pcGlob[1][ind]
                    if y > a3*x+b3 and y > a2*x+b2 and x < (y - b1)/a1 : 
                        xseen.append(x + 2*l1*np.random.rand() - l1 )
                        yseen.append(y + 2*l1*np.random.rand() - l1 )
            
            elif phase < np.pi/2 and phase >= 0:  
                for ind in range(np.size(self.pcGlob[0])) : 
                    x = self.pcGlob[0][ind]
                    y = self.pcGlob[1][ind]
                    if y > a3*x+b3 and y > a1*x+b1 and x > (y - b2)/a2 : 
                        xseen.append(x + 2*l1*np.random.rand() - l1 )
                        yseen.append(y + 2*l1*np.random.rand() - l1 )
            
            elif phase >= -np.pi/2 and phase < 0: 
                for ind in range(np.size(self.pcGlob[0])) : 
                    x = self.pcGlob[0][ind]
                    y = self.pcGlob[1][ind]
                    if y < a3*x+b3 and y < a2*x+b2 and x > (y - b1)/a1 : 
                        xseen.append(x + 2*l1*np.random.rand() - l1 )
                        yseen.append(y + 2*l1*np.random.rand() - l1 )
            
            elif phase < -np.pi/2 :  
                for ind in range(np.size(self.pcGlob[0])) : 
                    x = self.pcGlob[0][ind]
                    y = self.pcGlob[1][ind]
                    if y < a3*x+b3 and y < a1*x+b1 and x < (y - b2)/a2  : 
                        xseen.append(x + 2*l1*np.random.rand() - l1 )
                        yseen.append(y + 2*l1*np.random.rand() - l1 )

            
            v1 = np.array( [ [x3 - x0], [y3 - y0]] ) 
            v1 /= np.linalg.norm(v1)
            v2 = np.array( [ [x3 - x2] ,[y3 - y2]] )
            v2 /= np.linalg.norm(v2)

            Mp = np.concatenate((v1,v2),axis=1)            
            res = np.dot(np.linalg.inv(Mp), np.array( [np.array(xseen) - xc, np.array(yseen) - yc] ) )  

            verrou.acquire()            
            self.pcF = [res[0,:], res[1,:]]
            verrou.release()
            visionScatter.set_offsets(np.c_[xseen, yseen])

            ####### SIDE VIEW   ########
            if x3 !=  x2 : 
                r = (y3 - y2) / (x3 - x2)
                eps = (x3 - x2) / abs(x3 - x2)
            else : 
                r = 1e8
                eps = 1
            u = np.array([[1], [r]])*eps/np.sqrt(1 + r**2)            

            # Droite num 1
            v = np.dot(M1S,u)
            x = v[0]*t + x3
            y = v[1]*t + y3
            a1, b1 = v[1]/v[0] , y3 - v[1]/v[0]*x3            
            fieldplot3.set_xdata(x)
            fieldplot3.set_ydata(y)            

            # Droite num 2 
            v = np.dot(M2S, v )
            x = v[0]*t + x3
            y = v[1]*t + y3
            a2, b2 = v[1]/v[0] , y3 - v[1]/v[0]*x3
            fieldplot4.set_xdata(x)
            fieldplot4.set_ydata(y)

            # Droite num 3
            if x3 != x2 : 
                a3 = (y3 - y2)/(x3 - x2) 
            else : 
                a3 = 1e8
            b3 = y3 - x3*a3

            # detect point we should see SIDE 
            xseen, yseen = [], []
            phase = cmath.phase(complex(x2 - x1, y2 - y1))
            if  phase >= np.pi/2 :  
                for ind in range(np.size(self.pcGlob[0])) : 
                    x = self.pcGlob[0][ind]
                    y = self.pcGlob[1][ind]
                    if y > a3*x+b3 and y > a1*x+b1 and x < (y - b2)/a2 : 
                        xseen.append(x + 2*l1*np.random.rand() - l1 )
                        yseen.append(y + 2*l1*np.random.rand() - l1 )
            
            elif phase < np.pi/2 and phase >= 0:  
                for ind in range(np.size(self.pcGlob[0])) : 
                    x = self.pcGlob[0][ind]
                    y = self.pcGlob[1][ind]
                    if y > a3*x+b3 and y > a2*x+b2 and x > (y - b1)/a1 : 
                        xseen.append(x + 2*l1*np.random.rand() - l1 )
                        yseen.append(y + 2*l1*np.random.rand() - l1 )
            
            elif phase >= -np.pi/2 and phase < 0: 
                for ind in range(np.size(self.pcGlob[0])) : 
                    x = self.pcGlob[0][ind]
                    y = self.pcGlob[1][ind]
                    if y < a3*x+b3 and y < a1*x+b1 and x > (y - b2)/a2 : 
                        xseen.append(x + 2*l1*np.random.rand() - l1 )
                        yseen.append(y + 2*l1*np.random.rand() - l1 )
            
            elif phase < -np.pi/2 :  
                for ind in range(np.size(self.pcGlob[0])) : 
                    x = self.pcGlob[0][ind]
                    y = self.pcGlob[1][ind]
                    if y < a3*x+b3 and y < a2*x+b2 and x < (y - b1)/a1  : 
                        xseen.append(x + 2*l1*np.random.rand() - l1 )
                        yseen.append(y + 2*l1*np.random.rand() - l1 )

            v1 = np.array( [ [x2 - x3], [y2 - y3]] ) 
            v1 /= np.linalg.norm(v1)
            v2 = np.array( [ [x3 - x0] ,[y3 - y0]] )
            v2 /= np.linalg.norm(v2)

            Mp = np.concatenate((v1,v2),axis=1)            
            res = np.dot(np.linalg.inv(Mp), np.array( [np.array(xseen) - x3, np.array(yseen) - y3] ) )  

            verrou.acquire()            
            self.pcS = [res[0,:], res[1,:]]
            verrou.release()
            visionScatter2.set_offsets(np.c_[xseen, yseen])

            time.sleep(self.sleepingTime)

    def dispText(self) :
        print(('Text interface : OK'))
        deb = time.time() 
        myText = ax.text(0.98, 0.88,"", transform=ax.transAxes, horizontalalignment='right', verticalalignment='center')
        while True : 
            infos = str(round(self.angle*180/np.pi,1)) + "°"            
            infos += "\n" + str( round(time.time() - deb)) + " s"
            infos += "\n" + str( round(self.speed,1) ) + " m/s"
            x0, y0 = self.vertices[0,:]
            x1, y1 = self.vertices[1,:]
            infos += "\n" + str( round(cmath.phase(complex(x0 - x1, y0 - y1)), 1) ) + " rad"
            if self.touched : infos +=  "\nTouched"
            else : infos += "\nGood"
            myText.set_text(infos)
            time.sleep(self.sleepingTime)

    def dispXY(self) : 
        print(('Display : OK'))
        self.figPlot = ax.plot([],[], color='g')[0]
        while True : 
            self.figPlot.set_xdata(list(self.vertices[:,0]) + [self.vertices[0,0]]) 
            self.figPlot.set_ydata(list(self.vertices[:,1]) + [self.vertices[0,1]]) 
            time.sleep(self.sleepingTime)

class circle() :
    def __init__(self, rayon, centre, sleepingTime ) :
        self.figPlot = ax.plot([],[], color='k', alpha=0.5, linestyle= 'dotted')[0] 
        self.rayon = rayon
        self.centre = centre
        self.sleepingTime = sleepingTime 
        threading.Thread( target=self.dispXY ).start()
    
    def dispXY(self) : 
        x,y = [],[]
        while True :  
            if self.rayon > 1e2 : 
                x = []
                y = []  
            else :    
                t = np.linspace(0,2*np.pi, 100)
                x = self.rayon*np.cos(t) + self.centre[0]
                y = self.rayon*np.sin(t) + self.centre[1]
            self.figPlot.set_xdata(x)
            self.figPlot.set_ydata(y)
            time.sleep(self.sleepingTime)
     
################################################
###########     INITIALISATION       ###########   
################################################


### DEFINE THE POINT CLOUD
def initializePC(l0) : 
    pcScatter = ax.scatter([],[], color='grey')  
    def getPointCloud() : 
        """ This function simulate a point cloud """
        ylim=[-6, 4]
        xini = 4
        x, y = [], []
        for nb in range(6) :
            row = np.array( [[xini + nb*l0,ylim[0]],[xini + nb*l0,ylim[1]]] ) 
            lineY = list(np.linspace(row[1,1], row[0,1], 200))
            y += lineY
            x += np.size(lineY)*[row[0,0]]
        return [x, y]
    pc = getPointCloud()
    pcScatter.set_offsets(np.c_[pc[0], pc[1]])
    return pc
 
  
