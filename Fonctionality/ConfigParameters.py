import logging,os, sys
import numpy as np
import Tools  
PATH = os.getcwd().split('/rp')[0] + '/rp' 

""" This script allow to control all the parameters of the vehicle """

state = 1   # this variable represent the curent state of the machine : 
				# state = 0 --> machine off
				# state = 1 --> Initialisation 
				# state = 2 --> manual control 
				# state = 3 --> autonomous mode 

Autonomousmode = 1 # this variable represent the curent action of the autonomous mode : 
						# Autonomousmode = 0 --> Stop autonomous mode
						# Autonomousmode = 1 --> Straight ahead in the row 
						# Autonomousmode = 2 --> edge
						# Autonomousmode = 3 --> U turn 

# sleep time between two loops 
sleepTime = 1  

# informations about the log file 
logFileEvent = {}
logFileEvent['name'] = PATH +  "/LogFile/Events.csv"   
logFileEvent['format'] = '%(asctime)s;%(levelname)s;%(message)s'
logFileEvent['dateFormat'] = '%H:%M:%S'
logFileEvent['level'] = logging.INFO
EventLogger = Tools.CreateLogger(logFileEvent)

logFileMeasure = {}
logFileMeasure['name'] = PATH + "/LogFile/Values.csv"   
logFileMeasure['format'] = '%(asctime)s:%(msecs)03d;%(message)s'
logFileMeasure['dateFormat'] = '%H:%M:%S'
logFileMeasure['level'] = logging.INFO
MeasureLogger = Tools.CreateLogger(logFileMeasure)

logFileAutonomous = {}
logFileAutonomous['name'] = PATH + "/LogFile/autonomous.csv"   
logFileAutonomous['format'] = '%(asctime)s:%(msecs)03d;%(message)s'
logFileAutonomous['dateFormat'] = '%H:%M:%S'
logFileAutonomous['level'] = logging.INFO
AutonomousLogger = Tools.CreateLogger(logFileAutonomous)

# frequency of saving sensor data (Hz)
sensorFrequency = 10

# Connexion string by default 
connection_string = '/dev/ttyUSB0'   # '/dev/ttyUSB0'

# Obstacle detection parameters
Nbcsvtmax = 5   # number of consecutive frames for stopping
conf = 0.2   # percentage for object detection  
stop = False   # Stop the vehicle or not 
distMin = 0.3		# Min distance before an obstacle (m)
configFile = PATH + "/Fonctionality/ObstacleDetection/CaffeeModel/SSD_MobileNet.prototxt"    # config of model
weightFile = PATH + "/Fonctionality/ObstacleDetection/CaffeeModel/SSD_MobileNet.caffemodel"  # weight of model 

# Saving camera
fileNameColor = PATH + "/Svg/color.p"
fileNamePointCloud = PATH + "/Svg/pointCloud.p"
fileNameResult = PATH + "/Svg/result.p"

# Straight in the row 
deltaT = 0.5   # Interval for sending command 
wr = (1/1.1e-2)*np.pi/180   # Speed of rotation of the wheels   rad/s  
computingTrajectoryTime = 0.05  # (s) time of computation of the trajectory 
maxSteeringAngle = 20   # (degrees) maximum angle 
nbBlindStop = 5     # number of blind iteration before stopping 

# mode 
Debug = True
Save = True 


