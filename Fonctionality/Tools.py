import sys, time, logging
import pyrealsense2 as rs 


def CreateLogger(logFileInfo):
	"""This function allows to create as many log file as we want 
		logFileInfo - Information about the log file    format / dateFormat/name/level"""
	formatter = logging.Formatter(logFileInfo['format'], datefmt=logFileInfo['dateFormat'])
	handler = logging.FileHandler(logFileInfo['name'])        
	handler.setFormatter(formatter)
	logger = logging.getLogger(logFileInfo['name'].split('.')[0])
	logger.setLevel(logFileInfo['level'])
	logger.addHandler(handler)
	return logger

def ConnectCamera(logger=False) : 
	""" This function create a connexion with the camera set logger to write what append in the 
	log file """
	pipe, align, cfg = [rs.pipeline() , rs.align(rs.stream.color), rs.config()]
	while True : 
		try:
			print('Connexion realsense Camera ... ')
			if logger != False : 
				logger.info("Connexion realsense Camera")
			cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
			cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
			pipe.start(cfg)
			print('\033[92m' + 'Camera configurated and connected.' + '\033[0m')  
			if logger != False : 
				logger.info("Camera configurated and connected")   
			return pipe, align 
		except Exception as e:
			print('\033[91m' + str(e.args) + '\033[0m') 
			if logger != False : 
				logger.info(str(e.args) )       
			for tm in range(30) : 
				sys.stdout.write("\r{0}".format(" "*20))
				sys.stdout.write("\r{0}".format("Reconnection attempt in " + str(30 - tm) + " s ...", 10))
				sys.stdout.flush()
				time.sleep(1)

def disp( msg, color=-1,  logger=False) : 
	""" This function print colored msg and can send it to a logfile """ 
	if logger != False : 
		logger.info(msg)
	
	if color == 0 : 
		# RED 
		colorMsg = '\033[31m'
	elif color == 1 : 
		# GREEN 
		colorMsg = '\033[92m'
	elif color == 2 : 
		# YELLOW 
		colorMsg = '\033[93m'
	elif color == 3 : 
		# BLUE 
		colorMsg =  '\033[94m' 
	elif color == 4 : 
		# WHITE 
		print(msg)
		return 
	elif color == 5 : 
		# PURPLE
		colorMsg =  '\033[35m' 
	elif color == 6 : 
		# BLUE 
		colorMsg =  '\033[34m' 

	else : 
		# No print 
		return 
	print( colorMsg + msg + '\033[0;0m')




