import sys, time, threading  
sys.path.append("../ObstacleDetection")
sys.path.append("..")
import ConfigParameters as p
import ObjectDetection, Tools, smallVehicle


###################################################################################
#####							START MACHINE 								#######
###################################################################################

p.EventLogger.info('Launch of the Program - %s', time.strftime('%d/%m/%Y'))


while p.state != 0 : 	

	####################################################################################

	# Display mode 
	if p.state == 1 : 
		Tools.disp('\n## ---- Initialisation of the vehicle ---- ##', 3, p.EventLogger)

	while p.state == 1 :
		##############################
		# -- Initialisation state -- #
		##############################		

		# Create an instance of vehicle 
		vehicle = smallVehicle()
	
		# Changing state 
		p.state = 2


	####################################################################################

	# Display mode 
	if p.state == 2 : 
		Tools.disp('## ---- Control mode ---- ##' , 3, p.EventLogger)
		

	while p.state == 2 : 
		###############################
		# -- Groud station control -- #
		###############################

		# Time of sleeping 
		time.sleep(p.sleepTime)

	####################################################################################

	# Display mode 
	if p.state == 3 : 
		Tools.disp('## ---- Autonomous mode ---- ##', 3, p.EventLogger)
	

	while p.state == 3 :
		###############################
		# --   Autonomous control  -- #
		###############################
		
		# Time of sleeping	
		time.sleep(p.sleepTime)

	

	####################################################################################

# End of the programe 
Tools.disp('## ---- Vehicle OFF ---- ##', 2, p.EventLogger)
