import trajectory, sys
sys.path.append("../StateMachine")
sys.path.append("..")
import ConfigParameters as p 
import realsense # ,smallVehicle

# Connect to the camera
cameras = realsense.RealSense()

# Connect to the vehicle
vehicleControler = False

# Parameters
v0 = 0.23
L, l, l0, l1 = 0.38, 0.45, 1, 0.215
K = 1
nbTemps = 2

# Launch algorithm of trajectory 
trajectory.followRow(v0=v0,
                        vehicle=(L,l,l0), 
                        l1=l1, cameras=cameras, 
                        vehicleControler=vehicleControler, 
                        K=K, 
                        nbTemps=nbTemps, 
                        logger = p.AutonomousLogger )

