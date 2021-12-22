from gpiozero import PWMLED
from adafruit_servokit import ServoKit
import sys
import threading
from pynput.keyboard import Listener 
sys.path.append("..")
import Tools



class smallVehicle :
    def __init__(self, logger=False, pinThrottle=1, pinSteering=2, frequency=100) :
        # Initialize the PWM controler 
        self.pinThrottle, self.pinSteering = pinThrottle, pinSteering        
        self.Kit = ServoKit(channels=16, frequency=frequency)
        # Initialize the logger so save data about the vehicle 
        self.logger = logger 
        # Armed the vehicle 
        self.armed = True
        # Curent parameters of the vehicle 
        self.angle = 0
        self.speed = 0.22
        # Emergency stop 
        self.listener = Listener(on_press=self.emmergencyStop)
        self.stop()
        self.listener.start()
        Tools.disp('Vehicle connected and Armed', 1, logger)
    
    def get_angle(self) : 
        """ Return the value of the current steering angle """
        return self.angle 
    
    def get_speed(self) : 
        """ Return the value of the current speed"""
        return self.speed
        
    def go(self, speed, ignore=False) :
        """ Set the speed of the vehicle 
                speed - value of the speed 
                ignore - set value even if not armed """
        if (not ignore) and (not self.armed) :
            Tools.disp("Can't set speed vehicle not armed", 0, self.logger )
            return
        self.Kit.continuous_servo[self.pinThrottle].throttle = speed
        
    def turn(self, angle, ignore=False) :
        """ Set the speed of the vehicle 
                angle - value of the angle 
                ignore - set value iven if not armed """
        if (not ignore) and (not self.armed) :
            Tools.disp("Can't set steering angle vehicle not armed", 0, self.logger )
            return        
        angMax, valMax, biais  = 22, 0.5, -5        
        self.Kit.continuous_servo[self.pinSteering].throttle = (angle + biais)*(valMax/angMax)
        self.angle = angle*(valMax/angMax)
    
    def stop(self) :
        """ Stop the vehicle """
        self.go(0, ignore=True)
        self.turn(0, ignore=True)
    
    def emmergencyStop(self, key) :
        """ Emergency stop when we press the 'q' button """
        try :
            if key.char == 'q' : 
                self.armed = False
                self.stop()
                print('Disarmed')
        except :
            pass
                
        

