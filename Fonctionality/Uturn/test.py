import numpy as np 
import cmath

L, l, l0 = 3.5, 1.5, 3 
deltY = 1/2
alpha = 45*np.pi/180


## INTERIEUR pour sortir ##
Rint = L / np.tan(alpha)
Rext = np.sqrt( L**2 + (l + Rint)**2 ) 

wy = deltY - np.sqrt(Rint**2 - (Rext - l0)**2)
d0 = Rint + wy
print( d0)
