"""
This program solves the time independent Schrodinger equation for the finite potential well
The energy eigenvalues are calculated and the wavefunction is plotted for the bound states
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import scipy.linalg as spla
import math

n=220
L=10e-9   #length of well
x=np.linspace(0,L,n)
dx=x[1]-x[0]
#print(dx)

#constructing the potential well
Vo=0    
V=np.ones((n))*Vo
#V[100:160] = -1.2e-20
bx1 = 100
bx2 = 160
V[bx1:bx2] = -12e-20
V1 = V

#constructing the matrix T for KE
main_diag = 2*np.ones((n, 1)).ravel()
off_diag_upper = (-1)*np.ones((n, 1)).ravel()
off_diag_lower = (-1)*np.ones((n,1)).ravel()
t = main_diag.shape[0]
diagonals = [main_diag, off_diag_lower, off_diag_upper]
T = sparse.diags(diagonals, [0,-1,1], shape=(t,t)).toarray()
T=T/(dx**2)
h=6.626e-34
h=h/(2*3.14)  #h bar
m=9.1e-31
T=T*((h**2)/(2*m))

#constructing the matrix V for PE
main_diag = V*np.ones((n,1)).ravel()
off_diag_upper = 0*np.ones((n, 1)).ravel()
off_diag_lower = 0*np.ones((n,1)).ravel() 
v = main_diag.shape[0]
diagonals = [main_diag, off_diag_lower, off_diag_upper]
V = sparse.diags(diagonals, [0,-1,1], shape=(v,v)).toarray()

#Hamiltonian of the system
H=T+V

#finding the eigenvalues and eigenvectors
p,q = spla.eigh(H)


ei = np.argsort(p)  #sorting the eigenvalues
#print(p[ei[0:8]])
ei = ei[0:4]        #choosing the first 4 modes, returns index values

    
#prints the first 4 energy eigenvalues (in Joules)
print(p[ei])

#the wavefunctions are given by their eigenvectors
m1 = q[:,ei[0]]     #mode 1
m2 = q[:,ei[1]]     #mode 2
m3 = q[:,ei[2]]     #mode 3
m4 = q[:,ei[3]]     #mode 4


#points at which potential barrier is present
a = (L/n)*bx1
b = (L/n)*bx2
a1 = a*np.ones((100)).ravel()
b1 = b*np.ones((100)).ravel()
y = np.linspace(-0.2,0.2,100)


#plotting the wavefunctions
fig,a =  plt.subplots(2,2)

a[0][0].plot(x,m1)
a[0][0].plot(a1,y, color='C2')
a[0][0].plot(b1,y, color='C2')
a[0][0].set_title('1st mode')

a[0][1].plot(x,m2)
a[0][1].plot(a1,y, color='C2')
a[0][1].plot(b1,y, color='C2')
a[0][1].set_title('2nd mode')


a[1][0].plot(x,m3)
a[1][0].plot(a1,y, color='C2')
a[1][0].plot(b1,y, color='C2')
a[1][0].set_title('3rd mode')

a[1][1].plot(x,m4)
a[1][1].plot(a1,y, color='C2')
a[1][1].plot(b1,y, color='C2')
a[1][1].set_title('4th mode')

plt.tight_layout()
plt.show()


"""
here the first 4 modes are plotted
But if the 5th or higher mode is plotted, the confinement is not seen, as these are not bound states.
(for these particular parameters)
Only the first 4 modes are bound states, as is evident from their negative energy values
Also, the exponential decay at the boundaries can be seen and the penetration depth 1/alpha can be verified
by the expression h/((2*pi)*sqrt(2m*(V-E)))
"""
