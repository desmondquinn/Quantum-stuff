"""
This program solves the Schrodinger equation for quantum harmonic oscillator
The potential and the wavefunctions for the first 3 modes are plotted
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import scipy.linalg as spla
import math

n=1000
L=50
x=np.linspace(-L,L,n)
dx=x[1]-x[0]
#print(dx)

#to find KE operator T
main_diag = 2*np.ones((n, 1)).ravel()
off_diag_upper = (-1)*np.ones((n, 1)).ravel()
off_diag_lower = (-1)*np.ones((n,1)).ravel()
t = main_diag.shape[0]
diagonals = [main_diag, off_diag_lower, off_diag_upper]
T = sparse.diags(diagonals, [0,-1,1], shape=(t,t)).toarray()
#units are normalised for convenience 
h=1  #h bar
m=1
T=T*((h**2)/(2*m))
T=T/(dx**2)


#potential for a harmonic oscillator
V1= (1/2)*1*(x**2)  #for harmonic oscillator V=(kx^2)/2 here,k=1
main_diag = V1*np.ones((n,1)).ravel()
off_diag_upper = 0*np.ones((n, 1)).ravel()
off_diag_lower = 0*np.ones((n,1)).ravel() 
v = main_diag.shape[0]
diagonals = [main_diag, off_diag_lower, off_diag_upper]
V = sparse.diags(diagonals, [0,-1,1], shape=(v,v)).toarray()

#Hamiltonian of the system
H=T+V

#returns the energy eigenvalues anf eigenvectors
p,q = spla.eigh(H)


ei = np.argsort(p)
ei = ei[0:3]        #choosing the first 5 modes
Ei = (p[ei]/p[ei][0])

#prints the first 3 energy eigenvalues
print(Ei)

#the wavefunctions are given by their eigenvectors
m1 = q[:,0]
m2 = q[:,ei[1]]
m3 = q[:,ei[2]]

#plotting the potential and the first 3 modes
plt.xticks(np.arange(-5, 5, step=5))
plt.xlim(-5, 5)
plt.ylim(0, 6)
plt.plot(x, m1+p[ei[0]], x, m2+p[ei[1]], x, m3+p[ei[2]], x, V1)
plt.show()

#zoom in to each mode for better view
