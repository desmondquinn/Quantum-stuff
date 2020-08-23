"""
This program solves the time independent Schrodinger equations for the case of an infinite potential well
The first 4 energy values are calculated numerically and can be verified by the expression 
E(n) = (n^2)*(h^2)/(8*m*L^2) obtained by the analytical solution
The first 4 stationary states are plotted 
"""

import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import scipy.linalg as spla



n=220  #no of nodes(including end points)
L=1e-9     #dimension of well - 1nm
x=np.linspace(0,L,n)        #discrete x values
dx=x[2]-x[1]

#matrix for -D2psi
main_diag = (1/(dx**2))*(2)*np.ones((n, 1)).ravel()
off_diag_upper = (1/(dx**2))*(-1)*np.ones((n, 1)).ravel()
off_diag_lower = (1/(dx**2))*(-1)*np.ones((n,1)).ravel()
a = main_diag.shape[0]
diagonals = [main_diag, off_diag_lower, off_diag_upper]
A = sparse.diags(diagonals, [0,-1,1], shape=(a,a)).toarray()

h=6.626e-34
h=h/(2*3.14)  #h bar
m=9.1e-31
#KE operator 
A=A*((h**2)/(2*m))

#gives the eigenvalues and eigenvectors
#here b contains the eigenvalues and V contains all the eigenvectors. 
#the first column of V will correspond to the first eigenvector, and so on
b,V=spla.eigh(A)

#to sort eigenvalues in increasing order
ei=np.argsort(b)    #returns indices that would sort the array
ei=ei[0:4]          #considering the first 5


#prints first 4 energy eigenvalues
print(b[ei[0:4]])


#psi are given by eigenvector*exp(eigenvalue)

#mode 1
s1=V[:,ei[0]]*np.exp(b[ei[0]])  
#mode 2
s2=V[:,ei[1]]*np.exp(b[ei[1]])
#mode 3
s3=V[:,ei[2]]*np.exp(b[ei[2]])
#mode 4
s4=V[:,ei[3]]*np.exp(b[ei[3]])

#plot that represents the first 4 stationary states 
fig,a =  plt.subplots(2,2)
a[0][0].plot(x,s1)
a[0][0].set_title('1st mode')
a[0][1].plot(x,s2)
a[0][1].set_title('2nd mode')
a[1][0].plot(x,s3)
a[1][0].set_title('3rd mode')
a[1][1].plot(x,s4)
a[1][1].set_title('4th mode')
plt.tight_layout()
plt.show()
