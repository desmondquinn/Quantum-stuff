#simulating an electron striking a potential barrier 

import numpy as np
import matplotlib.pyplot as plt


Nz = 500    #no of cells

psi_r = np.zeros((Nz))  #real part of wavefunction
psi_i = np.zeros((Nz))  #imaginary part of wavefunction

me = 9.1e-31    #mass of electron in kg
hbar = 1.055e-34    #reduced Planck's constanr in joules

ddx = 1e-11
ra = 1/8    #the coefficient in the update equation is set to a stable value by trial and error
dt = 0.25*(2*me/hbar)*(ddx**2)
#print(dt)

#defining the potential barrier
V = np.zeros((Nz))
start = 300
end = 500
Vo = 500 #potentail barrier in eV
V[start:end] = Vo*(1.602e-19)   #converting the potential to joules

#initiating a particle with wavelength lambda in a Gaussian envelope
wl = 7e-9   #corresponds to an energy value lower than that of the barrier 
spread = 8
kc = 150 #point at which source is incident
ptot = 0
for k in range(1,start):
    psi_r[k] = (np.exp((-0.5)*((kc-k)/spread)**2))*(np.cos(2*3.1415*(kc-k)/wl))
    psi_i[k] = (np.exp((-0.5)*((kc-k)/spread)**2))*(np.sin(2*3.1415*(kc-k)/wl))
    ptot = ptot + ((psi_r[k])**2) + ((psi_i[k])**2)   #probability
    
#normalise the waveform
norm = np.sqrt(ptot)
for k in range(1,start):
    psi_r[k] = psi_r[k]/norm
    psi_i[k] = psi_i[k]/norm
    

nsteps = 1200    #total number of time steps

fig,a = plt.subplots(2,2)
fig.set_size_inches(18.5, 10.5)
x1 = np.linspace(200,450,250)
#plot line at 300 
x2 = 300*np.ones(500)
y2 = np.linspace(-0.25,0.25,500)


#main fdtd loop
for t in range(0,nsteps):
    #updating the real part
    for k in range(1,Nz-1):
        psi_r[k] = psi_r[k] - ra*(psi_i[k+1]-2*psi_i[k]+psi_i[k-1]) + (dt/hbar)*V[k]*psi_i[k]
    #updating the imaginary part
    for k in range(1,Nz-1):
        psi_i[k] = psi_i[k] + ra*(psi_r[k+1]-2*psi_r[k]+psi_r[k-1]) - (dt/hbar)*V[k]*psi_r[k]
        
    #to plot
    if t == 300:
        a[0][0].plot(x1,psi_r[200:450])
        a[0][0].plot(x2,y2)
        a[0][0].set_title('T = 1')
    if t == 600:
        a[0][1].plot(x1,psi_r[200:450])
        a[0][1].plot(x2,y2)
        a[0][1].set_title('T = 600')
    if t == 900:
        a[1][0].plot(x1,psi_r[200:450])
        a[1][0].plot(x2,y2)
        a[1][0].set_title('T = 900')
    if t == 1000:
        a[1][1].plot(x1,psi_r[200:450])
        a[1][1].plot(x2,y2)
        a[1][1].set_title('T = 1000')


plt.tight_layout()
plt.show()
