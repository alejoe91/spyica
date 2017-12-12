'''
   Convention:
   I - nA
   distance: um
   V: uV
   t = us
'''

import numpy as np
import matplotlib.pylab as plt



#Travelling gaussian on straight line x0
x0 = np.linspace(0, 1000, 100)
time = np.arange(0, 1001, 30)

dist = 50
x_vec = np.linspace(-2000, 2000, 100)
y_vec = np.linspace(-2000, 2000, 100)
z_vec = np.linspace(-2000, 2000, 100)

x, y, z = np.meshgrid(x_vec, y_vec, z_vec)

v_grid = np.zeros((len(y_vec), len(z_vec)))
c = 1 # m/s - um/ms

# gauss
amp = 25.
sig = 20.


fig1 = plt.figure()
ax1 = fig1.add_subplot(211)
ax2 = fig1.add_subplot(212)

v_gauss_im = []
v_dgauss_im = [9]

for t_inst in time:
    gauss = amp/np.sqrt(2*np.pi*sig**2) * np.exp(-(x0-c*t_inst)**2/(2*sig**2))
    dgauss_m = amp*(x0-c*t_inst) / (np.sqrt(2*np.pi)*sig**3) * np.exp(-(x0-c*t_inst)**2/(2*sig**2))
    ax1.plot(x0, gauss)
    ax2.plot(x0, dgauss_m)

plt.ion()
plt.show()


