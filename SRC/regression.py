import numpy as np
import matplotlib.pyplot as plt
<<<<<<< HEAD
from SRC.extract_data import extract_data, create_reg_arrays1
from SRC.filtre_convection import create_convection_filter
from scipy.optimize import curve_fit
import matplotlib as mpl

frame = extract_data()
filter = create_convection_filter()

# %% defining functions for manovariable regression
def linear(x, m, b): # linear function
    return m*x + b

def exponential(x, a, b, c): # exponential function
    return a * np.exp(b * x) + c

def logarithmic(x, a, b): # logarithmic function
    return a * np.log(x) + b    
=======
from scipy.optimize import curve_fit
import matplotlib as mpl

from SRC.extract_data import extract_data, create_reg_arrays1
from SRC.filtre_convection import create_convection_filter
from SRC.utils import *

print("start")
frame = extract_data()
filter = create_convection_filter()   
>>>>>>> main

# %% train & test matrix
total_len = 500*500
prop_train = 0.6
train_matrix, test_matrix = np.zeros((500,500)), np.ones((500,500))
train_indexes = np.random.choice(range(500*500), int(prop_train*total_len), replace=False)
np.put(train_matrix, train_indexes, 1)
np.put(test_matrix, train_indexes, 0)

# %% aos_1830BT linear regression
# masked_trained = frame['aos_1830BT']*filter*train_matrix
x_data = np.zeros((87, 500, 500))
for t in range(87):
    x_data[t,:,:] = (frame['aos_1830BT'][t+1,:,:]-frame['aos_1830BT'][t,:,:])*filter[t,:,:]*filter[t+1,:,:]*train_matrix/30

x_data_filtered = x_data[np.nonzero(x_data)]

y_data = frame['W_at_BT'][:87,:,:]
y_data_filtered = y_data[np.nonzero(x_data)]
<<<<<<< HEAD

=======
print("finish creation matrix")
>>>>>>> main
# %% 
t=0
plt.figure()
norm1 = mpl.colors.Normalize(vmin=-0.3, vmax=0.1)
cmapb = mpl.colors.ListedColormap(['None','black','None'])
cmapw = mpl.colors.ListedColormap(['None','white','None'])
bounds=[-1,-0.1,0.1,1]
norm = mpl.colors.BoundaryNorm(bounds, cmapb.N)
plt.imshow(x_data[t,:,:], origin='lower', cmap='viridis', norm=norm1)
plt.colorbar()
plt.imshow(train_matrix, origin='lower', cmap=cmapb, norm=norm)
plt.imshow(filter[t,:,:]*filter[t+1,:,:], origin='lower', cmap=cmapw, norm=norm)
plt.title(f'Training data points overlayed on Δaos_1830BT at t={t}')
<<<<<<< HEAD

# %% Distributions 183 BT
=======
print("begin fit")
# %% fitting
>>>>>>> main
x_data_filtered1, y_data_filtered1 = create_reg_arrays1('1830', frame, filter)
x_data_filtered7, y_data_filtered7 = create_reg_arrays1('1837', frame, filter)
x_data_filtered10, y_data_filtered10 = create_reg_arrays1('183T', frame, filter)
plt.figure()
<<<<<<< HEAD
# popt, pcov = curve_fit(linear, x_data_filtered, y_data_filtered)
=======
popt, pcov = curve_fit(linear, x_data_filtered, y_data_filtered)
>>>>>>> main
plt.plot(x_data_filtered10, y_data_filtered10, 'g.', label=r'$183\pm 10$', alpha=0.4)
plt.plot(x_data_filtered7, y_data_filtered7, 'r.', label=r'$183\pm 7$', alpha=0.4)
plt.plot(x_data_filtered1, y_data_filtered1, 'b.', label=r'$183\pm 1$', alpha=0.4)
# plt.plot(x_data_filtered, linear(x_data_filtered, *popt), 'r-', label='fit: m=%5.3f, b=%5.3f' % tuple(popt))
plt.xlabel(r'$\Delta$ aos_1830BT / 30s')
plt.ylabel('W_at_BT')
<<<<<<< HEAD
plt.title(r'$\Delta$aos_183 $\pm 3,5,10$ BT VS W_at_BT')
# plt.ylim(np.min(y_data_filtered),np.max(y_data_filtered))
plt.legend()
# plt.savefig('Distribution_183BT_VS_WatBT.pdf', dpi=300)
plt.show()  

# %% Distributions 325 BT
x_data_filtered1, y_data_filtered1 = create_reg_arrays1('3250', frame, filter)
x_data_filtered7, y_data_filtered7 = create_reg_arrays1('3257', frame, filter)
x_data_filtered10, y_data_filtered10 = create_reg_arrays1('325T', frame, filter)
plt.figure()
popt, pcov = curve_fit(linear, x_data_filtered, y_data_filtered)
plt.plot(x_data_filtered10, y_data_filtered10, 'g.', label=r'$235\pm 10$', alpha=0.4)
plt.plot(x_data_filtered7, y_data_filtered7, 'r.', label=r'$235\pm 7$', alpha=0.4)
plt.plot(x_data_filtered1, y_data_filtered1, 'b.', label=r'$235\pm 1$', alpha=0.4)
# plt.plot(x_data_filtered, linear(x_data_filtered, *popt), 'r-', label='fit: m=%5.3f, b=%5.3f' % tuple(popt))
plt.xlabel(r'$\Delta$ aos_2350BT / 30s')
plt.ylabel('W_at_BT')
plt.title(r'$\Delta$aos_235 $\pm 3,5,10$ BT VS W_at_BT')
plt.ylim(np.min(y_data_filtered),np.max(y_data_filtered))
plt.legend()
plt.savefig('Distribution_235BT VS WatBT.pdf', dpi=300)
=======
plt.title('Linear regression between Δaos_1830BT and W_at_BT')
plt.ylim(np.min(y_data_filtered),np.max(y_data_filtered))
plt.legend()
>>>>>>> main
plt.show()  

#%% verif filtre convection VS W_at mask
# plt.figure()
# plt.imshow(y_data[0,:,:], origin='lower', cmap='viridis')
# plt.colorbar()
# plt.imshow(filter[t,:,:]*filter[t+1,:,:], origin='lower', cmap=cmapw, norm=norm)
# plt.show()

#%% W_at - DELTA TB at t
t=0
plt.figure()
<<<<<<< HEAD
plt.imshow((y_data[t,:,:]-x_data[t,:,:])*filter[t,:,:], origin='lower', cmap='viridis')
=======
plt.imshow(y_data[t,:,:]-x_data[t,:,:], origin='lower', cmap='viridis')
>>>>>>> main
plt.colorbar()
plt.title(f'W_at_BT - Δaos_1830BT/30s at t={t}')
plt.show()

<<<<<<< HEAD
=======
def PCA (): 
    """Principal Component Analysis function to reduce the dimension of the problem.
    """
    pass
>>>>>>> main
