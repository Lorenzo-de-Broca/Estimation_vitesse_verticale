import numpy as np
import matplotlib.pyplot as plt
from SRC.extract_data import extract_data
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

y_data = frame['W_at_BT'][:87,:,:]*filter[87,:,:]
y_data_filtered = y_data[np.nonzero(x_data)]

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

# %% fitting
plt.figure()
popt, pcov = curve_fit(linear, x_data_filtered, y_data_filtered)
plt.plot(x_data_filtered, y_data_filtered, 'b.', label='data')
# plt.plot(x_data_filtered, linear(x_data_filtered, *popt), 'r-', label='fit: m=%5.3f, b=%5.3f' % tuple(popt))
plt.xlabel(r'$\Delta$ aos_1830BT / 30s')
plt.ylabel('W_at_BT')
plt.title('Linear regression between Δaos_1830BT and W_at_BT')
plt.legend()
plt.show()  

plt.figure()
plt.imshow(y_data[0,:,:], origin='lower', cmap='viridis')
plt.colorbar()
plt.imshow(filter[t,:,:]*filter[t+1,:,:], origin='lower', cmap=cmapw, norm=norm)
plt.show()
