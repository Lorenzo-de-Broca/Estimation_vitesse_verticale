import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib as mpl

from SRC.extract_data import extract_data
from SRC.filtre_convection import create_convection_filter
from SRC.function import *

frame = extract_data()
filter = create_convection_filter()

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

y_data = frame['W_at_BT']*filter*train_matrix
y_data_filtered = y_data[np.nonzero(y_data)]

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
