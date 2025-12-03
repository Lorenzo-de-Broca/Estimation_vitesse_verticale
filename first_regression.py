import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys
import yaml
from SRC.extract_data import extract_data
from scipy.optimize import curve_fit

frame = extract_data()

#%%
plt.figure(layout='constrained')
plt.imshow(frame['W_at_BT'][0,:,:], origin='lower')
plt.colorbar(label='W_at_BT')
plt.title('t=0')

#%% linear fitting function
def linear(x, m, b): # linear function
    return m*x + b  

def fit_line(x_data, y_data):
    popt, pcov = curve_fit(linear, x_data, y_data)
    return popt  # returns m, b

#%% try with some data
total_len = 500*500
prop_test = 0.6
train_matrix, test_matrix = np.zeros((500,500)), np.ones((500,500))
train_indexes = np.random.choice(range(500*500), int(prop_test*total_len), replace=False)
np.put(train_matrix, train_indexes, 1)
np.put(test_matrix, train_indexes, 0)

#%%

x_data = ((frame['aos_1830BT'][1,:,:]-frame['aos_1830BT'][0,:,:])/30)[np.nonzero(train_matrix)]
y_data = frame['W_at_BT'][0,:,:][np.nonzero(train_matrix)]

m, b = fit_line(x_data, y_data)

plt.figure(layout='constrained')
plt.xscale('log')
plt.plot(x_data, y_data, 'b.', label='data points')
plt.plot(x_data, linear(x_data, m, b), 'r-', label='fitted line')
plt.xlabel(r'$\Delta$ aos_1830BT')
plt.ylabel('W_at_BT')
plt.title('Linear Fit on Training Data')
plt.legend()

# %%
