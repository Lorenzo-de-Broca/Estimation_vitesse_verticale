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
fig, ax = plt.subplots(1,2, layout='constrained')
a0 = ax[0].imshow(frame['W_at_BT'][50,:,:], origin='lower')
ax[0].set_title('W_at_BT')
a1 = ax[1].imshow((frame['aos_1830BT'][51,:,:]-frame['aos_1830BT'][50,:,:])/30, origin='lower')
ax[1].set_title(r'$\Delta$ aos_1830BT')
fig.suptitle('t=50')
fig.colorbar(a0, ax=ax[0], shrink=0.5)
fig.colorbar(a1, ax=ax[1], shrink=0.5)


#%% linear fitting function
def linear(x, m, b): # linear function
    return m*x + b  

def fit_line(x_data, y_data):
    popt, pcov = curve_fit(linear, x_data, y_data)
    return popt  # returns m, b

#%% exponential fitting function
def exponential(x, a, b, c): # exponential function
    return a * np.exp(b * x) + c

#%% fitting function
def fit_func(func, x_data, y_data):
    popt, pcov = curve_fit(func, x_data, y_data)
    return popt  # returns fitted parameters

#%% try with some data
total_len = 500*500
prop_test = 0.6
train_matrix, test_matrix = np.zeros((500,500)), np.ones((500,500))
train_indexes = np.random.choice(range(500*500), int(prop_test*total_len), replace=False)
np.put(train_matrix, train_indexes, 1)
np.put(test_matrix, train_indexes, 0)

#%% firt linear fit
x_data = ((frame['aos_1830BT'][1,:,:]-frame['aos_1830BT'][0,:,:])/30)[np.nonzero(train_matrix)]
y_data = frame['W_at_BT'][0,:,:][np.nonzero(train_matrix)]

m, b = fit_line(x_data, y_data)

plt.figure(layout='constrained')
# plt.xscale('log')
plt.yscale('log')
plt.plot(x_data, y_data, 'b.', label='data points')
plt.plot(x_data, linear(x_data, m, b), 'r-', label='fitted line')
plt.xlabel(r'$\Delta$ aos_1830BT')
plt.ylabel('W_at_BT')
plt.title('Linear Fit on Training Data')
plt.legend()

# %% first exponential fit
x_data = ((frame['aos_1830BT'][1,:,:]-frame['aos_1830BT'][0,:,:])/30)[np.nonzero(train_matrix)]
y_data = frame['W_at_BT'][0,:,:][np.nonzero(train_matrix)]      

a, b, c = fit_func(exponential, x_data, y_data)

plt.figure(layout='constrained')
# plt.xscale('log')
# plt.yscale('log')
plt.plot(x_data, y_data, 'darkblue', linestyle='None', marker='.', label='data points')
plt.plot(x_data, exponential(x_data, a, b, c), 'orangered', label='fitted exponential')
plt.xlabel(r'$\Delta$ aos_1830BT')
plt.ylabel('W_at_BT')
plt.title('Exponential Fit on Training Data')
plt.legend()

