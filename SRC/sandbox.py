import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from SRC.extract_data import extract_data, create_reg_array1
from SRC.filtre_convection import create_convection_filter
from SRC.utils import *

def multi_lin_reg (x_data,y_data): 
    """performs a multiple linear regression between x_data and y_data

    Args:
        x_data (np.array): 2D array (n_samples, n_features) of input data
        y_data (np.array): 1D array (n_samples,) of output data

    Returns:
        model (LinearRegression): trained linear regression model
    """
    
    model = LinearRegression()
    model.fit(x_data, y_data)

    return model

def test_model(model, X_test, y_test):
    """Tests the linear regression model on test data and computes the mean squared error

    Args:
        model (LinearRegression): trained linear regression model
        X_test (np.array): 2D array (n_samples, n_features) of test input data
        y_test (np.array): 1D array (n_samples,) of test output data

    Returns:
        rmse (float): root mean squared error of the model on the test data
    """
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    residuals = y_test - y_pred

    r2 = model.score(X_test, y_test)
    print("R² =", r2)

    
    return rmse, residuals, r2
    
    
print("start")
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

y_data = frame['W_at_BT'][:87,:,:]
y_data_filtered = y_data[np.nonzero(x_data)]
print("finish creation matrix")
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
print("begin fit")
# %% fitting
x_data_filtered1, y_data_filtered1 = create_reg_array1('1830', frame, filter)
x_data_filtered7, y_data_filtered7 = create_reg_array1('1837', frame, filter)
x_data_filtered10, y_data_filtered10 = create_reg_array1('183T', frame, filter)
plt.figure()
popt, pcov = curve_fit(linear, x_data_filtered, y_data_filtered)
plt.plot(x_data_filtered10, y_data_filtered10, 'g.', label=r'$183\pm 10$', alpha=0.4)
plt.plot(x_data_filtered7, y_data_filtered7, 'r.', label=r'$183\pm 7$', alpha=0.4)
plt.plot(x_data_filtered1, y_data_filtered1, 'b.', label=r'$183\pm 1$', alpha=0.4)
# plt.plot(x_data_filtered, linear(x_data_filtered, *popt), 'r-', label='fit: m=%5.3f, b=%5.3f' % tuple(popt))
plt.xlabel(r'$\Delta$ aos_1830BT / 30s')
plt.ylabel('W_at_BT')
plt.title('Linear regression between Δaos_1830BT and W_at_BT')
plt.ylim(np.min(y_data_filtered),np.max(y_data_filtered))
plt.legend()
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
plt.imshow(y_data[t,:,:]-x_data[t,:,:], origin='lower', cmap='viridis')
plt.colorbar()
plt.title(f'W_at_BT - Δaos_1830BT/30s at t={t}')
plt.show()
