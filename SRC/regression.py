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
    print("X_test shape", X_test.shape)
    y_pred = model.predict(X_test)
    print("y_pred shape", y_pred.shape)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    residuals = y_test - y_pred

    r2 = model.score(X_test, y_test)

    return y_pred, rmse, residuals, r2
