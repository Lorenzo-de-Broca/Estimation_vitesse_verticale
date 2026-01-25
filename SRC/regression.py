import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
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

def test_model(model, X_test, y_test, model_name="Linear Regression"):
    """Tests the linear regression model on test data and computes the mean squared error

    Args:
        model (LinearRegression): trained linear regression model
        X_test (np.array): 2D array (n_samples, n_features) of test input data
        y_test (np.array): 1D array (n_samples,) of test output data

    Returns:
        rmse (float): root mean squared error of the model on the test data
    """
    print("X_test shape", X_test.shape)
    if model_name == "Neuronal Network":
        y_pred = model.predict(X_test)[:,0]
    else:
        y_pred = model.predict(X_test)
    print("y_pred shape", y_pred.shape)
    
    # Flatten y_test to 1D if it's 2D to avoid broadcasting issues
    if model_name == "Random Forest":
        y_test = y_test.ravel()

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    residuals = y_test - y_pred

    if model_name == "Neuronal Network":
        r2 = model.evaluate(X_test, y_test, verbose=1)
    else:
        r2 = model.score(X_test, y_test)

    return y_pred, rmse, residuals, r2

def random_forest_reg(x_data, y_data, n_estimators=10, max_depth=20, 
                      min_samples_split=5, random_state=42):
    """Trains a Random Forest regression model
    
    Args:
        x_data (np.array): 2D array (n_samples, n_features) of input data
        y_data (np.array): 1D array (n_samples,) of output data
        n_estimators (int): Number of trees in the forest (default: 100)
        max_depth (int): Maximum depth of each tree (default: 20)
        min_samples_split (int): Minimum samples required to split (default: 5)
        random_state (int): Random seed for reproducibility (default: 42)
    
    Returns:
        model (RandomForestRegressor): trained random forest model
    """
    y_data = y_data.ravel()  # Ensure y_data is 1D
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1  # Use all available processors
    )
    
    print(f"Training Random Forest with {n_estimators} trees...")
    model.fit(x_data, y_data)
    print(f"Training complete!")
    print(f"Feature importance (top 10):")
    
    # Print feature importances (top 10)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    for i, idx in enumerate(indices):
        print(f"  {i+1}. Feature {idx}: {importances[idx]:.4f}")
    
    return model
