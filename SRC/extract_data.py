#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 12:03:59 2025

@author: titouanrenaud

Script to extract data from netCDF file and return a dictionnary with nd-arrays.
"""

import netCDF4 as nc
import yaml
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


default_train = np.ones((500, 500))

def extract_data():
    """
    Parameters
    ----------

    Returns
    -------
    frame : dict
        dictionnary with all extracted variables as masked arrays
        - time: 88 times steps every 30 seconds (88,)
        - longitude, lattitude: (500, 500)
        - aos: brightness temperatures (88, 500, 500)
        - W_at_BT: vertical velocity (88, 500, 500)

    """
    
    base_dir = Path(__file__).resolve().parent.parent

    # path to path.yaml
    path_yaml = base_dir / "inputs/paths.yaml"

    with open(path_yaml, 'r') as f:
        path = yaml.safe_load(f)
    
    path = path['data_file']

    ds = nc.Dataset(path)
    frame = {'time': np.array(ds.variables['time'][:]),
             'longitude': np.array(ds.variables['longitude'][:]),
             'lattitude': np.array(ds.variables['latitude'][:]),
             'ni': np.array(ds.variables['ni'][:]),
             'nj': np.array(ds.variables['nj'][:]),
             'aos_1830BT': np.array(ds.variables['aos_1830BT'][:]),
             'aos_1833BT': np.array(ds.variables['aos_1833BT'][:]),
             'aos_1835BT': np.array(ds.variables['aos_1835BT'][:]),
             'aos_1837BT': np.array(ds.variables['aos_1837BT'][:]),
             'aos_183TBT': np.array(ds.variables['aos_183TBT'][:]),
             'aos_3250BT': np.array(ds.variables['aos_3250BT'][:]),
             'aos_3253BT': np.array(ds.variables['aos_3253BT'][:]),
             'aos_3255BT': np.array(ds.variables['aos_3255BT'][:]),
             'aos_3257BT': np.array(ds.variables['aos_3257BT'][:]),
             'aos_325TBT': np.array(ds.variables['aos_325TBT'][:]),
             'W_at_BT': ds.variables['W_at_BT'][:]}
    
    return frame

def create_reg_array1(freq, frame, filter):
    """
    Create arrays of Delta_TB filtered by convection filter only

    Parameters
    ----------
    freq : str
        frequency channel among ['1830', '1833', '1835', '1837', '183T', '3250', '3253', '3255', '3257', '325T']

    Returns
    -------
    x_data : np.ndarray
        1D array of shape (N,) with Δaos_freqBT/30s filtered
    y_data : np.ndarray
        1D array of shape (N,) with W_at_BT filtered

    """
        
    x_data = np.zeros((87, 500, 500))
    index_filter = np.zeros((87, 500, 500))
    for t in range(87):
        x_data[t,:,:] = (frame[f'aos_{freq}BT'][t+1,:,:]-frame[f'aos_{freq}BT'][t,:,:])/30
        index_filter[t,:,:] = filter[t,:,:]*filter[t+1,:,:]

    x_data_filtered = x_data[np.nonzero(index_filter)]

    y_data = frame['W_at_BT'][:87,:,:]
    y_data_filtered = y_data[np.nonzero(index_filter)]

    return x_data_filtered, y_data_filtered

def create_train_test_matrix(train_ratio=0.6):
    """
    Create train and test matrix for masking data points

    Parameters
    ----------
    train_ratio : float, optional
        ratio of data points used for training. The default is 0.6.

    Returns
    -------
    train_matrix : np.ndarray
        2D array of shape (500, 500) with 1 for training points and 0 elsewhere
    test_matrix : np.ndarray
        2D array of shape (500, 500) with 1 for testing points and 0 elsewhere  

    """
    total_len = 500*500
    all_indexes = np.arange(total_len)
    np.random.shuffle(all_indexes)
    train_size = int(train_ratio * total_len)
    train_indexes = all_indexes[:train_size]
    test_indexes = all_indexes[train_size:]

    train_matrix = np.zeros((500, 500))
    test_matrix = np.zeros((500, 500))

    np.put(train_matrix, train_indexes, 1)
    np.put(test_matrix, test_indexes, 1)

    return train_matrix, test_matrix

def create_reg_array2(freq, frame, filter, train_matrix=default_train):
    """
    Create arrays of Delta_TB filtered by convection filter and training

    Parameters
    ----------
    freq : str
        frequency channel among ['1830', '1833', '1835', '1837', '183T', '3250', '3253', '3255', '3257', '325T']

    Returns
    -------
    x_data : np.ndarray
        1D array of shape (N,) with Δaos_freqBT/30s filtered
    y_data : np.ndarray
        1D array of shape (N,) with W_at_BT filtered

    """
        
    x_data = np.zeros((87, 500, 500))
    index_filter = np.zeros((87, 500, 500))
    for t in range(87):
        x_data[t,:,:] = (frame[f'aos_{freq}BT'][t+1,:,:]-frame[f'aos_{freq}BT'][t,:,:])/30
        index_filter[t,:,:] = filter[t,:,:]*filter[t+1,:,:]*train_matrix

    x_data_filtered = x_data[np.nonzero(index_filter)]

    y_data = frame['W_at_BT'][:87,:,:]
    y_data_filtered = y_data[np.nonzero(index_filter)]

    return x_data_filtered, y_data_filtered

def create_reg_array3(freq, frame, filter, train_matrix = default_train):
    """
    Create arrays of raw TB filtered by convection filter and training

    Parameters
    ----------
    freq : str
        frequency channel among ['1830', '1833', '1835', '1837', '183T', '3250', '3253', '3255', '3257', '325T']

    Returns
    -------
    x_data : np.ndarray
        1D array of shape (N,) with Δaos_freqBT/30s filtered
    y_data : np.ndarray
        1D array of shape (N,) with W_at_BT filtered

    """
        
    index_filter = np.zeros((88, 500, 500))
    for t in range(88):
        index_filter[t,:,:] = filter[t,:,:]*train_matrix

    x_data_filtered = frame[f'aos_{freq}BT'][np.nonzero(index_filter)]

    y_data = frame['W_at_BT'][:88,:,:]
    y_data_filtered = y_data[np.nonzero(index_filter)]

    return x_data_filtered, y_data_filtered

def create_combined_regression_array_deltat(frame,filter,train_matrix = default_train):
    """
    This function create a matrix containing filtered datas of all differents experimental measurement in order to compute PCA or covariance for instance

    """
    freqs = ['1830', '1833', '1835', '1837', '183T', '3250', '3253', '3255', '3257', '325T']
    
    combined_x, combined_y = create_reg_array2(freqs[0], frame, filter, train_matrix)
    
    n_line = np.shape(combined_x)[0]

    combined_x = combined_x.reshape(n_line,1)
    combined_y = combined_y.reshape(n_line,1)
    
    for f in freqs[1:] :
        x_filtered, y_filtered = create_reg_array2(f, frame, filter, train_matrix)
        combined_x = np.append(combined_x, x_filtered.reshape(n_line,1),axis=1)
    
    return combined_x, combined_y

def create_combined_regression_array(frame,filter,train_matrix):
    """
    This function create a matrix containing filtered datas of all differents experimental measurement in order to compute PCA or covariance for instance

    """
    freqs = ['1830', '1833', '1835', '1837', '183T', '3250', '3253', '3255', '3257', '325T']
  
    combined_x, combined_y = create_reg_array3(freqs[0], frame, filter, train_matrix)
    
    n_line = np.shape(combined_x)[0]

    combined_x = combined_x.reshape(n_line,1)
    combined_y = combined_y.reshape(n_line,1)
    
    for f in freqs[1:] :
        x_filtered, y_filtered = create_reg_array3(f, frame, filter, train_matrix)
        combined_x = np.append(combined_x, x_filtered.reshape(n_line,1),axis=1)
    
    return combined_x, combined_y

    
def create_PCA (combined_x, combined_y, pca_components):
    """
    This function compute PCA on combined regression array

    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(combined_x)  # X = ton tableau de mesures

    pca = PCA(n_components=pca_components)  # conserve 95% de l'information
    X_pca = pca.fit_transform(X_scaled)
    
    print(pca.explained_variance_ratio_)
    print("Variance cumulée :", np.cumsum(pca.explained_variance_ratio_))
    print("Composantes principales :", pca.components_)
    print("Nombre de composantes principales conservées :", pca.n_components_)

    print("Shape of X_pca:", X_pca.shape)
    print("PCA computed.")
    
    return X_pca



if __name__ == "__main__":
    frame = extract_data()
    print("Data extracted from netCDF file.")
    