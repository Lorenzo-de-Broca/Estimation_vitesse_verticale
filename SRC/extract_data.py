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

def create_combined_regression_array_delta_t(frame,filter,train_matrix = default_train):
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

def create_data_neuronal(frame, filter, train_ratio, pop_size):
    """
    creatin training an testong data for neuronal network
    """
    train_mat, test_mat = create_train_test_matrix(train_ratio=train_ratio)

    DTB_1830_train, W_filtered_1830_train = create_reg_array2('1830', frame, filter, train_mat)
    DTB_1833_train, W_filtered_1833_train = create_reg_array2('1833', frame, filter, train_mat)
    DTB_1835_train, W_filtered_1835_train = create_reg_array2('1835', frame, filter, train_mat)
    DTB_1837_train, W_filtered_1837_train = create_reg_array2('1837', frame, filter, train_mat)
    DTB_183T_train, W_filtered_183T_train = create_reg_array2('183T', frame, filter, train_mat)
    DTB_3250_train, W_filtered_3250_train = create_reg_array2('3250', frame, filter, train_mat)
    DTB_3253_train, W_filtered_3253_train = create_reg_array2('3253', frame, filter, train_mat)
    DTB_3255_train, W_filtered_3255_train = create_reg_array2('3255', frame, filter, train_mat)
    DTB_3257_train, W_filtered_3257_train = create_reg_array2('3257', frame, filter, train_mat)
    DTB_325T_train, W_filtered_325T_train = create_reg_array2('325T', frame, filter, train_mat)

    DTB_1830_test, W_filtered_1830_test = create_reg_array2('1830', frame, filter, test_mat)
    DTB_1833_test, W_filtered_1833_test = create_reg_array2('1833', frame, filter, test_mat)
    DTB_1835_test, W_filtered_1835_test = create_reg_array2('1835', frame, filter, test_mat)
    DTB_1837_test, W_filtered_1837_test = create_reg_array2('1837', frame, filter, test_mat)
    DTB_183T_test, W_filtered_183T_test = create_reg_array2('183T', frame, filter, test_mat)
    DTB_3250_test, W_filtered_3250_test = create_reg_array2('3250', frame, filter, test_mat)
    DTB_3253_test, W_filtered_3253_test = create_reg_array2('3253', frame, filter, test_mat)
    DTB_3255_test, W_filtered_3255_test = create_reg_array2('3255', frame, filter, test_mat)
    DTB_3257_test, W_filtered_3257_test = create_reg_array2('3257', frame, filter, test_mat)
    DTB_325T_test, W_filtered_325T_test = create_reg_array2('325T', frame, filter, test_mat)

    DTB_1830 = np.array([(frame['aos_1830BT'][t+1,:,:]-frame['aos_1830BT'][t,:,:])/30 for t in range(87)])
    DTB_1833 = np.array([(frame['aos_1833BT'][t+1,:,:]-frame['aos_1833BT'][t,:,:])/30 for t in range(87)])
    DTB_1835 = np.array([(frame['aos_1835BT'][t+1,:,:]-frame['aos_1835BT'][t,:,:])/30 for t in range(87)])
    DTB_1837 = np.array([(frame['aos_1837BT'][t+1,:,:]-frame['aos_1837BT'][t,:,:])/30 for t in range(87)])
    DTB_183T = np.array([(frame['aos_183TBT'][t+1,:,:]-frame['aos_183TBT'][t,:,:])/30 for t in range(87)])
    DTB_3250 = np.array([(frame['aos_3250BT'][t+1,:,:]-frame['aos_3250BT'][t,:,:])/30 for t in range(87)])
    DTB_3253 = np.array([(frame['aos_3253BT'][t+1,:,:]-frame['aos_3253BT'][t,:,:])/30 for t in range(87)])
    DTB_3255 = np.array([(frame['aos_3255BT'][t+1,:,:]-frame['aos_3255BT'][t,:,:])/30 for t in range(87)])
    DTB_3257 = np.array([(frame['aos_3257BT'][t+1,:,:]-frame['aos_3257BT'][t,:,:])/30 for t in range(87)])
    DTB_325T = np.array([(frame['aos_325TBT'][t+1,:,:]-frame['aos_325TBT'][t,:,:])/30 for t in range(87)])

    np.random.seed(42)
    size = pop_size
    indices_train = np.random.choice(len(DTB_1830_train), size=size, replace=False)
    DTB_1830_train, DTB_1833_train, DTB_1835_train, DTB_1837_train, DTB_183T_train = DTB_1830_train[indices_train], DTB_1833_train[indices_train], DTB_1835_train[indices_train], DTB_1837_train[indices_train], DTB_183T_train[indices_train]
    DTB_3250_train, DTB_3253_train, DTB_3255_train, DTB_3257_train, DTB_325T_train = DTB_3250_train[indices_train], DTB_3253_train[indices_train], DTB_3255_train[indices_train], DTB_3257_train[indices_train], DTB_325T_train[indices_train]
    W_filtered_train = W_filtered_1830_train[indices_train]

    indices_test = np.random.choice(len(DTB_1830_test), size=int((1-train_ratio)/train_ratio*size), replace=False)
    DTB_1830_test, DTB_1833_test, DTB_1835_test, DTB_1837_test, DTB_183T_test = DTB_1830_test[indices_test], DTB_1833_test[indices_test], DTB_1835_test[indices_test], DTB_1837_test[indices_test], DTB_183T_test[indices_test]
    DTB_3250_test, DTB_3253_test, DTB_3255_test, DTB_3257_test, DTB_325T_test = DTB_3250_test[indices_test], DTB_3253_test[indices_test], DTB_3255_test[indices_test], DTB_3257_test[indices_test], DTB_325T_test[indices_test]
    W_filtered_test = W_filtered_1830_test[indices_test]

    x_data_train=np.array([DTB_1830_train, DTB_1833_train, DTB_1835_train, DTB_1837_train, DTB_183T_train, DTB_3250_train, DTB_3253_train, DTB_3255_train, DTB_3257_train, DTB_325T_train]).T
    x_data_test=np.array([DTB_1830_test, DTB_1833_test, DTB_1835_test, DTB_1837_test, DTB_183T_test, DTB_3250_test, DTB_3253_test, DTB_3255_test, DTB_3257_test, DTB_325T_test]).T

    t=0
    x_data_pred = (np.array([DTB_1830[t,:,:].reshape(-1,1), DTB_1833[t,:,:].reshape(-1,1), DTB_1835[t,:,:].reshape(-1,1), DTB_1837[t,:,:].reshape(-1,1), DTB_183T[t,:,:].reshape(-1,1), DTB_3250[t,:,:].reshape(-1,1), DTB_3253[t,:,:].reshape(-1,1), DTB_3255[t,:,:].reshape(-1,1), DTB_3257[t,:,:].reshape(-1,1), DTB_325T[t,:,:].reshape(-1,1)]).T)[0]
    W_filtered_pred = frame['W_at_BT'][t,:,:].reshape(-1,1)

    return x_data_train, W_filtered_train, x_data_test, W_filtered_test, x_data_pred, W_filtered_pred

if __name__ == "__main__":
    frame = extract_data()
    print("Data extracted from netCDF file.")
    