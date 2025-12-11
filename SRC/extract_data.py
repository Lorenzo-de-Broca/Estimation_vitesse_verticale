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

def create_reg_arrays1(freq, frame, filter):
    """
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
    for t in range(87):
        x_data[t,:,:] = (frame[f'aos_{freq}BT'][t+1,:,:]-frame[f'aos_{freq}BT'][t,:,:])*filter[t,:,:]*filter[t+1,:,:]/30

    x_data_filtered = x_data[np.nonzero(x_data)]

    y_data = frame['W_at_BT'][:87,:,:]
    y_data_filtered = y_data[np.nonzero(x_data)]

    return x_data_filtered, y_data_filtered

