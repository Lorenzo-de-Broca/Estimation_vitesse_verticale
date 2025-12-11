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
             'W_at_BT': np.array(ds.variables['W_at_BT'][:])}
    
    return frame

