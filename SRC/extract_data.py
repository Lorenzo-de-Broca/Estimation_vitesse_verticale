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

def extract_data():
    """
    Parameters
    ----------
    path : str
        path where to find the netCDF «MesoNH-ice3_CADDIWAF7_1km_projectHB.nc» file

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
    frame = {'time': ds.variables['time'][:],
             'longitude': ds.variables['longitude'][:],
             'lattitude': ds.variables['latitude'][:],
             'ni': ds.variables['ni'][:],
             'nj': ds.variables['nj'][:],
             'aos_1830BT': ds.variables['aos_1830BT'][:],
             'aos_1833BT': ds.variables['aos_1833BT'][:],
             'aos_1835BT': ds.variables['aos_1835BT'][:],
             'aos_1837BT': ds.variables['aos_1837BT'][:],
             'aos_183TBT': ds.variables['aos_183TBT'][:],
             'aos_3250BT': ds.variables['aos_3250BT'][:],
             'aos_3253BT': ds.variables['aos_3253BT'][:],
             'aos_3255BT': ds.variables['aos_3255BT'][:],
             'aos_3257BT': ds.variables['aos_3257BT'][:],
             'aos_325TBT': ds.variables['aos_325TBT'][:],
             'W_at_BT': ds.variables['W_at_BT'][:]}
    
    return frame

