#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 09:50:30 2025

@author: titouanrenaud
"""

import os
os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2025/bin/universal-darwin'
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
from mpl_toolkits.basemap import Basemap
import math
import netCDF4 as nc

# Specify the file path
file_path = '/Users/titouanrenaud/Documents/M2 ECLAT/STATISTIQUES/PROJET HB/MesoNH-ice3_CADDIWAF7_1km_projectHB.nc'
# Open the NetCDF file
ds = nc.Dataset(file_path)

#%% 

longitude = ds.variables['longitude'][:]
latitude = ds.variables['latitude'][:]
time = ds.variables['time'][:]
ni = ds.variables['ni'][:]
nj = ds.variables['nj'][:]
aos_183TBT = ds.variables['aos_183TBT'][0,:,:]
aos_325TBT = ds.variables['aos_325TBT'][0,:,:]
aos_3255BT0 = ds.variables['aos_3255BT'][0,:,:]
aos_3255BT1 = ds.variables['aos_3255BT'][1,:,:]
aos_3255BT3 = ds.variables['aos_3255BT'][3,:,:]

plt.figure(layout='constrained')

plt.imshow(aos_3255BT1-aos_3255BT0, origin='lower')
plt.colorbar(label=r'aos_325TBT')

plt.show()

#%%

plt.figure(layout='constrained')
map = Basemap(projection='merc', llcrnrlat=min(latitude[:,0])-25, urcrnrlat=max(latitude[:,0])+25,
   llcrnrlon=min(longitude[0,:])-25, urcrnrlon=max(longitude[0,:])+25, resolution='c')

# Draw coastlines and countries
# map.drawcoastlines()
map.drawcountries()

map.shadedrelief()

# x, y = map([min(longitude[0,:]), min(longitude[0,:]), max(longitude[0,:]), max(longitude[0,:])], [min(latitude[:,0]), max(latitude[:,0]), min(latitude[:,0]), max(latitude[:,0])])
x, y = map(longitude, latitude)

plt.scatter(x, y, 1, marker='o', color='red', alpha=0.008)

#%%