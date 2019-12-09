#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:38:05 2019

@author: anqili
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.time import Time

#%%
year = [2008]
month = [10]

#%% load iris ver
path = '/home/anqil/Documents/osiris_database/iris_ver_o3/'
filenames = 'ver_{}{}_v5p0.nc'
files = [path+filenames.format(year[i], str(month[i]).zfill(2)) for i in range(len(month))]
data_ver = xr.open_mfdataset(files)
data_ver = data_ver.assign_coords(latitude = data_ver.latitude, 
                                            longitude = data_ver.longitude)

#%% load iris o3
path = '/home/anqil/Documents/osiris_database/iris_ver_o3/'
filenames = 'o3_{}{}_mr08_o3false.nc'
files = [path+filenames.format(year[i], str(month[i]).zfill(2)) for i in range(len(month))]
data_o3 = xr.open_mfdataset(files)

if (len(data_ver.mjd) == len(data_o3.mjd)):
    data_o3 = data_o3.assign_coords(latitude = data_ver.latitude, 
                                            longitude = data_ver.longitude)

#%% look at retrieved ozone zonal mean
lat_bins = np.linspace(-90, 90, 46, endpoint=True)
lat_bins_center = np.diff(lat_bins)/2+lat_bins[:-1]
vmin=1e6#1e4 #1e6
vmax=1e10# 1e7 #1e10
mean_o3 = data_o3.o3.where(data_o3.cost_lsq<1e0).groupby_bins(
                        'latitude', lat_bins, 
                                 labels=lat_bins_center).mean(dim='mjd')
mean_ver = data_ver.ver.where(
        data_ver.mr>0.8).groupby_bins('latitude', lat_bins, 
                                 labels=lat_bins_center).mean(dim='mjd')

fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=False, figsize=(10,5))
#fig.tight_bbox()
mean_o3.plot(y='z', norm=LogNorm(vmin=vmin, vmax=vmax), ax=ax[0], cmap='RdBu_r')
mean_ver.plot(y='z', norm=LogNorm(vmin=0.9e5, vmax=1e6), ax=ax[1], cmap='RdBu_r')
ax[0].set(title='O3')
ax[1].set(title='VER')
