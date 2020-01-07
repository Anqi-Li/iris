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
month = [3]

#%% load iris ver
path = '/home/anqil/Documents/osiris_database/iris_ver_o3/'
filenames = 'ver_{}{}_v5p0.nc'
files = [path+filenames.format(year[i], str(month[i]).zfill(2)) for i in range(len(month))]
data_ver = xr.open_mfdataset(files)
data_ver = data_ver.assign_coords(latitude = data_ver.latitude, 
                                            longitude = data_ver.longitude)

#%% load iris o3
path = '/home/anqil/Documents/osiris_database/iris_ver_o3/'
#filenames = 'o3_{}{}_mr08_o3false.nc'
#filenames = 'o3_{}{}_5p1_o3false_posVER.nc'
filenames = 'o3_{}{}_v6p0.nc'
files = [path+filenames.format(year[i], str(month[i]).zfill(2)) for i in range(len(month))]
data_o3 = xr.open_mfdataset(files)
#data_o3 = data_o3.where(data_o3.status!=0, drop=True)
data_o3 = data_o3.assign_coords(latitude = data_ver.latitude.sel(mjd=data_o3.mjd), 
                                        longitude = data_ver.longitude.sel(mjd=data_o3.mjd))    

#o3_filter = np.logical_and(data_o3.o3.sel(z=90)<1e9, data_o3.cost_lsq<1e0)
o3_filter = (data_o3.status == 0)

#%% merge ver and o3 data
data_iris = data_o3.merge(data_ver)

#%% look at retrieved ozone zonal mean
lat_bins = np.linspace(-90, 90, 46, endpoint=True)
lat_bins_center = np.diff(lat_bins)/2+lat_bins[:-1]
vmin=1e6#1e4 #1e6
vmax=1e10# 1e7 #1e10

mean_o3 = data_o3.o3.where(o3_filter).groupby_bins('latitude', lat_bins, 
                                 labels=lat_bins_center).mean(dim='mjd')
mean_ver = data_ver.ver.where(data_ver.mr>0.8).groupby_bins('latitude', lat_bins, 
                                 labels=lat_bins_center).mean(dim='mjd')

fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=False, figsize=(10,5))
#fig.tight_bbox()
mean_o3.plot(y='z', norm=LogNorm(vmin=vmin, vmax=vmax), ax=ax[0], cmap='RdBu_r')
mean_ver.plot(y='z', norm=LogNorm(vmin=0.9e5, vmax=1e6), ax=ax[1], cmap='RdBu_r')
ax[0].set(title='O3')
ax[1].set(title='VER')

#%%

count_o3 = data_o3.o3.where(o3_filter).groupby_bins('latitude', lat_bins, 
                                 labels=lat_bins_center).count()
count_ver = data_ver.ver.where(data_ver.mr>0.8).groupby_bins('latitude', lat_bins, 
                                 labels=lat_bins_center).count()

plt.figure()
plt.step(lat_bins_center, count_o3, label='O3')
plt.step(lat_bins_center, count_ver, label='VER')
plt.legend()

#%%
plt.figure()
dict(data_o3.o2delta.where(o3_filter).groupby_bins('latitude',lat_bins, 
     labels=lat_bins_center))[56].plot.line(y='z', xscale='log', add_legend=False, color='r')
dict(data_o3.o2delta.where(~o3_filter).groupby_bins('latitude',lat_bins, 
     labels=lat_bins_center))[56].plot.line(y='z', xscale='log', add_legend=False, color='k')
plt.show()