#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:27:03 2019

@author: anqil
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from astropy.time import Time

#%%

path = '/home/anqil/Documents/osiris_database/iris_ver_o3/'
file = 'ver_o3_sza_376*.nc'
ds = xr.open_mfdataset(path+file)

#%%
lat_bins = np.linspace(-90, 90, 46, endpoint=True)
lat_bins_center = np.diff(lat_bins)/2+lat_bins[:-1]
zonal_mean = ds.o3_iris.where(ds.mr>0.8).groupby_bins(ds.tan_lat.sel(pixel=60), lat_bins,
                                    labels=lat_bins_center).mean(dim='mjd', skipna=True)
zonal_mean.plot(x='tan_lat_bins', vmin=0, vmax=1e8)

#%%
plt.figure()
ds.ver.where(ds.mr>0.8).plot(x='mjd', y='z', 
                            vmin=0, vmax=8e6, 
                            size=5, aspect=3)
plt.figure()
ds.ver_apriori.plot(x='mjd', y='z', 
                    vmin=0, vmax=8e6,
                    size=5, aspect=3)
ax = plt.gca()
ax.set(title='IRIS 1d retrieved VER')
for i in range(0,1000,100):
    ax.axvline(x=ds.mjd[i], color='r')
plt.show()

plt.figure()
ds.sza.plot(x='mjd',
            size=4, aspect=3)



plt.figure()
ds.ver.where(ds.mr>0.8).isel(mjd=np.arange(0,1000,100)).plot.line(y='z', 
                            ls='-', marker=' ',
                            size=5)
ax = plt.gca()
ds.ver_apriori.where(ds.mr>0.8).isel(mjd=np.arange(0,1000,100)).plot.line(y='z', 
                                    ls='-', marker=' ')
                                    
#plt.legend([])
plt.xlim(left=-1e6)
plt.show()