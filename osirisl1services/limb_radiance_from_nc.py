#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 18:11:09 2019

@author: anqil
"""

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from netCDF4 import num2date, date2num
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
import xarray as xr

#%% single scan
#folder = '/home/anqil/Documents/OSIRIS_IR/IR_ch1/2004/'
#orbit = 18514
#scan = 0
#scanno = int(('%04d' % orbit) + ('%03d' % scan))
#filename = str(scanno)+'.nc'
#file = folder+filename
#data = Dataset(file, mode='r+')
#lat = data.variables['latitude'] # degrees north
#lon = data.variables['longitude'] # degrees east
#tang_alt = data.variables['altitude'] # m
#l1 = data.variables['data']
#mjd = data.variables['mjd']

#%% loop over all sacans in one orbit
ch = 1
folder = '/home/anqil/Documents/OSIRIS_IR/IR_ch'+str(ch)+'/2004/'
orb = range(18500, 18520)
sc = range(70)
lat = np.empty((0,128))
lon = np.empty((0,128))
tang_alt = np.empty((0,128))
l1 = np.empty((0,128))
mjd = np.empty(0)
sza = np.empty((0,128))
scanno_memo = np.empty(0)
scanno_mask = np.empty(0)

for orbit in orb:
    for scan in sc:
        scanno = int(('%04d' % orbit) + ('%03d' % scan))
        filename = str(scanno)+'.nc'
        file = folder+filename
        
        try:
            data = Dataset(file)
            lat = np.append(lat, data.variables['latitude'][:].data,axis=0) # degrees north
            lon = np.append(lon, data.variables['longitude'][:].data,axis=0) # degrees east
            tang_alt = np.append(tang_alt, data.variables['altitude'][:].data,axis=0) # m
            l1 = np.append(l1, data.variables['data'][:].data,axis=0)
            mjd = np.append(mjd, data.variables['mjd'][:].data)
            sza = np.append(sza, data.variables['sza'][:].data, axis=0)
            scanno_memo = np.append(scanno_memo, scanno)
            scanno_mask = np.append(scanno_mask, True)
#            print(scanno)
        except:
            scanno_mask = np.append(scanno_mask, False)
#            print('do not find scanno='+str(scanno))
            pass

#%% interpolate on a fixed altitude grid and plot
plt.figure()
FIG_SIZE = (15, 6)
alts_interp = np.arange(10e3, 120e3, .25e3)
ir_altitude = []
for (l1_data, alt) in zip(l1, tang_alt):
    f = interp1d(alt, l1_data, bounds_error=False)
    ir_altitude.append(f(alts_interp))
ir_altitude = xr.DataArray(ir_altitude, 
                           coords=[mjd, alts_interp], 
                           dims=['mjd', 'altitude'])
ir_altitude.plot(x='mjd', y='altitude', 
                 norm=LogNorm(), 
                 vmin=1e9, vmax=1e13, 
                 figsize=FIG_SIZE)
units = 'days since 1858-11-17 00:00:00.000'
start_time = num2date(mjd[0], units)
end_time = num2date(mjd[-1], units)
title = str(start_time.year)+', '+str(start_time.month)+', '+str(start_time.day)+' channel '+str(ch)
plt.title(title)
    
#%%
#units = 'days since 1858-11-17 00:00:00.000'
#fig = plt.figure()
##ax = plt.subplot(111)
#plt.contourf(ir_altitude.mjd, ir_altitude.altitude, ir_altitude.T)
#plt.colorbar()






