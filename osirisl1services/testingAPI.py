#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 14:38:25 2018

@author: anqil
"""
#%%
import numpy as np
import xarray as xr
import warnings
from osirisl1services.readlevel1 import open_level1_ir
from osirisl1services.services import Level1Services
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
from netCDF4 import num2date
units = 'days since 1858-11-17 00:00:00.000'


#%%

channel = 3
ir = open_level1_ir(orbit=20900, channel=channel)

print(ir)

#print(ir.l1.look_ecef)
#print(ir.l1.position_ecef)
print(ir.l1.altitude)

#%%

FIG_SIZE = (15, 6)

alts_interp = np.arange(10e3, 110e3, .25e3)
ir_altitude = []
for (data, alt) in zip(ir.data, ir.l1.altitude):
    f = interp1d(alt, data, bounds_error=False)
    ir_altitude.append(f(alts_interp))
ir_altitude = xr.DataArray(ir_altitude, coords=[ir.mjd, alts_interp], dims=['mjd', 'altitude'])
ir_altitude.plot(x='mjd', y='altitude', norm=LogNorm(), vmin=1e9, vmax=1e12, figsize=FIG_SIZE)


#
#from osirisl1services.readlevel1 import open_level1_ecmwf
#ecmwf = open_level1_ecmwf(scanno=6432012)

#%% take all scan in 1 orbit and save all in list
channel = 1

orbit = 18519
scanno_start = orbit*1000
scanno_end = scanno_start + 70 

ir = []
altitude = [] 
for scan in range(scanno_start,scanno_end):
    try:
        temp = open_level1_ir(scanno=scan, channel=channel)
#        print(scan)
        ir.append(temp)
        altitude.append(temp.l1.altitude)
    except:
        pass

#%% take another orbit
orbit += 1
scanno_start = orbit*1000
scanno_end = scanno_start + 70 
for scan in range(scanno_start,scanno_end):
    try:
        temp = open_level1_ir(scanno=scan, channel=channel)
#        print(scan)
        ir.append(temp)
        altitude.append(temp.l1.altitude)
    except:
        pass

# %%plot

alts_interp = np.arange(10e3, 110e3, .25e3)
data_interp = []
ir = xr.concat(ir, dim='mjd')
altitude = xr.concat(altitude, dim='mjd')

for (data, alt) in zip(ir.data, altitude):
    f = interp1d(alt, data, bounds_error=False)
    data_interp.append(f(alts_interp))
data_interp = xr.DataArray(data_interp, coords=[ir.mjd, alts_interp], dims=['mjd', 'altitude'])

FIG_SIZE = (15,6)
data_interp.plot(x='mjd', y='altitude', norm=LogNorm(), vmin=1e9, vmax=1e12, figsize=FIG_SIZE)
plt.title(str(num2date(ir.mjd[0],units))+' channel '+str(channel))



