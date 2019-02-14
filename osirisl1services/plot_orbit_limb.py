#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 15:23:22 2019

@author: anqil
"""

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
channel = 1

orbit = 8564 #start orbit
ir = []
altitude = [] 
i = 0
while i<5: #choose how many orbits you want
    scanno_start = orbit*1000
    scanno_end = scanno_start + 60 #assuming there are max 60 scan per orbit 

    for scan in range(scanno_start,scanno_end):
        try:
            temp = open_level1_ir(scanno=scan, channel=channel)
#           print(scan)
            ir.append(temp)
            altitude.append(temp.l1.altitude)
        except:
            pass
    orbit += 1
    i += 1
    
#%% interpolation 
alts_interp = np.arange(10e3, 110e3, .25e3)
data_interp = []
ir = xr.concat(ir, dim='mjd')
altitude = xr.concat(altitude, dim='mjd')

for (data, alt) in zip(ir.data, altitude):
    f = interp1d(alt, data, bounds_error=False)
    data_interp.append(f(alts_interp))
data_interp = xr.DataArray(data_interp, coords=[ir.mjd, alts_interp], dims=['mjd', 'altitude'])

#%%plotting
FIG_SIZE = (15,6)
data_interp.plot(x='mjd', y='altitude', norm=LogNorm(), vmin=1e9, vmax=1e13, figsize=FIG_SIZE)
plt.title(str(num2date(ir.mjd[0],units))+' channel '+str(channel))
