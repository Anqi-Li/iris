#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 13:48:48 2019

@author: anqil
"""

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from netCDF4 import num2date, date2num
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
import xarray as xr
import warnings

#%%
channel = 3
orbit = 22015
path = '/home/anqil/Downloads/IR/'
filename = 'ir_for_anqi_ch{}_orbit{}.nc'.format(channel, orbit)
file = path + filename
ir = xr.open_dataset(file)
l1 = ir.data.sel(pixel=slice(14,128))
error = ir.error.sel(pixel=slice(14,128))
altitude = ir.altitude.sel(pixel=slice(14,128))
latitude = ir.latitude
longitude = ir.longitude
sza = ir.sza
look = ir.look_ecef.sel(pixel=slice(14,128))
pos = ir.position_ecef

#%%
warnings.filterwarnings('ignore')

FIG_SIZE = (15, 6)
#l1.plot(x='mjd', cmap=plt.cm.Reds, robust=True,# norm=LogNorm(), 
#        vmin=0, vmax=2000, figsize=FIG_SIZE)

alts = np.arange(60e3, 110e3, .5e3)
ir_altitude = []
for (data, alt) in zip(l1, altitude):
    f = interp1d(alt, data, bounds_error=False)
    ir_altitude.append(f(alts))
ir_altitude = xr.DataArray(ir_altitude, coords=[ir.mjd, alts], dims=['mjd', 'altitude'])

plt.figure()
ir_altitude.plot(x='mjd', y='altitude', figsize=FIG_SIZE)

plt.figure()
ir_altitude.isel(mjd=0).plot.line(y='altitude')
plt.legend([])
plt.show()

#%%

from osirisl1services.readlevel1 import open_level1_ir
from osirisl1services.services import Level1Services
import sys
sys.path.append('..')
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d

#channel = 3
#orbit = 20900
ir = open_level1_ir(orbit, channel, valid=False)
tan_alt = ir.l1.altitude
tan_lat = ir.l1.latitude
tan_lon = ir.l1.longitude
sc_look = ir.l1.look_ecef
sc_pos = ir.l1.position_ecef
l1 = ir.data
mjd = ir.mjd.data
pixel = ir.pixel.data
#%%
alts = np.arange(60e3, 110e3, .5e3)
ir_altitude_old = []
for (data, alt) in zip(l1, tan_alt):
    f = interp1d(alt, data, bounds_error=False)
    ir_altitude_old.append(f(alts))
ir_altitude_old = xr.DataArray(ir_altitude_old, coords=[ir.mjd, alts], dims=['mjd', 'altitude'])

plt.figure()
ir_altitude_old.isel(mjd=0).plot.line(y='altitude')
plt.legend([])
plt.show()

plt.figure()
ir_altitude_old.plot(x='mjd', y='altitude', robust=True, figsize=FIG_SIZE)