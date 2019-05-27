#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 18:44:23 2018

@author: anqil
"""

from osirisl1services.readlevel1 import open_level1_ir
from osirisl1services.services import Level1Services

orbit = 6432
#scan = 14
#scanno = int(str(orbit) + ('%03d' % scan))

channel = 1
ncfilename = 'test'+'.nc'


ir = open_level1_ir(orbit=orbit, channel=channel)
#ir.mjd.attrs['units'] = 'days since 1858-11-17 00:00:00.0'


print('data')
ir.data.to_netcdf(ncfilename)
ir.error.to_netcdf(ncfilename, mode='a')
ir.flags.to_netcdf(ncfilename, mode='a')

print('other things')
ir.stw.to_netcdf(ncfilename, mode='a')
ir.exposureTime.to_netcdf(ncfilename, mode='a')
ir.temperature.to_netcdf(ncfilename, mode='a')
ir.tempavg.to_netcdf(ncfilename, mode='a')
ir.mode.to_netcdf(ncfilename, mode='a')
ir.scienceprog.to_netcdf(ncfilename, mode='a')
ir.shutter.to_netcdf(ncfilename, mode='a')
ir.lamp1.to_netcdf(ncfilename, mode='a')
ir.lamp2.to_netcdf(ncfilename, mode='a')
ir.targetIndex.to_netcdf(ncfilename, mode='a')
ir.exceptions.to_netcdf(ncfilename, mode='a')
ir.processingflags.to_netcdf(ncfilename, mode='a')

ir.channel.to_netcdf(ncfilename, mode='a')

print('tangent points')
ir.l1.altitude.rename('altitude').to_netcdf(ncfilename, mode='a')
ir.l1.longitude.rename('longitude').to_netcdf(ncfilename, mode='a')
ir.l1.latitude.rename('latitude').to_netcdf(ncfilename, mode='a')
ir.l1.sza.rename('sza').to_netcdf(ncfilename, mode='a')

#%%testing the saved nc file
from netCDF4 import Dataset
from netCDF4 import num2date, date2num

data = Dataset(ncfilename)

alt = data.variables['altitude']
l1 = data.variables['data']
mjd = data.variables['mjd']
units = 'days since 1858-11-17 00:00:00.0'

#%%
import matplotlib.pyplot as plt
import numpy as np
plt.semilogx(l1[20,:], alt[20,:])

plt.figure()
plt.plot(alt[:,60])


