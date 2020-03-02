#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 13:44:56 2020

@author: anqil
"""

import numpy as np
import scipy.sparse as sp
import xarray as xr
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
from netCDF4 import num2date
units = 'days since 1858-11-17 00:00:00.000'

#%% load IRIS limb radiance data 
orbit100 = np.arange(374, 420)
filename = 'ir_slc_{}xx_ch3.nc'
path = 'https://arggit.usask.ca/opendap/'
#ir = xr.open_mfdataset([path+filename.format(orbit100[i]) for i in range(len(orbit100))])
save_dir = '/home/anqil/Documents/osiris_database/odin-osiris.usask.ca/IR_slc/'
for i in range(len(orbit100)):
    print(orbit100[i])
    ir = xr.open_mfdataset([path+filename.format(orbit100[i])])
    ir.to_netcdf(save_dir+filename.format(orbit100[i]))

