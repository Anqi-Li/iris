#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:27:07 2019

@author: anqil
"""

import numpy as np
import xarray as xr
import matplotlib.pylab as plt
from osirisl1services.readlevel1 import open_level1_ir
from osirisl1services.services import Level1Services
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
from netCDF4 import num2date
units = 'days since 1858-11-17 00:00:00.000'

#%% load IRIS data
channel = 3
orbit = 37608

ir = open_level1_ir(orbit, channel, valid=False)
altitude = ir.l1.altitude.sel(pixel=slice(14, 128))
latitude = ir.l1.latitude.sel(pixel=slice(14, 128))
longitude = ir.l1.longitude.sel(pixel=slice(14, 128))
#sc_look = ir.l1.look_ecef.sel(pixel=slice(14, 128))
#sc_pos = ir.l1.position_ecef
l1 = ir.data.sel(pixel=slice(14, 128)) 
mjd = ir.mjd.data
pixel = ir.pixel.sel(pixel=slice(14, 128)).data

# calculate sza
import astropy.coordinates as coord
from astropy.time import Time
import astropy.units as u

loc = coord.EarthLocation(lon=longitude.sel(pixel=60) * u.deg,
                          lat=latitude.sel(pixel=60) * u.deg)
now = Time(ir.mjd, format='mjd', scale='utc')
altaz = coord.AltAz(location=loc, obstime=now)
sun = coord.get_sun(now)
sza = 90 - sun.transform_to(altaz).alt.deg
sza = xr.DataArray(sza, coords=(mjd,), dims=('mjd',), name='sza')

lst = np.mod(longitude.sel(pixel=60)/15 + np.modf(mjd)[0]*24, 24)
lst = xr.DataArray(lst, coords=(mjd,), dims=('mjd',), name='sza')

#%% load climatology
path = '/home/anqil/Documents/osiris_database/ex_data/'
file = 'msis_cmam_climatology.nc'
clima = xr.open_dataset(path+file)
clima = clima.update({'m':clima.o + clima.o2 + clima.n2})
o3_clima = clima.o3_vmr * clima.m

#%% load smr
path = '/home/anqil/Documents/osiris_database/ex_data/'
file = 'smr_2008_sza_interp.nc'
smr = xr.open_dataset(path+file)
