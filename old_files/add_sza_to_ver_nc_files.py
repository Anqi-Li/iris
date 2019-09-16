#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 15:39:36 2019

@author: anqil
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
Av = 6.023e23 #Avogadro's number: molec/mol
from matplotlib.colors import LogNorm
from osirisl1services.readlevel1 import open_level1_ir
from osirisl1services.services import Level1Services
import astropy.coordinates as coord
from astropy.time import Time
import astropy.units as u
from multiprocessing import Pool
#%%

def add_loc_to_ds(orbit):
    path = '/home/anqil/Documents/osiris_database/iris_ver_o3/'
    file = 'ver_o3_{}.nc'.format(orbit)
#    ds = xr.open_dataset(path+file)
    with xr.open_dataset(path+file) as ds:
        
        ir = open_level1_ir(orbit, channel=3, valid=False)
        tan_alt = ir.l1.altitude.sel(pixel=slice(14, 128)).sel(mjd=ds.mjd)
        tan_lat = ir.l1.latitude.sel(pixel=slice(14, 128)).sel(mjd=ds.mjd)
        tan_lon = ir.l1.longitude.sel(pixel=slice(14, 128)).sel(mjd=ds.mjd)
        loc = coord.EarthLocation(lon=tan_lon.sel(pixel=60) * u.deg,
                                  lat=tan_lat.sel(pixel=60) * u.deg)
        now = Time(ds.mjd, format='mjd', scale='utc')
        altaz = coord.AltAz(location=loc, obstime=now)
        sun = coord.get_sun(now)
        sza = 90 - sun.transform_to(altaz).alt.deg
        sza = xr.DataArray(sza, coords=(ds.mjd,), dims=('mjd',), name='sza')
        
        ds = ds.update({'sza':sza, 'tan_lat': tan_lat, 'tan_lon':tan_lon, 'tan_alt':tan_alt})
        ds.to_netcdf(path+'ver_o3_sza_{}.nc'.format(orbit))
 
    return 

def f(orbit):
    try:
        add_loc_to_ds(orbit)
        print(orbit)
    except LookupError:
        print('no ir data for orbit ', orbit)
        pass
    return

if __name__ == '__main__':    
    with Pool(processes=6) as pool:
        pool.map(f, range(39588, 41308, 20))    
      
 
    

