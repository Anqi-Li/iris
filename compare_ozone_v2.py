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
clima = clima.update({'m':clima.o + clima.o2 + clima.n2}) #m-3
o3_clima = clima.o3_vmr * clima.m #m-3

#%% load smr
path = '/home/anqil/Documents/osiris_database/ex_data/'
file = 'smr_2008_sza_interp.nc'
smr = xr.open_dataset(path+file)
smr = smr.where((smr.mjd>mjd[0]) & (smr.mjd<mjd[-1]), drop=True)

#%% clip iris data for further processing
#====drop data below and above some altitudes
top = 100e3
bot = 60e3
l1 = l1.where((altitude<top) & (altitude>bot))
#====retireval grid
z = np.arange(bot, top, 1e3) # m
z_top = 110e3
day_mjd_lst = mjd[sza<90]

#%% 1D inversion and retrieve ozone
import sys
sys.path.append('/home/anqil/Documents/Python/iris/old_files')
#from chemi import cal_o2delta
from scipy.optimize import least_squares
from oem_functions import linear_oem
from geometry_functions import pathl1d_iris
#from chemi import gfactor
from o2delta_model import cal_o2delta, gA
#def residual(o3, T, m, z, zenithangle, gA, o2delta_meas):
#    o2delta_model = cal_o2delta(o3, T, m, z, zenithangle, gA)    
#    return o2delta_meas - o2delta_model

A_o2delta = 2.23e-4
fr = 0.5 # filter fraction 
normalize = np.pi*4 / fr
from multiprocessing import Pool
def f(i):
    try:
        print(i, 'out of', len(day_mjd_lst))
    
#        gA = gA_table.interp(z=z, 
#                             month=num2date(day_mjd_lst[i], units).month,
#                             lat=0, #temp
#                             sza=sza.sel(mjd=day_mjd_lst[i], method='nearest'),
#                             ).values # temp
        o3_a = o3_clima.interp(z=z*1e-3, 
                               month=num2date(day_mjd_lst[i], units).month,
                               lat=latitude.sel(pixel=60, mjd=day_mjd_lst[i], method='nearest'),
                               lst_bins=lst.sel(mjd=day_mjd_lst[i], method='nearest'),
                               )
                               
        T_smr = smr.temperature.sel(mjd=day_mjd_lst[i], method='nearest'
                                    ).assign_coords(z=smr.altitude.sel(mjd=day_mjd_lst[i], method='nearest')
                                                    ).interp(z=z)
        m_smr = smr.density.sel(mjd=day_mjd_lst[i], method='nearest'
                                ).assign_coords(z=smr.altitude.sel(mjd=day_mjd_lst[i], method='nearest')
                                                ).interp(z=z) * 1e6 #cm-3 --> m-3
        p_smr = smr.pressure.sel(mjd=day_mjd_lst[i], method='nearest'
                                ).assign_coords(z=smr.altitude.sel(mjd=day_mjd_lst[i], method='nearest')
                                                ).interp(z=z)
        xa = cal_o2delta(o3_a, T_smr, m_smr, z, 
                         sza.sel(mjd=day_mjd_lst[i], method='nearest'),
                         gA(p_smr, sza[i])) * A_o2delta
#        xa = cal_o2delta(o3_a, T_smr, m_smr, z, 
#                         sza.sel(mjd=day_mjd_lst[i], method='nearest').item()) * A_o2delta
        Sa = np.diag(xa**2)
        h = altitude.sel(mjd=day_mjd_lst[i]).where(l1.sel(mjd=day_mjd_lst[i]).notnull(), drop=True)
        K = pathl1d_iris(h, z, z_top) * 1e2 # m-->cm    
        y = l1.sel(mjd=day_mjd_lst[i]).where(l1.sel(mjd=day_mjd_lst[i]).notnull(),drop=True).data *normalize
        Se = np.diag(np.ones(len(y))) *(1e11*normalize)**2 #temp
    
        x, A, Ss, Sm = linear_oem(K, Se, Sa, y, xa)

    #    #lsq fit to get ozone
    #    o2delta_meas = x / A_o2delta # cm-3?
    #    res_lsq = least_squares(residual, o3_a, bounds=(-np.inf, np.inf), verbose=0, 
    #                            args=(T_smr, m_smr, z, sza.sel(mjd=day_mjd_lst[i], method='nearest').item(), 
    #                                  gA, o2delta_meas))
    #    o3_iris = res_lsq.x
        mr = A.sum(axis=1)
    except:
        pass

    return x, xa, np.diag(Sm), mr, o3_a, o3_a

with Pool(processes=4) as pool:
    result = np.array(pool.map(f, range(len(day_mjd_lst[:4])))) 

# organize resulting arrays and save in nc file
result_1d = xr.DataArray(result[:,0,:], 
                         coords=(day_mjd_lst[:4], z), 
                         dims=('mjd', 'z'), name='VER', attrs={'units': 'photons cm-3 s-1'})
ds = xr.Dataset({'ver': result_1d, 
                 'ver_apriori': (['mjd', 'z'], result[:,1,:], {'units': 'photons cm-3 s-1'}),
                 'ver_error':(['mjd', 'z'], result[:,2,:], {'units': '(photons cm-3 s-1)**2'}), 
                 'mr':(['mjd', 'z'], result[:,3,:]), 
                 'o3_iris':(['mjd', 'z'], result[:,4,:], {'units': 'molecule cm-3'}),
                 'o3_xa': (['mjd', 'z'], result[:,5,:], {'units': 'molecule cm-3'})})
#path = '/home/anqil/Documents/osiris_database/iris_ver_o3/'
#ds.to_netcdf(path+'ver_o3_{}.nc'.format(orbit))