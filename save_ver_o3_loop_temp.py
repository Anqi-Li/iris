#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 15:11:14 2019

@author: anqil
"""

import requests 
import numpy as np
import json
import matplotlib.pylab as plt
from chemi import cal_o2delta, cal_o2delta_thomas
from scipy.optimize import least_squares
from oem_functions import linear_oem
from geometry_functions import pathl1d_iris
from chemi import gfactor
from osirisl1services.readlevel1 import open_level1_ir
from osirisl1services.services import Level1Services
import xarray as xr
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
import astropy.coordinates as coord
from astropy.time import Time
import astropy.units as u
from netCDF4 import num2date
from multiprocessing import Pool
units = 'days since 1858-11-17 00:00:00.000'
A_o2delta = 2.23e-4#2.58e-4#2.23e-4 # radiative lifetime of singlet delta to ground state
channel = 3


def residual(o3, T, m, z, zenithangle, gA, o2delta_meas):
    o2delta_model = cal_o2delta(o3, T, m, z, zenithangle, gA)    
    return o2delta_meas - o2delta_model

#%%
def main(orbit):
    ir = open_level1_ir(orbit, channel, valid=False)
    tan_alt = ir.l1.altitude.sel(pixel=slice(14, 128))
    tan_lat = ir.l1.latitude.sel(pixel=slice(14, 128))
    tan_lon = ir.l1.longitude.sel(pixel=slice(14, 128))
#    sc_look = ir.l1.look_ecef.sel(pixel=slice(14, 128))
#    sc_pos = ir.l1.position_ecef
    l1 = ir.data.sel(pixel=slice(14, 128)) #/np.pi
    mjd = ir.mjd.data
    pixel = ir.pixel.sel(pixel=slice(14, 128)).data
    loc = coord.EarthLocation(lon=tan_lon.sel(pixel=60) * u.deg,
                              lat=tan_lat.sel(pixel=60) * u.deg)
    now = Time(ir.mjd, format='mjd', scale='utc')
    altaz = coord.AltAz(location=loc, obstime=now)
    sun = coord.get_sun(now)
    sza = 90 - sun.transform_to(altaz).alt.deg
    sza = xr.DataArray(sza, coords=(mjd,), dims=('mjd',), name='sza')
    
    #% load apriori 
    apriori = xr.open_dataset('apriori_temp.nc')
    
    #% clip iris data
    #====drop data below and above some altitudes
    top = 100e3
    bot = 60e3
    l1 = l1.where(tan_alt<top).where(tan_alt>bot)
    day_mjd_lst = mjd[sza<90]

    #====retireval grid
    z = np.arange(bot, top, 1e3) # m
    z_top = z[-1] + 2e3
    
    #% 1D inversion and retrieve ozone
    gA_table = np.load('gA_table.npz')['gA']
    z_table = np.load('gA_table.npz')['z']
    sza_table = np.load('gA_table.npz')['sza']
    month_table = np.load('gA_table.npz')['month'] #look up gA table instead of calculate at each mjd
    month = num2date(mjd[0], units).month
    
    fr = 0.5 # filter fraction 
    normalize = np.pi*4 / fr
    
    o3_a = interp1d(apriori.altgrid*1e3, apriori.o3_vmr * apriori.m, fill_value="extrapolate")(z)
    T_a = interp1d(apriori.altgrid*1e3, apriori.T, fill_value="extrapolate")(z)
    m_a = interp1d(apriori.altgrid*1e3, apriori.m, fill_value="extrapolate")(z)
    
    result = []
    for i in range(len(day_mjd_lst)):
        print(i, 'in mjd_lst ', len(day_mjd_lst), ', in orbit ', orbit)
        
    #        gA = gfactor(0.21*m_SMR, T_SMR, z, sza.sel(mjd=day_mjd_lst[i]).item())
        gA = interp1d(z_table, 
                  gA_table[:,(np.abs(month_table - month)).argmin(), 0,
                           (np.abs(sza_table - sza.sel(mjd=day_mjd_lst[i]).item())).argmin()])(z)
    
        xa = cal_o2delta(o3_a, T_a, m_a, z, sza.sel(mjd=day_mjd_lst[i]).item(), gA) * A_o2delta
        Sa = np.diag(xa**2)
        h = tan_alt.sel(mjd=day_mjd_lst[i], pixel=pixel[l1.notnull().sel(mjd=day_mjd_lst[i])])
        K = pathl1d_iris(h, z, z_top) * 1e2 # m-->cm    
        y = l1.sel(mjd=day_mjd_lst[i], pixel=pixel[l1.notnull().sel(mjd=day_mjd_lst[i])]).data *normalize
        Se = np.diag(np.ones(len(y))) *(1e11*normalize)**2
    
        x, A, Ss, Sm = linear_oem(K, Se, Sa, y, xa)
    
        #lsq fit to get ozone
        o2delta_meas = x / A_o2delta # cm-3
        res_lsq = least_squares(residual, o3_a, bounds=(-np.inf, np.inf), verbose=0, 
                                args=(T_a, m_a, z, sza.sel(mjd=day_mjd_lst[i]).item(), gA, o2delta_meas))

        result.append((x, xa, np.diag(Sm), A.sum(axis=1), res_lsq.x, o3_a))
    

    result = np.array(result)
    # organize resulting arrays and save in nc file
    result_1d = xr.DataArray(result[:,0,:],  
                             coords=(day_mjd_lst, z), 
                             dims=('mjd', 'z'), name='VER', attrs={'units': 'photons cm-3 s-1'})
    ds = xr.Dataset({'ver': result_1d, 
                     'ver_apriori': (['mjd', 'z'], result[:,1,:], {'units': 'photons cm-3 s-1'}),
                     'ver_error':(['mjd', 'z'], result[:,2,:], {'units': '(photons cm-3 s-1)**2'}), 
                     'mr':(['mjd', 'z'], result[:,3,:]), 
                     'o3_iris':(['mjd', 'z'], result[:,4,:], {'units': 'molecule cm-3'}),
                     'o3_xa': (['mjd', 'z'], result[:,5,:], {'units': 'molecule cm-3'})})
    ds = ds.update({'sza':sza[sza<90], 
                    'tan_lat': tan_lat.sel(mjd=day_mjd_lst), 
                    'tan_lon':tan_lon.sel(mjd=day_mjd_lst), 
                    'tan_alt':tan_alt.sel(mjd=day_mjd_lst)})    
    return ds
    

#%%
def f(orbit):
    try:
        path = '/home/anqil/Documents/osiris_database/iris_ver_o3/'
        ds = main(orbit)
        ds.to_netcdf(path+'ver_o3_sza_{}.nc'.format(orbit))
    except LookupError:
        print('no ir data for orbit ', orbit)
        pass
    return

if __name__ == '__main__':    
    with Pool(processes=4) as pool:
        pool.map(f, range(41348, 41348+2000, 20))    
      
    