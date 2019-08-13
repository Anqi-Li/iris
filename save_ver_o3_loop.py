#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 15:56:08 2019

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

def load_smr(mjd):
    start = num2date(mjd[0]-1/24/60*5, units)
    end = num2date(mjd[-1]-1/24/60*4, units)
#    start_year = start.year
#    start_month = start.month
#    start_day = start.day
#    start_hour = start.hour
#    start_minute = start.minute
#    end_year = end.year
#    end_month = end.month
#    end_day = end.day
#    end_hour = end.hour
#    end_minute = end.minute
    
    start_date = '{}-{}-{}%20{}%3A{}%3A00'.format(start.year, start.month, 
                  start.day, start.hour, start.minute)
    end_date = '{}-{}-{}%20{}%3A{}%3A59'.format(end.year, end.month, end.day, 
                end.hour, end.minute)
    
    dataset = 'ALL'
    fm = 2
    product = "O3 / 545 GHz / 20 to 85 km"
    baseurl = "http://odin.rss.chalmers.se/rest_api/v5/level2/development/"
    scansurl = baseurl+"{0}/{1}/scans/?limit=1000&offset=0&"
    scansurl += "start_time={2}&end_time={3}"
    a = requests.get(scansurl.format(dataset,fm,start_date,end_date))
    aaa = json.loads(a.text)
    
    scanno_lst = np.zeros(len(aaa['Data']))
    o3_vmr_a = np.zeros((len(aaa['Data']), 51))
    o3_vmr = np.zeros((len(aaa['Data']), 51))
    z_smr = np.zeros((len(aaa['Data']), 51))
    mjd_smr = np.zeros((len(aaa['Data'])))
    p_smr = np.zeros((len(aaa['Data']), 51))
    T_smr = np.zeros((len(aaa['Data']), 51))
    mr_smr = np.zeros((len(aaa['Data']), 51))
    error_vmr = np.zeros((len(aaa['Data']), 51))
    
    for i in range(len(aaa['Data'])):
        scanno_lst[i] = aaa['Data'][i]['ScanID']
        scansurl = aaa['Data'][i]['URLS']['URL-level2'] + 'L2/?product={}'.format(product)
        a = requests.get(scansurl)
        result = json.loads(a.text)['Data'][0]
        
        o3_vmr_a[i,:] = np.array(result['Apriori'])
        o3_vmr[i,:] = np.array(result['VMR'])
        z_smr[i,:] = np.array(result['Altitude'])
        mjd_smr[i] = result['MJD']
        p_smr[i,:] = np.array(result['Pressure'])
        T_smr[i,:] = np.array(result['Temperature'])
        mr_smr[i,:] = np.array(result['AVK']).sum(axis=1) #measurement response
        error_vmr[i,:] = np.array(result['ErrorTotal'])
    
    Av = 6.023e23 #Avogadro's number: molec/mol
    R = 8.31 # gas constant: J/mol/K
    m = Av * p_smr / (R * T_smr) * 1e-6 # number density of air cm-3
    o3_smr = m * o3_vmr  # cm-3
    o3_smr_a = m * o3_vmr_a # cm-3
    error_smr = m * error_vmr #cm-3
    
    o3_vmr = xr.DataArray(o3_vmr, coords=(mjd_smr, np.arange(51)),
                          dims=('mjd', 'z'))
    smr = xr.Dataset({'o3_vmr': o3_vmr,
                     'o3_vmr_a':(['mjd', 'z'], o3_vmr_a,{'units': ' '}),
                     'z':(['mjd', 'z'], z_smr, {'units': 'm'}),
                     'p':(['mjd', 'z'], p_smr, {'units': 'Pa'}),
                     'T':(['mjd', 'z'], T_smr, {'units': 'K'}),
                     'mr':(['mjd', 'z'], mr_smr, {'units': ' '}),
                     'error_vmr':(['mjd', 'z'], error_vmr, {'units': ' '}),
                     'o3':(['mjd', 'z'], o3_smr, {'units': 'cm-3'}),
                     'o3_a':(['mjd', 'z'], o3_smr_a, {'units': 'cm-3'}),
                     'error':(['mjd', 'z'], error_smr, {'units': 'cm-3'}),
                     'm':(['mjd', 'z'], m, {'units': 'cm-3'})})
    return smr

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
    
    #%% load smr whole orbit
    smr = load_smr(mjd)
    
    #%% clip iris data
    #====drop data below and above some altitudes
    top = 100e3
    bot = 60e3
    l1 = l1.where(tan_alt<top).where(tan_alt>bot)
    day_mjd_lst = mjd[sza<90]

    #====retireval grid
    z = np.arange(bot, top, 1e3) # m
    z_top = z[-1] + 2e3
    
    #%% 1D inversion and retrieve ozone
    gA_table = np.load('gA_table.npz')['gA']
    z_table = np.load('gA_table.npz')['z']
    sza_table = np.load('gA_table.npz')['sza']
    month_table = np.load('gA_table.npz')['month'] #look up gA table instead of calculate at each mjd
    
    fr = 0.5 # filter fraction 
    normalize = np.pi*4 / fr
    
    def f(i):
        print(i, 'out of mjd_lst ', len(day_mjd_lst))
        #match the closest scan of smr 
        closest_scan_idx = (np.abs(smr.mjd - day_mjd_lst[i])).argmin()
        o3_SMR_a = interp1d(smr.z[closest_scan_idx,:], smr.o3_a[closest_scan_idx,:],
                           fill_value="extrapolate")(z)
        T_SMR = interp1d(smr.z[closest_scan_idx,:], smr.T[closest_scan_idx,:],
                         fill_value="extrapolate")(z)
        m_SMR = interp1d(smr.z[closest_scan_idx,:], smr.m[closest_scan_idx,:],
                         fill_value="extrapolate")(z)
    #        gA = gfactor(0.21*m_SMR, T_SMR, z, sza.sel(mjd=day_mjd_lst[i]).item())
        month = num2date(mjd[0], units).month
        gA = interp1d(z_table, 
                      gA_table[:,(np.abs(month_table - month)).argmin(), 0,
                               (np.abs(sza_table - sza.sel(mjd=day_mjd_lst[i]).item())).argmin()])(z)
        
        xa = cal_o2delta(o3_SMR_a, T_SMR, m_SMR, z, sza.sel(mjd=day_mjd_lst[i]).item(), gA) * A_o2delta
        Sa = np.diag(xa**2)
        h = tan_alt.sel(mjd=day_mjd_lst[i], pixel=pixel[l1.notnull().sel(mjd=day_mjd_lst[i])])
        K = pathl1d_iris(h, z, z_top) * 1e2 # m-->cm    
        y = l1.sel(mjd=day_mjd_lst[i], pixel=pixel[l1.notnull().sel(mjd=day_mjd_lst[i])]).data *normalize
        Se = np.diag(np.ones(len(y))) *(1e11*normalize)**2
    
        x, A, Ss, Sm = linear_oem(K, Se, Sa, y, xa)
    
        #lsq fit to get ozone
        o2delta_meas = x / A_o2delta # cm-3?
        res_lsq = least_squares(residual, o3_SMR_a, bounds=(-np.inf, np.inf), verbose=0, 
                                args=(T_SMR, m_SMR, z, sza.sel(mjd=day_mjd_lst[i]).item(), gA, o2delta_meas))

        return x, xa, np.diag(Sm), A.sum(axis=1), res_lsq.x, o3_SMR_a
    
    with Pool(processes=4) as pool:
        result = np.array(pool.map(f, range(len(day_mjd_lst)))) #len(day_mjd_lst)
    
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
    return ds
    
if __name__ == '__main__':
    for orbit in range(37587, 37587+50, 20):
        ds = main(orbit)
        path = '/home/anqil/Documents/osiris_database/iris_ver_o3/'
        ds.to_netcdf(path+'ver_o3_{}.nc'.format(orbit))