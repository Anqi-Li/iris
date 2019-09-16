#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 11:00:19 2019

@author: anqil
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
from osirisl1services.readlevel1 import open_level1_ir
from osirisl1services.services import Level1Services
import xarray as xr
#%% load orbit data
channel = 3
#orbit = 20900
orbit = 22643

ir = open_level1_ir(orbit, channel, valid=False)
#tan_alt = ir.l1.altitude.sel(pixel=slice(14, 128))
tan_lat = ir.l1.latitude.sel(pixel=slice(14, 128))
#tan_lon = ir.l1.longitude.sel(pixel=slice(14, 128))

#%% load ver1d data
nc_filename = '{}_propermodel_pi.nc'.format(orbit)
ver = xr.open_dataset(nc_filename).ver
z = xr.open_dataset(nc_filename).z
day_mjd_lst = xr.open_dataset(nc_filename).mjd 
o3_iris = xr.open_dataset(nc_filename).o3_iris
mr = xr.open_dataset(nc_filename).mr
mr_threshold = 0.9

plt.rcParams.update({'font.size': 15})
for im in range(len(day_mjd_lst)):
    print('VER ', im, '/', 1452)
    fig, ax = plt.subplots(ncols=1, nrows=2, sharey=True, figsize=(15,10))
    ax[0].pcolormesh(ver.T, norm=LogNorm(), vmin=1e5, vmax=1e7, cmap='Spectral')
    #plt.colorbar()
    ax[0].set(xlabel='Latitude / degree N',
              ylabel='Altitude / km')
    
    ax[0].set_xticklabels(np.round(tan_lat.isel(mjd=ax[0].get_xticks().astype(int),pixel=60).data))
    ax[0].set_yticklabels([0,60,70,80,90,100,110])
    ax[0].axvline(x=im, color='r', linewidth=2)
    
    ax[1].plot(ver[im], np.arange(len(ver.z)), 'o', ms=8)
    ax[1].set_xscale('log')
    ax[1].set(xlim=[1e5, 3e7],
              ylabel='Altitude / km',
              xlabel='Volumn emission rate / photons cm-3 s-1')
    
    path = '/home/anqil/Documents/osiris_database/plots_for_presentations/Limb_workshop2019/VER/{}/'.format(orbit)
    filename = '1D_VER_pi_{}_{}.png'.format(orbit, im)
    plt.savefig(path+filename, bbox_inches='tight')
    plt.close(fig)
    

#%% load smr whole orbit
import requests 
import numpy as np
import json
from netCDF4 import num2date
units = 'days since 1858-11-17 00:00:00.000'
start = num2date(ir.mjd[0]-1/24/60*5, units)
end = num2date(ir.mjd[-1]-1/24/60*4, units)
start_year = start.year
start_month = start.month
start_day = start.day
start_hour = start.hour
start_minute = start.minute
end_year = end.year
end_month = end.month
end_day = end.day
end_hour = end.hour
end_minute = end.minute

start_date = '{}-{}-{}%20{}%3A{}%3A00'.format(start_year, start_month, 
              start_day, start_hour, start_minute)
end_date = '{}-{}-{}%20{}%3A{}%3A59'.format(end_year, end_month, end_day, 
            end_hour, end_minute)

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
    mr_smr[i,:] = np.array(result['AVK']).sum(axis=1)
    error_vmr[i,:] = np.array(result['ErrorTotal'])

Av = 6.023e23 #Avogadro's number: molec/mol
R = 8.31 # gas constant: J/mol/K
m = Av * p_smr / (R * T_smr) * 1e-6 # number density of air cm-3
o3_smr = m * o3_vmr  # cm-3
o3_smr_a = m * o3_vmr_a # cm-3
error_smr = m * error_vmr #cm-3

#%%
#nc_filename = '{}_pi.nc'.format(orbit)


grid = plt.GridSpec(ncols=2 ,nrows=3, hspace=0.5, wspace=0.2)
plt.rcParams.update({'font.size': 14})
for i in range(len(day_mjd_lst)):
    print('ozone ', i, 'out of', len(day_mjd_lst))
    closest_smr_scan_idx = (np.abs(mjd_smr - day_mjd_lst[i].data)).argmin()
    
    fig = plt.figure(figsize=(15,10))
    ax0 = fig.add_subplot(grid[0,:])
    ax1 = fig.add_subplot(grid[1,:],  sharex=ax0)
    ax2 = fig.add_subplot(grid[2,0])
    ax3 = fig.add_subplot(grid[2,1],  sharey=ax2)

    ax0.pcolormesh(np.tile(mjd_smr,(len(z_smr.T),1)), z_smr.T*1e-3, 
                   np.ma.masked_where(mr_smr.T<0.9,o3_smr.T), 
               norm=LogNorm(vmin=1e6, vmax=1e10), cmap='inferno')
    ax0.axvline(x=mjd_smr[closest_smr_scan_idx], color='r')
#    CS = ax0.contour(mjd_smr, z_smr[0,:]*1e-3, mr_smr.T, levels=[0.8, 1.2])
    #ax[0].clabel(CS, inline=1, fontsize=10)
    ax0.set(ylim=[60, 110], 
              title='SMR',
              ylabel='Altitude / km')
    ax0.set_xticklabels(np.round(tan_lat.isel(mjd=[0,366,800,1235,1669,2104,2539,2896],pixel=60).data))
    
    im = ax1.pcolormesh(day_mjd_lst[:-2], z*1e-3, np.ma.masked_where(mr[:-2].T<0.9,o3_iris[:-2].T),
                    norm=LogNorm(vmin=1e6, vmax=1e10), cmap='inferno')
    ax1.axvline(x=day_mjd_lst[i], color='r')
    ax1.set(title='IRIS',
            ylabel='Altitude / km',
            xlabel='Latitude / degree N')
    fig.colorbar(im, ax=[ax0,ax1,ax2,ax3], label='Ozone number density /cm-3')
    
    ax2.semilogx(o3_smr_a[closest_smr_scan_idx], z_smr[closest_smr_scan_idx]*1e-3, 
                 'k--', label='SMR a priori')
    ax2.semilogx(o3_iris[i,mr[i]>mr_threshold], z[mr[i]>mr_threshold]*1e-3, '*', 
                  label='IRIS (mr>{})'.format(mr_threshold))
    ax2.semilogx(o3_smr[closest_smr_scan_idx,mr_smr[closest_smr_scan_idx]>mr_threshold], 
                  z_smr[closest_smr_scan_idx,mr_smr[closest_smr_scan_idx]>mr_threshold]*1e-3, '*',
                  label='SMR (mr>{})'.format(mr_threshold))
    ax2.set_xlim(left=1e4, right=1e12)
    ax2.set(title='Number density',
              xlabel='cm-3',
              ylabel='Altitude / km')
    ax2.legend(loc='lower left')
    
    m_SMR = interp1d(z_smr[closest_smr_scan_idx,:], m[closest_smr_scan_idx,:],
                    fill_value="extrapolate")(z[mr[i]>mr_threshold])
    ax3.semilogx(o3_vmr_a[closest_smr_scan_idx]*1e6, z_smr[closest_smr_scan_idx]*1e-3, 'k--')
    ax3.semilogx(o3_iris[i,mr[i]>mr_threshold]/m_SMR*1e6, z[mr[i]>mr_threshold]*1e-3, '*',
                  label='IRIS (mr>{})'.format(mr_threshold))
    ax3.semilogx(o3_vmr[closest_smr_scan_idx,mr_smr[closest_smr_scan_idx]>mr_threshold]*1e6, 
                  z_smr[closest_smr_scan_idx,mr_smr[closest_smr_scan_idx]>mr_threshold]*1e-3, '*',
                  label='SMR (mr>{})'.format(mr_threshold))
    ax3.set_xlim(left=1e-2, right=1e1)
    ax3.set(title='volume mixing ratio',
              xlabel='ppmv')
#    ax3.legend(loc='lower left')
    
    
    path = '/home/anqil/Documents/osiris_database/plots_for_presentations/Limb_workshop2019/Ozone/{}/'.format(orbit)
    filename = 'ozone_pi_{}_{}.png'.format(orbit, i)
    plt.savefig(path+filename)
    plt.close(fig)
#    
    
