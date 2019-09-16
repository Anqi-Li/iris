#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 14:16:33 2019

@author: anqil
"""

from osirisl1services.readlevel1 import open_level1_ir
from osirisl1services.services import Level1Services
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
from netCDF4 import num2date
units = 'days since 1858-11-17 00:00:00.000'

#%% load data
channel = 3
#orbit = 20900
orbit = 22643


ir = open_level1_ir(orbit, channel, valid=False)
tan_alt = ir.l1.altitude.sel(pixel=slice(14, 128))
tan_lat = ir.l1.latitude.sel(pixel=slice(14, 128))
tan_lon = ir.l1.longitude.sel(pixel=slice(14, 128))
sc_look = ir.l1.look_ecef.sel(pixel=slice(14, 128))
sc_pos = ir.l1.position_ecef
l1 = ir.data.sel(pixel=slice(14, 128))
mjd = ir.mjd.data
pixel = ir.pixel.sel(pixel=slice(14, 128)).data

# calculate sza
import astropy.coordinates as coord
from astropy.time import Time
import astropy.units as u

loc = coord.EarthLocation(lon=tan_lon.sel(pixel=60) * u.deg,
                          lat=tan_lat.sel(pixel=60) * u.deg)
now = Time(ir.mjd, format='mjd', scale='utc')
altaz = coord.AltAz(location=loc, obstime=now)
sun = coord.get_sun(now)
sza = 90 - sun.transform_to(altaz).alt.deg
sza = xr.DataArray(sza, coords=(mjd,), dims=('mjd',), name='sza')

#%% interpolate on altitude grid
top = 150e3
bot = 20e3
#l1 = l1.where(tan_alt<top).where(tan_alt>bot)
alts_interp = np.arange(bot, top, .25e3)
data_interp = []

for (data, alt) in zip(l1, tan_alt):
    f = interp1d(alt, data, bounds_error=False)
    data_interp.append(f(alts_interp))
data_interp = xr.DataArray(data_interp, coords=(mjd, alts_interp), 
                           dims=('mjd', 'alt'))

#%%====plotting
#im = 2500
#plt.rcParams.update({'font.size': 14})
#for im in range(1):#len(mjd)):#range(193, len(mjd)):
#    print(im, ' out of ', len(mjd))
#    limbrange = [1e10, 1e13]
#    fig, ax = plt.subplots(ncols=1, nrows=2, sharey=True, figsize=(15,10))
#    plt.rcParams.update({'font.size': 15})
#    data_interp.plot(x='mjd', y='alt', 
#                     norm=LogNorm(), 
#                     vmin=limbrange[0], vmax=limbrange[1], 
#                     ax=ax[0])
#    
#    ax[0].set(title='From {} \n to {}, \n channel {}'.format(num2date(mjd[0],units),
#              num2date(mjd[-1], units), channel),
#                ylim=(40e3, 120e3))
#    ax[0].axvline(x=mjd[im], color='r', linewidth=2)
#    
#    
#    data_interp.isel(mjd=im).plot.line(y='alt', ax=ax[1], marker='*', ls=' ')
#    #plt.ylim([50e3, 100e3])
#    ax[1].legend([])
#    ax[1].set(#ylabel=' ',
#              title=num2date(mjd[im], units),
#              xlim=[limbrange[0], limbrange[1]*10])
#    ax[1].set_xscale('log')
#    
#    plt.tight_layout()
    
#    path = '/home/anqil/Documents/osiris_database/plots_for_presentations/Limb_workshop2019/Limb_radiance/'
#    filename = 'limb_radiance_{}_{}.png'.format(orbit, im)
#    plt.savefig(path+filename)
#    plt.close(fig)

#plt.figure(figsize=(12.5,2))
#sza.plot(x='mjd')
#plt.axhline(y=90)
#plt.autoscale(enable=True, axis='x', tight=True)
#plt.show()
    
#%%
plt.rcParams.update({'font.size': 15})
limbrange = [1e10, 1e13]
for im in range(len(mjd)):#range(193, len(mjd)):
    print(im, ' out of ', len(mjd))
    fig, ax = plt.subplots(ncols=1, nrows=2, sharey=True, sharex=False, figsize=(15,10))
    ax[0].pcolormesh(mjd, alts_interp, 
                     data_interp.T, 
                     norm=LogNorm(), vmin=limbrange[0], vmax=limbrange[1])

#    ax[0].set_xticklabels([21, 66, 58, 5, -49, -75, -24, 21])
    ax[0].set_xticklabels(np.round(tan_lat.isel(mjd=[0,366,800,1235,1669,2104,2539,2896],pixel=60).data))
    ax[0].set_yticklabels([0, 20, 40, 60, 80, 100, 120, 140, 160])
    ax[0].set(title='From {} \n to {}, \n channel {}'.format(num2date(mjd[0],units),
              num2date(mjd[-1], units), channel),
              xlabel='Latitude / degree N',
              ylabel='Altitude / km')
#                ylim=(40e3, 120e3))
    ax[0].axvline(x=mjd[im], color='r', linewidth=2)
    
    
    data_interp.isel(mjd=im).plot.line(y='alt', ax=ax[1], marker='*', ls='-')
    #plt.ylim([50e3, 100e3])
    ax[1].legend([])
    ax[1].set(ylabel='Altitude / km',
              title=num2date(mjd[im], units),
              xlim=[limbrange[0], limbrange[1]*10],
              xlabel='Limb radiance / photons cm-2 s-1 sr-1')
    ax[1].set_xscale('log')
    plt.tight_layout()
    
#    path = '/home/anqil/Documents/osiris_database/plots_for_presentations/Limb_workshop2019/Limb_radiance/{}/'.format(orbit)
#    filename = 'limb_radiance_{}_{}.png'.format(orbit, im)
#    plt.savefig(path+filename, bbox_inches='tight')
#    plt.close(fig)