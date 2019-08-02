#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 14:23:52 2019

@author: anqil
"""
from osirisl1services.pyhdfadapter import HDF4VS
from osirisl1services.readlevel1 import open_level1_ir
from osirisl1services.services import Level1Services
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
from netCDF4 import num2date
units = 'days since 1858-11-17 00:00:00.000'

plt.ioff()
top = 120e3
bot = 40e3



def plot_iris_limb(orbit, channel, figure_dir):
    print(orbit)
    ir = open_level1_ir(orbit, channel, valid=False)
    tan_alt = ir.l1.altitude.sel(pixel=slice(14, 128))
    l1 = ir.data.sel(pixel=slice(14, 128))
    mjd = ir.mjd.data
    
    #l1 = l1.where(tan_alt<top).where(tan_alt>bot)
    alts_interp = np.arange(bot, top, .25e3)
    data_interp = []
    
    for (data, alt) in zip(l1, tan_alt):
        f = interp1d(alt, data, bounds_error=False)
        data_interp.append(f(alts_interp))
    data_interp = xr.DataArray(data_interp, coords=(mjd, alts_interp), 
                               dims=('mjd', 'alt'))
    #==== plot the full orbit
#    data_interp.plot(x='mjd', y='alt', 
#                     norm=LogNorm(), 
#                     vmin=1e9, vmax=1e13, 
#                     size=5, aspect=3)
#    fig = plt.figure()
#    ax = plt.gca()
#    ax.set(title='From {} \n to {}, \n channel {}'.format(num2date(mjd[0],units),
#           num2date(mjd[-1], units), channel),
#          xlabel='image index')
#
#    ax.set_xticks(mjd[np.arange(0,len(mjd),300, dtype=int)])
#    ax.set_xticklabels(np.arange(0,len(mjd),300))
#    fig.savefig(figure_dir+'limb_{}.png'.format(orbit), bbox_inches = "tight") #fig file name
#    #plt.close(fig)
#    fig.clf()
    del ir, tan_alt, l1, data_interp
    print(HDF4VS.cache_info())
    return

import multiprocessing as mp
from multiprocessing import Pool
import gc  

figure_dir = '/home/anqil/Documents/osiris_database/limb_radiance_plots/2008/'
start_orbit = 39312
end_orbit = 42900
channel = 3


#for orbit in range(start_orbit, end_orbit):
#    try:
#        plot_iris_limb(orbit, channel, figure_dir)
#
#        gc.collect()
#    except:
#        pass
        
def f(x):
    plot_iris_limb(x, channel, figure_dir)
            
with Pool(processes=4) as pool:
    result = pool.map(f, range(start_orbit, end_orbit))