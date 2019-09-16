#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 17:48:02 2019

@author: anqil
"""

import sqlite3 as sql
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
from netCDF4 import num2date
units = 'days since 1858-11-17 00:00:00.000'

#%% open database file
db = sql.connect('OSIRIS.db')
cur = db.cursor()

orbit = 20900
num_of_orbits = 4
ch = 2
return_column = ('data,' + 'mjd,' + 'look_ecef,' + 'sc_position_ecef')
                 
select_str = 'SELECT {} FROM IRI JOIN channel{} ON IRI.stw = channel{}.stw WHERE orbit>={} AND orbit<={}'
result = cur.execute(select_str.format(return_column, ch, ch, orbit, orbit+num_of_orbits))
all_image = result.fetchall()
db.close()
if len(all_image) == 0:
    print('No data for this orbit(s)')
    

print('num of image: {}'.format(len(all_image)))

#%%
l1_blob = np.array(all_image)[:,0]
mjd = np.array(all_image)[:,1].astype(float)
date = num2date(mjd, units)
sc_look_blob = np.array(all_image)[:,2]
sc_pos_blob = np.array(all_image)[:,3]

#unfolding blobs
l1 = np.empty((len(all_image),128))
sc_look = np.empty((len(all_image), 128, 3))
sc_pos = np.empty((len(all_image), 3))
for i in range(len(all_image)):
    l1[i,:] = np.frombuffer(l1_blob[i])
    sc_look[i,:,:] = np.frombuffer(sc_look_blob[i]).reshape(128,3)
    sc_pos[i,:] = np.frombuffer(sc_pos_blob[i])

# construct xarray dataset
pixel = np.arange(22,128)
l1 = xr.DataArray(l1[:, 22:], coords=(date, pixel), 
                  dims=('date', 'pixel'), 
                  attrs={'units':'Rayleigh??'})
sc_look = xr.DataArray(sc_look[:,22:,:], coords=(date, pixel, ['x', 'y', 'z']), 
                       dims=('date', 'pixel', 'xyz'))
sc_pos = xr.DataArray(sc_pos, coords=(date, ['x', 'y', 'z']), dims=('date', 'xyz'))

#%% generate points along los
from geometry_functions import los_points_fix_dl, lla2ecef, ecef2lla
image = 299
look = sc_look[image]
pos = sc_pos[image]

dl = 3e3 #fixed distance between all points
nop = 500 # choose number of points along the line
lx, ly, lz = los_points_fix_dl(look, pos, dl=dl, nop=nop)
    
#convert xyz to lon lat alt for all points
los_lat, los_lon, los_alt = ecef2lla(lx, ly, lz)
    
#%% plot los in 3d to check
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure()
ax = fig.gca(projection='3d')
for pix in np.arange(22, 106, 20):
    ax.plot(lx.sel(pixel=pix)*1e-3, 
            ly.sel(pixel=pix)*1e-3,
            lz.sel(pixel=pix)*1e-3)
    ax.set(xlabel='x',
           ylabel='y',
           zlabel='z')
        
fig = plt.figure()
ax = fig.gca(projection='3d')
for pix in np.arange(22, 106, 20):
    ax.plot(los_lon.sel(pixel=pix), 
            los_lat.sel(pixel=pix),
            los_alt.sel(pixel=pix))
ax.set(xlabel='lon',
       ylabel='lat',
       zlabel='alt')

#%% counting points in an atmospheric grid using histogramdd
pix = 2
edges_lat = np.linspace((los_lat.min()-0.5).round(), (los_lat.max()+0.5).round(), 301)
edges_lon = np.linspace((los_lon.min()-0.5).round(), (los_lon.max()+0.5).round(), 301)
edges_alt = np.arange(los_alt.min(), los_alt.max(), 0.5e3)
los = xr.concat([los_lat[:,pix], los_lon[:,pix], los_alt[:,pix]], dim='lla')
#los = xr.concat([los_lat, los_lon, los_alt], dim='lla')

H, edges = np.histogramdd(los.T, bins=(edges_lat[:-1], edges_lon[:-1], edges_alt[:-1]))
lat_idx, lon_idx, alt_idx = H.nonzero() #index of lla bins which contains at least 1 point in it
values = H[H.nonzero()] #number of points
print(values.sum())

#%% building 1 row for jacobian matrix K
dll = dl * np.ones(nop) #temp
#define the bin edges (atmosphere grid)
edges_lat = np.linspace(los_lat.min(), los_lat.max(), 300)
edges_lon = np.linspace(los_lon.min(), los_lon.max(), 300)
edges_alt = np.arange(los_alt.min(), los_alt.max(), 0.5e3)

from oem_functions import jacobian_row

#choose which pixel number / los number you wanna look at
pix = 0 
edges = edges_lat, edges_lon, edges_alt
los = los_lat[:,pix], los_lon[:,pix], los_alt[:,pix]
measurement_idxx, grid_idxx, pathlengthh, column_len = jacobian_row(dll, edges, los, pix)
