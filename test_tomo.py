#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 14:40:54 2019

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
db = sql.connect('/home/anqil/Documents/Python/iris/OSIRIS_three_orbits_test.db')
cur = db.cursor()

orbit = 20900
num_of_orbits = 4
ch = 2
return_column = ('data, mjd, look_ecef, sc_position_ecef, latitude, longitude, altitude')
                 
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
tan_lat_blob = np.array(all_image)[:,4]
tan_lon_blob = np.array(all_image)[:,5]
tan_alt_blob = np.array(all_image)[:,6]

#====unfolding blobs
l1 = np.empty((len(all_image),128))
sc_look = np.empty((len(all_image), 128, 3))
sc_pos = np.empty((len(all_image), 3))
tan_lat = np.empty((len(all_image),128))
tan_lon = np.empty((len(all_image),128))
tan_alt = np.empty((len(all_image),128))
for i in range(len(all_image)):
    l1[i,:] = np.frombuffer(l1_blob[i])
    sc_look[i,:,:] = np.frombuffer(sc_look_blob[i]).reshape(128,3)
    sc_pos[i,:] = np.frombuffer(sc_pos_blob[i])
    tan_lat[i,:] = np.frombuffer(tan_lat_blob[i])
    tan_lon[i,:] = np.frombuffer(tan_lon_blob[i])
    tan_alt[i,:] = np.frombuffer(tan_alt_blob[i])
#====construct xarray data array
pixel = np.arange(128)
l1 = xr.DataArray(l1, coords=(date, pixel), 
                  dims=('date', 'pixel'), 
                  attrs={'units':'Rayleigh??'})
sc_look = xr.DataArray(sc_look, coords=(date, pixel, ['x', 'y', 'z']), 
                       dims=('date', 'pixel', 'xyz'))
sc_pos = xr.DataArray(sc_pos, coords=(date, ['x', 'y', 'z']), dims=('date', 'xyz'))
tan_lat = xr.DataArray(tan_lat, coords=(date, pixel),
                       dims=('date', 'pixel'), attrs={'units':'degree'})
tan_lon = xr.DataArray(tan_lon, coords=(date, pixel),
                       dims=('date', 'pixel'), attrs={'units':'degree'})
tan_alt = xr.DataArray(tan_alt, coords=(date, pixel),
                       dims=('date', 'pixel'), attrs={'units':'meter'})


#%% building jacobian matrix K for several images
#====define the bin edges (atmosphere grid)
edges_lat = np.linspace(53, 66, 300)
edges_lon = np.linspace(150, 157, 300)
edges_alt = np.arange(50e3, 110e3, 0.5e3)
edges = edges_lat, edges_lon, edges_alt
#====num of columns of jacobian
col_len = (len(edges_lat)+1)*(len(edges_lon)+1)*(len(edges_alt)+1) 

from geometry_functions import los_points_fix_dl, ecef2lla
from oem_functions import jacobian_row

dl = 3e3 #fixed distance between all points
nop = 500 # choose number of points along the line
K_row_idx = []
K_col_idx = []
K_value = []
dll = dl * np.ones(nop) #temp
im_start, im_end = 299, 303
pix_start, pix_end = 22, 128
measurement_id = 0
for image in range(im_start, im_end):
    #====generate points of los for all pixels in each image
    #====all points in ecef coordinate xyz
    lx, ly, lz = los_points_fix_dl(sc_look[image], sc_pos[image], dl=dl, nop=nop)    
    #====convert xyz to lat lon alt for all points
    los_lat, los_lon, los_alt = ecef2lla(lx, ly, lz)
    
    #====build K
    for pix in range(pix_start, pix_end):   
        los = los_lat.sel(pixel=pix), los_lon.sel(pixel=pix), los_alt.sel(pixel=pix)
        measurement_idx, grid_idx, pathlength = jacobian_row(dll, edges, los, measurement_id)
        K_row_idx.append(measurement_idx)
        K_col_idx.append(grid_idx)
        K_value.append(pathlength)
        measurement_id += 1
        
K_row_idx = np.concatenate(K_row_idx).astype('int')
K_col_idx = np.concatenate(K_col_idx).astype('int')
K_value = np.concatenate(K_value) # in meter

#==== create sparse matrix
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import inv
K_coo = coo_matrix((K_value, (K_row_idx, K_col_idx)))
print(K_coo)

#%% inversion
from oem_functions import linear_oem_sp
import scipy.sparse as sp

xa = np.ones(col_len) * 0 # temp
Sa = sp.diags([1], shape=(col_len, col_len)) *1e-9 #temp
Se = np.diag(np.ones(len(pixel))) * 30 #temporary

#%% unravel linear indices 
from oem_functions import unfold_measure_id, unfold_grid_id
#====test
measurement_id = 200
grid_id = 340
image, pix = unfold_measure_id(measurement_id, im_end, im_start, pix_end, pix_start)
max_lat, max_lon, max_alt = unfold_grid_id(grid_id, edges_lat, edges_lon, edges_alt)













#%% generate points along los
#==== method 2: points based on scale factor of distance between tangent point to sc
from geometry_functions import los_points, ecef2lla
#image = 299
#nop = 500
#====all points in ecef coordinate xyz
lx, ly, lz, dl = los_points(sc_pos[image], tan_lat[image], tan_lon[image], tan_alt[image], nop=nop)
#====convert xyz to lon lat alt for all points
los_lat, los_lon, los_alt = ecef2lla(lx, ly, lz)

dl = dl[:, 0]

#%% generate points along los for 1 image
#==== method 1: equal distance points along each los
from geometry_functions import los_points_fix_dl, ecef2lla

dl = 3e3 #fixed distance between all points
nop = 500 # choose number of points along the line
image = 299
#====all points in ecef coordinate xyz
lx, ly, lz = los_points_fix_dl(sc_look[image], sc_pos[image], dl=dl, nop=nop)    
#====convert xyz to lat lon alt for all points
los_lat, los_lon, los_alt = ecef2lla(lx, ly, lz)

dl = dl * np.ones(nop) #temp

#%% building jacobian matrix K for 1 image
#====define the bin edges (atmosphere grid)
edges_lat = np.linspace(los_lat.min(), los_lat.max(), 300)
edges_lon = np.linspace(los_lon.min(), los_lon.max(), 300)
edges_alt = np.arange(los_alt.min(), los_alt.max(), 0.5e3)
edges = edges_lat, edges_lon, edges_alt
#====num of columns of jacobian
column_len = (len(edges_lat)+1)*(len(edges_lon)+1)*(len(edges_alt)+1) 

from oem_functions import jacobian_row
K_row_idx = []
K_column_idx = []
K_value = []
pix_start, pix_end = 22, 128
for pix in range(pix_start, pix_end):
    pix_idx = pix #temp     
    los = los_lat.sel(pixel=pix), los_lon.sel(pixel=pix), los_alt.sel(pixel=pix)
    measurement_idx, grid_idx, pathlength = jacobian_row(dl, edges, los, pix_idx)
    K_row_idx.append(measurement_idx)
    K_column_idx.append(grid_idx)
    K_value.append(pathlength)

K_row_idx = np.concatenate(K_row_idx)
K_column_idx = np.concatenate(K_column_idx)
K_value = np.concatenate(K_value)

#%% plot los in 3d to check
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
mpl.rcParams['legend.fontsize'] = 10
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#for pix in np.arange(22, 106, 20):
#    ax.plot(lx.sel(pixel=pix)*1e-3, 
#            ly.sel(pixel=pix)*1e-3,
#            lz.sel(pixel=pix)*1e-3)
#    ax.set(xlabel='x',
#           ylabel='y',
#           zlabel='z')
        
fig = plt.figure()
ax = fig.gca(projection='3d')
for pix in np.arange(22, 106, 20):
    ax.plot(los_lon.sel(pixel=pix), 
            los_lat.sel(pixel=pix),
            los_alt.sel(pixel=pix))
ax.set(xlabel='lon',
       ylabel='lat',
       zlabel='alt')


