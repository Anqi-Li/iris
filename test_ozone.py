#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 10:06:30 2019

@author: anqil
"""


import sqlite3 as sql
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
from netCDF4 import num2date, date2num
units = 'days since 1858-11-17 00:00:00.000'

#%% open database file
db = sql.connect('/home/anqil/Documents/Python/iris_git/OSIRIS_three_orbits_test.db')
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
tan_lon = xr.DataArray(tan_lon-180, coords=(date, pixel),
                       dims=('date', 'pixel'), attrs={'units':'degree'})
tan_alt = xr.DataArray(tan_alt, coords=(date, pixel),
                       dims=('date', 'pixel'), attrs={'units':'meter'})

l1 = l1.fillna(0)
#%% building jacobian matrix K for several images
#====choose mesurements
im_lst = np.arange(2000,2050,2)
pix_lst = np.arange(22,128)
#num of rows of jacobian
row_len = len(im_lst) * len(pix_lst)

#====define the bin edges (atmosphere grid)
edges_lat = np.linspace(tan_lat.isel(date=im_lst[-1], pixel=60), 
                        tan_lat.isel(date=im_lst[0], pixel=60), 11)
edges_lon = np.linspace(tan_lon.isel(date=im_lst[-1], pixel=60),
                        tan_lon.isel(date=im_lst[0], pixel=60), 10)
#edges_lat = np.linspace(-92, -72, 11) #temp
#edges_lon = np.linspace(-25, -5, 10) #temp
edges_alt = np.arange(25e3, 175e3, 2e3)
edges = edges_lat, edges_lon, edges_alt
#num of columns of jacobian
col_len = (len(edges_lat)+1)*(len(edges_lon)+1)*(len(edges_alt)+1) 

from geometry_functions import los_points_fix_dl, ecef2lla
from oem_functions import jacobian_row

dl = 3e3 #fixed distance between all points
nop = 500 # choose number of points along the line
K_row_idx = []
K_col_idx = []
K_value = []
dll = dl * np.ones(nop) #temp
all_los_lat, all_los_lon, all_los_alt = [], [], []

measurement_id = 0
for image in im_lst:#range(im_start, im_end):
    #====generate points of los for all pixels in each image
    #====all points in ecef coordinate xyz
    lx, ly, lz = los_points_fix_dl(sc_look[image], sc_pos[image], dl=dl, nop=nop)    
    #====convert xyz to lat lon alt for all points
    los_lat, los_lon, los_alt = ecef2lla(lx, ly, lz)
    all_los_lat.append(los_lat)
    all_los_lon.append(los_lon)
    all_los_alt.append(los_alt)
    
    #====build K
    for pix in pix_lst:#range(pix_start, pix_end):   
        los = los_lat.sel(pixel=pix), los_lon.sel(pixel=pix), los_alt.sel(pixel=pix)
        measurement_idx, grid_idx, pathlength = jacobian_row(dll, edges, los, measurement_id)
        K_row_idx.append(measurement_idx)
        K_col_idx.append(grid_idx)
        K_value.append(pathlength)
        measurement_id += 1

#==== create sparse matrix        
K_row_idx = np.concatenate(K_row_idx).astype('int')
K_col_idx = np.concatenate(K_col_idx).astype('int')
K_value = np.concatenate(K_value) # in meter

from scipy.sparse import coo_matrix
K_coo = coo_matrix((K_value, (K_row_idx, K_col_idx)), shape = (row_len, col_len))
#print(K_coo)

#==== all points in all los
import pandas as pd
all_los_lat = xr.concat(all_los_lat, pd.Index(im_lst, name='im'))
all_los_lon = xr.concat(all_los_lon, pd.Index(im_lst, name='im'))
all_los_alt = xr.concat(all_los_alt, pd.Index(im_lst, name='im'))

from geometry_functions import plot_los
%matplotlib qt
plot_los((all_los_lat, all_los_lon, all_los_alt), sc_pos, 
         (tan_lat, tan_lon, tan_alt), edges, im_lst, pix_lst) 
%matplotlib inline

#%% tomo inversion
from oem_functions import linear_oem_sp
import scipy.sparse as sp
y = l1.isel(date=im_lst, pixel=pix_lst).data.ravel()
y[y<0] = 0
xa = np.ones(col_len) # temp
Sa = sp.diags([1], shape=(col_len, col_len)) *1 #temp
Se = sp.diags([1], shape=(measurement_id, measurement_id)) * 1e10 #temporary
x_hat = linear_oem_sp(K_coo, Se, Sa, y, xa)

result_tomo = x_hat.reshape(len(edges_lat)+1, len(edges_lon)+1, len(edges_alt)+1)
result_tomo = xr.DataArray(result_tomo[:-1,:-1,:-1], coords=(edges_lat, edges_lon, edges_alt), 
                      dims=('lat', 'lon', 'alt'), attrs={'unit':'photons m-3 s-1?'})

#%plot some tomo results
#====check residual
plt.figure()
plt.plot(y, label='y')
plt.plot(K_coo.dot(x_hat), label='Kx')

plt.ylabel('signal')
plt.legend()
plt.show()

#====contour plot 
plt.figure()

plt.subplot(121)
plt.contourf(edges_lon, edges_alt*1e-3, result_tomo.sum(axis=0).T, 10)
plt.xlabel('longitude')
plt.ylabel('altitude')
plt.title('sum over lat')

plt.subplot(122)
plt.contourf(edges_lat, edges_alt*1e-3, result_tomo.sum(axis=1).T, 10)
plt.xlabel('latitude')
#plt.ylabel('altitude')
plt.title('sum over lon')
plt.yticks([])
plt.show()

plt.figure()
fig, ax = plt.subplots()
ax.contourf(edges_lon, edges_lat, result_tomo.sum(axis=2), 10)
ax.plot(tan_lon.isel(date=im_lst, pixel=60,), tan_lat.isel(date=im_lst, pixel=60,), label='tangent point')
ax.axis('equal')
plt.ylabel('latitude')
plt.xlabel('longitude')
plt.title('sum over alt')
plt.legend()
#sc_lat, sc_lon, sc_alt = ecef2lla(sc_pos.sel(xyz='x'),sc_pos.sel(xyz='y'),sc_pos.sel(xyz='z'))
#ax.plot(sc_lon.isel(date=im_lst)+360, sc_lat.isel(date=im_lst))
plt.show()

#%% ozone 
from chemi import ozone_sme, ozone_textbook, jfactors, gfactor
from scipy.io import loadmat
MSIS = loadmat('msisdata.mat')
zMsis = MSIS['zMsis'].squeeze() # in km
TMsis = MSIS['TMsis'] # in K
NMsis = MSIS['NMsis'] # in cm-3 ???
monthMsis = MSIS['monthMsis'].squeeze()
latMsis = MSIS['latMsis'].squeeze()
month = 6 #temp
lat = 9 #temp
T = interp1d(zMsis*1e3, TMsis[:,month,lat], fill_value='extrapolate')(edges_alt)
M = interp1d(zMsis*1e3, NMsis[:,month,lat], fill_value='extrapolate')(edges_alt)
#jhart, jsrc, jlya, j3, j2 = jfactors(O, O2, O3, N2, z, zenithangle)

plt.figure()
ax = plt.gca()
o3_tomo = ozone_sme(M, T, result_tomo) #volumn emission rate is photons m-3s-1?
for nlat in range(len(edges_lat)):
    for nlon in range(len(edges_lon)):
        plt.plot(o3_tomo.isel(lat=nlat, lon=nlon), edges_alt, '*')        
ax.set_xscale('log')
ax.set(xlabel='ozone number density / cm-3', 
       ylabel='altitude grid',
       xlim=[1e6,1e12],
       ylim=[40e3, 100e3],
       title='ozone from tomography')       
        

#%% 1D inversion
from oem_functions import linear_oem
from geometry_functions import pathl1d_iris

z = edges_alt
z_top = edges_alt[-1] + 2e3
result_1d = np.zeros((len(im_lst), len(z)))
xa = np.ones(len(z)) # temp
Sa = np.diag(np.ones(len(z))) #*1e-9 #temp
Se = np.diag(np.ones(len(pixel))) * 1e10 #30 #temporary
for i in range(len(im_lst)):
    h = tan_alt.data[im_lst[i],:]
    K = pathl1d_iris(h, z, z_top)    
    y = l1.isel(date=im_lst[i]).data    
#    Se = np.diag(error.data[i,:]**2)
    x, A, Ss, Sm = linear_oem(K, Se, Sa, y, xa)
    result_1d[i,:] = x
#    plt.plot(y, np.arange(128))
#    plt.plot(K.dot(x), np.arange(128))

result_1d = xr.DataArray(result_1d, coords=(date[im_lst], z), dims=('date', 'z'))
result_1d.attrs['units'] = 'photons m-3 s-1 ?'
result_1d.attrs['long_name'] = '1d inversion VER'
result_1d.plot(x='date', y='z',
#         norm=LogNorm(), 
         vmin=0, vmax=1e7, 
         size=5, aspect=3)
ax = plt.gca()
ax.set(title='1d retrieved VER')

plt.figure()
plt.plot(result_1d.T,z, '*')
ax = plt.gca()
ax.set_xscale('log')
ax.set(xlabel='volumn emission rate photons cm-3 s-1', 
       ylabel='altitdue grid',
       title='1d retrieval')

# ozone 
o3 = ozone_sme(M, T, result_1d)
plt.figure()
ax = plt.gca()
ax.plot(o3.T, z, '*')
ax.set_xscale('log')
ax.set(xlabel='ozone number density / cm-3',
       ylabel='altitude grid',
       title='ozone from 1d retrieval',
       xlim=[1e6,1e12],
       ylim=[40e3, 100e3])

#%% 1D interpolation and plotting limb radiance
alts_interp = np.arange(40e3, 120e3, .25e3)
data_interp = []

for (data, alt) in zip(l1, tan_alt):
    f = interp1d(alt, data, bounds_error=False)
    data_interp.append(f(alts_interp))
data_interp = xr.DataArray(data_interp, coords=(date, alts_interp), 
                           dims=('date', 'alt'))
data_interp.attrs['units'] = 'Rayleigh?'
data_interp.attrs['long_name'] = 'interpolated data'

#FIG_SIZE = (15,6)
plt.figure()
data_interp.isel(date=im_lst).plot(x='date', y='alt', 
                 norm=LogNorm(), 
                 vmin=1e9, vmax=1e13, 
                 size=5, aspect=3)
ax = plt.gca()
ax.set(title='From {} \n to {}, \n channel {}'.format(num2date(mjd[0],units),
       num2date(mjd[-1], units), ch))
plt.axvline(x=date[im_lst[0]], color='k', linewidth=1)
plt.axvline(x=date[im_lst[-1]], color='k', linewidth=1)
plt.show()

plt.figure(figsize=(10,5))
plt.subplot(121)
plt.plot(data_interp.isel(date=im_lst).T, alts_interp)
ax = plt.gca()
ax.set(ylabel='tangent altitude', 
       xlabel='limb radiance',
       title='limb radiance for selected images',
       ylim=[50e3,100e3])

plt.subplot(122)
plt.plot(data_interp.isel(date=im_lst).T, alts_interp)
ax = plt.gca()
ax.set(xlabel='limb radiance in unit?',
       title='limb radiance for selected images',
       ylim=[50e3,100e3])
ax.set_xscale('log')
plt.yticks([])
plt.show()
