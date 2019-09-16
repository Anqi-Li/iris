#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 14:12:15 2019

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
db_filename = 'OSIRIS_three_orbits_test.db'
db = sql.connect(db_filename)
cur = db.cursor()

#check column titles for each table 
table_name = "IRI" #choose which table you want to check
table_info = cur.execute('SELECT * FROM pragma_table_info("{}")'.format(table_name))
header = table_info.fetchall()
print(np.array(header))

db.close()

#%% check hwo many orbits are in the database
select_str = 'select orbit,mjd from IRI'
db_filename = 'OSIRIS_three_orbits_test.db'
db = sql.connect(db_filename)
cur = db.cursor()
result = cur.execute(select_str)
all_image = result.fetchall()
db.close()
orbit_list = np.unique(np.transpose(all_image)[0])
date_list = num2date(np.transpose(all_image)[1],units)
print(orbit_list)
print(date_list)

#%% open database file
db_filename = 'OSIRIS_three_orbits_test.db'
db = sql.connect(db_filename)
cur = db.cursor()

orbit = 20900
num_of_orbits = 10
ch = 2
return_column = ('data,'+ 'error,' + 'altitude,' 
                 + 'mjd,' + 'latitude,' + 'longitude,'
                 + 'sc_position_ecef')
                 
select_str = 'SELECT {} FROM IRI JOIN channel{} ON IRI.stw = channel{}.stw WHERE orbit>={} AND orbit<={}'
result = cur.execute(select_str.format(return_column, ch, ch, orbit, orbit+num_of_orbits))
all_image = result.fetchall()
db.close()
if len(all_image) == 0:
    print('No data for this orbit(s)')
    

print('num of image: {}'.format(len(all_image)))
#%%
l1_blob = np.array(all_image)[:,0]
error_blob = np.array(all_image)[:,1]
alt_blob = np.array(all_image)[:,2]
mjd = np.array(all_image)[:,3].astype(float)
date = num2date(mjd, units)
lat_blob = np.array(all_image)[:,4]
lon_blob = np.array(all_image)[:,5]
sc_pos_blob = np.array(all_image)[:,6]

#unfolding blobs
l1 = np.empty((len(all_image),128))
error = np.empty((len(all_image),128))
lat = np.empty((len(all_image),128))
lon = np.empty((len(all_image),128))
alts_tan = np.empty((len(all_image),128))
sc_pos = np.empty((len(all_image), 3))
for i in range(len(all_image)):
    l1[i,:] = np.frombuffer(l1_blob[i])
    error[i,:] = np.frombuffer(error_blob[i])
    lat[i,:] = np.frombuffer(lat_blob[i])
    lon[i,:] = np.frombuffer(lon_blob[i])
    alts_tan[i,:] = np.frombuffer(alt_blob[i])
    sc_pos[i,:] = np.frombuffer(sc_pos_blob[i])

# construct xarray dataset
pixel = np.arange(22,128)
l1 = xr.DataArray(l1[:,22:], coords=(date, pixel), dims=('date', 'pixel'), 
                  attrs={'units':'Rayleigh??'})
error = xr.DataArray(error[:,22:], coords=l1.coords, dims=l1.dims, 
                  attrs={'units':'??'})
alts_tan = xr.DataArray(alts_tan[:,22:], coords=l1.coords, dims=l1.dims, 
                  attrs={'units':'m'})
lat = xr.DataArray(lat[:,22:], coords=l1.coords, dims=l1.dims)
lon = xr.DataArray(lon[:,22:], coords=l1.coords, dims=l1.dims)
sc_pos = xr.DataArray(sc_pos, coords=(date, ['x', 'y', 'z']), dims=('date', 'xyz'))
#dataset = xr.Dataset({'l1':l1, 'error':error, 'tan_alt':alts_tan, 
#                      'lat':lat_pix64, 'lon':lon_pix64}) 


#%% 1D interpolation and plotting limb radiance
alts_interp = np.arange(40e3, 120e3, .25e3)
data_interp = []

for (data, alt) in zip(l1, alts_tan):
    f = interp1d(alt, data, bounds_error=False)
    data_interp.append(f(alts_interp))
data_interp = xr.DataArray(data_interp, coords=(date, alts_interp), 
                           dims=('date', 'alt'))
data_interp.attrs['units'] = 'Rayleigh?'
data_interp.attrs['long_name'] = 'interpolated data'

#FIG_SIZE = (15,6)
data_interp.plot(x='date', y='alt', 
                 norm=LogNorm(), 
                 vmin=1e9, vmax=1e12, 
                 size=5, aspect=3)
ax = plt.gca()
ax.set(title='From {} \n to {}, \n channel {}'.format(num2date(mjd[0],units),
       num2date(mjd[-1], units), ch))


#%% 1D inversion
from oem_functions import linear_oem
from geometry_functions import pathl1d_iris

z = np.arange(60e3, 100e3, 0.5e3) 
z_top = 100e3
ver = np.zeros((len(mjd), len(z)))
xa = np.ones(len(z)) * 0 # temp
Sa = np.diag(np.ones(len(z))) *1e-9 #temp
Se = np.diag(np.ones(len(pixel))) * 30 #temporary
for i in range(len(mjd)):
    h = alts_tan.data[i,:]
    K = pathl1d_iris(h, z, z_top)    
    y = l1.data[i,:]    
#    Se = np.diag(error.data[i,:]**2)

    x, A, Ss, Sm = linear_oem(K, Se, Sa, y, xa)
    ver[i,:] = x

ver = xr.DataArray(ver, coords=(date, z), dims=('date', 'z'))
ver.attrs['units'] = 'xxx?'
ver.attrs['long_name'] = '1d inversion VER'
ver.plot(x='date', y='z',
#         norm=LogNorm(), 
         vmin=0, vmax=1e6, 
         size=5, aspect=3)



#%% calculate sza
from skyfield.api import load
from skyfield.api import Topos

planets = load('de421.bsp')
earth, sun = planets['earth'], planets['sun']

ts = load.timescale()
t = ts.now() #change to mjd

boston = earth + Topos(latitude_degrees=lat[0], 
                       longitude_degrees=lon[0],
                       elevation_m=alts_tan[0,0]) #change altitude
astrometric = boston.at(t).observe(sun)
elv = astrometric.apparent().altaz(0,0)

print(elv)
#print(az)
#print(d.m)


#%% line of sight 
#comput points along line of sights for all pixels (in meter)
from geometry_functions import los_points, lla2ecef, ecef2lla
lx, ly, lz, dl = los_points(sc_pos[299], lat[299], lon[299], alts_tan[299])

#convert xyz to lon lat alt for all points
los_lat, los_lon, los_alt = ecef2lla(lx, ly, lz)

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


