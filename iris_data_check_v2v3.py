#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 11:21:52 2019

@author: anqil
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
Av = 6.023e23 #Avogadro's number: molec/mol
from matplotlib.colors import LogNorm
from astropy.time import Time


#%%
year = [2008]
month = [8]
t_bounds = Time(['{}-{}-01T00:00:00'.format(year[0], str(month[0]).zfill(2)),
               '{}-{}-01T00:00:00'.format(year[-1], str(month[-1]+1).zfill(2))], 
                format='isot', scale='utc')

#%% load old iris data
# make an interpolation of the orbit numbers and time, based on the csv file
path = '/home/anqil/Documents/osiris_database/'
data = np.genfromtxt(path+'odin_rough_orbit.csv', delimiter=',', dtype='str')
orbits, years, months, days, times = data[1:,0], data[1:,1], data[1:,2], data[1:,3], data[1:,4]
orbits = orbits.astype(int)

t = []
for i in range(962):
    t.append(years[i]+'-'+months[i].zfill(2)+'-'+days[i].zfill(2)+'T'+times[i])
t = Time(t, format='isot', scale='utc')

orbit_bounds = np.interp(t_bounds.mjd, t.mjd, orbits)

# extract files that are within the time range
import glob
path = '/home/anqil/Documents/osiris_database/iris_ver_o3/old_process/'
files = [f for f in glob.glob(path + '*sza*')]
files_in_range = []
for i in range(len(files)):
    if int(files[i][-8:-3]) > orbit_bounds[0] and int(files[i][-8:-3]) < orbit_bounds[1]:
        files_in_range.append(files[i])

# load all nc files in the ds
data_iris = xr.open_mfdataset(files_in_range)
data_iris['z'] = data_iris.z*1e-3 #m->km
data_iris = data_iris.assign_coords(latitude = data_iris.tan_lat.sel(pixel=60).drop('pixel'), 
                                    longitude = data_iris.tan_lon.sel(pixel=60).drop('pixel'))
o3_iris = data_iris.o3_iris.where(data_iris.mr>0.8)

#%% load new (v3) iris data
#new calibration,new o3 apriori new kinetic coefficients, new TOA altitude, added B-band and IRA-band
path = '/home/anqil/Documents/osiris_database/iris_ver_o3/'
filenames = 'ver_o3_{}{}_new2.nc'
files = [path+filenames.format(year[i], str(month[i]).zfill(2)) for i in range(len(month))]

data_iris_new = xr.open_mfdataset(files)
data_iris_new['z'] = data_iris_new.z*1e-3 #m->km
data_iris_new = data_iris_new.assign_coords(latitude = data_iris_new.latitude, 
                                            longitude = data_iris_new.longitude)


#%%
plt.figure()
data_iris_new.mjd.plot(ls='', marker='.', label='new')
data_iris.mjd.plot(ls='', marker='.', label='old')

#%%
mjd_sel = slice(0,20000,1000)

plt.figure()
data_iris_new.ver.isel(mjd=mjd_sel).plot.line(y='z', ls='', marker='.', color='r', add_legend=False, xscale='log')
data_iris.ver.isel(mjd=mjd_sel).plot.line(y='z', ls='', marker='.', color='k', add_legend=False, xscale='log')
plt.show()

plt.figure()
data_iris_new.o3.where(data_iris_new.mr>0.8).isel(mjd=mjd_sel).plot.line(y='z', ls='', marker='.', color='r', add_legend=False, xscale='log')
data_iris.o3_iris.where(data_iris.mr>0.8).isel(mjd=mjd_sel).plot.line(y='z', ls='', marker='.', color='k', add_legend=False, xscale='log')

data_iris_new.o3_apriori.isel(mjd=mjd_sel).plot.line(y='z', ls='-', marker='.', color='r', add_legend=False, xscale='log')
data_iris.o3_xa.isel(mjd=mjd_sel).plot.line(y='z', ls='-', marker='.', color='r', add_legend=False, xscale='log')

plt.show()

#%% look at retrieved ozone zonal mean
lat_bins = np.linspace(-90, 90, 46, endpoint=True)
lat_bins_center = np.diff(lat_bins)/2+lat_bins[:-1]

mean_new = data_iris_new.o3.where(data_iris_new.mr>0.8).groupby_bins('latitude', lat_bins, 
                                 labels=lat_bins_center).mean(dim='mjd')
mean_old = data_iris.o3_iris.where(data_iris.mr>0.8).groupby_bins('latitude', lat_bins, 
                                  labels=lat_bins_center).mean(dim='mjd')
counts_new = data_iris_new.o3.where(data_iris_new.mr>0.8).groupby_bins('latitude', lat_bins, 
                                 labels=lat_bins_center).count()
counts_old = data_iris.o3_iris.where(data_iris.mr>0.8).groupby_bins('latitude', lat_bins, 
                                  labels=lat_bins_center).count()
fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=False, figsize=(15,10))
mean_new.plot(y='z', norm=LogNorm(vmin=1e6, vmax=1e10), ax=ax[0], ylim=(60,100))
mean_old.plot(y='z', norm=LogNorm(vmin=1e6, vmax=1e10), ax=ax[1], ylim=(60,100))
ax[0].set(title='new')
ax[1].set(title='old')

lat_sel = np.array([26, 32])
[ax[0].axvline(x=mean_new.latitude_bins[i], color='k') for i in lat_sel]
[ax[1].axvline(x=mean_old.latitude_bins[i], color='k') for i in lat_sel]

plt.figure()
mean_new.isel(latitude_bins=lat_sel).plot.line(y='z', color='r', ls='', marker='.', 
             xscale='log', xlim=(1e6, 1e10), add_legend=False)
mean_old.isel(latitude_bins=lat_sel).plot.line(y='z', color='k', ls='', marker='.', 
             xscale='log', xlim=(1e6, 1e10), add_legend=False)

plt.figure()
#fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=False, figsize=(10,5))
counts_new.plot.step()
counts_old.plot.step()

#%% look at apriori
mean_new = data_iris_new.o3_apriori.groupby_bins('latitude', lat_bins, 
                                 labels=lat_bins_center).mean(dim='mjd')
mean_old = data_iris.o3_xa.groupby_bins('latitude', lat_bins, 
                                  labels=lat_bins_center).mean(dim='mjd')
counts_new = data_iris_new.o3.where(data_iris_new.mr>0.8).groupby_bins('latitude', lat_bins, 
                                 labels=lat_bins_center).count()
counts_old = data_iris.o3_iris.where(data_iris.mr>0.8).groupby_bins('latitude', lat_bins, 
                                  labels=lat_bins_center).count()
fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=False, figsize=(10,5))
mean_new.plot(y='z', norm=LogNorm(vmin=1e6, vmax=1e10), ax=ax[0], ylim=(60,100))
mean_old.plot(y='z', norm=LogNorm(vmin=1e6, vmax=1e10), ax=ax[1], ylim=(60,100))

lat_sel = np.array([22, 30])

[ax[0].axvline(x=mean_new.latitude_bins[i], color='k') for i in lat_sel]
[ax[1].axvline(x=mean_old.latitude_bins[i], color='k') for i in lat_sel]

plt.figure()
mean_new.isel(latitude_bins=lat_sel).plot.line(y='z', color='r', ls='', marker='.', 
             xscale='log', xlim=(1e6, 1e10), add_legend=False, ylim=(60,100))
mean_old.isel(latitude_bins=lat_sel).plot.line(y='z', color='k', ls='', marker='.', 
             xscale='log', xlim=(1e6, 1e10), add_legend=False)

#%% check profiles that are at latitude higher than 50
data_iris_new.o3.where(data_iris_new.latitude>50,drop=True).isel(
        mjd=slice(0,10000,100)).plot.line(y='z', add_legend=False, xscale='log')
plt.show()

##find out which profiles are not normal --> mjd
mjd_problem = data_iris_new.mjd.where(data_iris_new.o3.where(
        data_iris_new.latitude>50,drop=True).isel(
        mjd=slice(0,10000,100)).sel(z=90)>1e9, drop=True).values
        
##%%
#from scipy.optimize import least_squares
#from o2delta_model import cal_o2delta_new
#from multiprocessing import Pool
##%%
#def residual(o3, T, m, z, zenithangle, p, o2delta_meas):
#    o2delta_model = cal_o2delta_new(o3, T, m, z, zenithangle, p)[0]  
#    print(round(o3[30]*1e-6), round(T[30]), round(m[30]), round(p[30]), zenithangle)
##    plt.semilogx(o3, z)#temp
##    plt.xlim=(1e6,1e13)
##    plt.show()
#    return o2delta_meas - o2delta_model
#
##%%
#A_o2delta = 2.23e-4
#bot = 60e3
#top = 102e3
#top_extra = 149e3 # for apriori, cal_o2delta (J and g)
##====retireval grid
#z = np.arange(bot, top, 1e3) # m
#z = np.append(z, top_extra)
#
## load climatology
#path = '/home/anqil/Documents/osiris_database/ex_data/'
#file = 'msis_cmam_climatology.nc'
#clima = xr.open_dataset(path+file)#.interp(z=z*1e-3)
#clima = clima.update({'m':(clima.o + clima.o2 + clima.n2)*1e-6}) #cm-3
#clima = clima.sel(month=month[0])
#o3_clima = clima.o3_vmr * clima.m #cm-3
#
#i = 0
#ver = data_iris_new.ver.sel(mjd=mjd_problem[i])
#mr = data_iris_new.mr.sel(mjd=mjd_problem[i])
#sol_zen = data_iris_new.sza.sel(mjd=mjd_problem[i])
#o3_a = o3_clima.interp(lst_bins=data_iris_new.lst.sel(mjd=mjd_problem[i]),
#                       kwargs={'fill_value': 'extrapolate'}).interp(
#                               lat=ver.latitude, z=z*1e-3)
#T_a = clima.T.interp(lat=ver.latitude, z=z*1e-3)
#m_a = clima.m.interp(lat=ver.latitude, z=z*1e-3)
#p_a = clima.p.interp(lat=ver.latitude, z=z*1e-3) 
#o2delta_meas = ver / A_o2delta
#res_lsq = least_squares(residual, 
#                        o3_a.values[mr>0.8], #initial guess
#                        bounds=(-np.inf, np.inf), verbose=0, 
##                                max_nfev=3, #temp
#                        args=(T_a.values[mr>0.8],
#                              m_a.values[mr>0.8],
#                              z[mr>0.8], sol_zen, 
#                              p_a.values[mr>0.8],
#                              o2delta_meas[mr>0.8]))
#
#o3_iris = xr.DataArray(res_lsq.x, coords={'z': z[mr>0.8]}, dims='z').reindex(z=z)
