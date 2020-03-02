#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:38:05 2019

@author: anqili
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.time import Time

#%%
year = [2008]
month = [2]

#%% load iris ver
path = '/home/anqil/Documents/osiris_database/iris_ver_o3/ver/'
filenames = 'ver_{}{}_v5p0.nc'
files = [path+filenames.format(year[i], str(month[i]).zfill(2)) for i in range(len(month))]
data_ver = xr.open_mfdataset(files)
data_ver = data_ver.assign_coords(latitude = data_ver.latitude, 
                                            longitude = data_ver.longitude)

#%% load iris o3
path = '/home/anqil/Documents/osiris_database/iris_ver_o3/o3/'#o3_v6p2_temp/'
#filenames = 'o3_{}{}_mr08_o3false.nc'
#filenames = 'o3_{}{}_5p1_o3false_posVER.nc'
filenames = 'o3_{}{}_v6p2.nc'
files = [path+filenames.format(year[i], str(month[i]).zfill(2)) for i in range(len(month))]
data_o3 = xr.open_mfdataset(files)
#data_o3 = data_o3.where(data_o3.status!=0, drop=True)
data_o3 = data_o3.assign_coords(latitude = data_ver.latitude.sel(mjd=data_o3.mjd), 
                                        longitude = data_ver.longitude.sel(mjd=data_o3.mjd))    

#o3_filter = np.logical_and(data_o3.o3.sel(z=90)<1e9, data_o3.cost_lsq<1e0)
#o3_filter = np.logical_and((data_o3.status == 1), (data_o3.cost_y<=2e2))
#o3_filter = np.logical_or((data_o3.status==2), (data_o3.status==1))
cost_max = 10
o3_filter = np.logical_and(data_o3.status==2, (data_o3.cost_y + data_o3.cost_x) < cost_max)
#o3_filter = (data_o3.o3_mr>0.8)
#o3_filter = np.logical_and(data_o3.o3_mr>0.8, (data_o3.cost_y + data_o3.cost_x) < cost_max)

mr_min = 0.8
bin_count_min = 50

#%% merge ver and o3 data
data_iris = data_o3.merge(data_ver)
lat_bins = np.linspace(-90, 90, 46, endpoint=True)
lat_bins_center = np.diff(lat_bins)/2+lat_bins[:-1]

#%% check how much data is filtered out
count_o3 = data_o3.o3.where(data_o3.o3_mr>mr_min).where(o3_filter).groupby_bins('latitude',lat_bins, 
                                 labels=lat_bins_center).count(dim='mjd')
count_ver = data_ver.ver.where(data_ver.mr>mr_min).groupby_bins('latitude', lat_bins, 
                                 labels=lat_bins_center).count(dim='mjd')

#plt.figure()
#plt.step(lat_bins_center, count_o3, label='O3')
#plt.step(lat_bins_center, count_ver, label='VER')
##plt.legend()
#plt.show()
fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(10,5))
#fig.tight_bbox()
count_o3.plot(y='z', ax=ax[0], vmin=0, vmax=count_ver.max().values, cmap='RdBu_r')
count_ver.plot(y='z', ax=ax[1], vmin=0, vmax=count_ver.max().values, cmap='RdBu_r')
ax[0].set(title='O3', ylim=(60,100))
ax[1].set(title='VER')

#%% look at retrieved ozone zonal mean
vmin=1e6#1e4 #1e6
vmax=1e9# 1e7 #1e10

mean_o3 = data_o3.o3.where(data_o3.o3_mr>mr_min).where(o3_filter).groupby_bins('latitude', lat_bins, 
                                 labels=lat_bins_center).mean(dim='mjd')
mean_ver = data_ver.ver.where(data_ver.mr>mr_min).groupby_bins('latitude', lat_bins, 
                                 labels=lat_bins_center).mean(dim='mjd')
mean_o3 = mean_o3.where(count_o3>bin_count_min)
mean_ver = mean_ver.where(count_ver>bin_count_min)

fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(10,5))
#fig.tight_bbox()
mean_o3.plot(y='z', norm=LogNorm(vmin=vmin, vmax=vmax), extend='both', ax=ax[0], cmap='RdBu_r')
mean_ver.plot(y='z', norm=LogNorm(vmin=5e4, vmax=1e6), extend='both',ax=ax[1], cmap='RdBu_r')
ax[0].set(title='O3', ylim=(60,100))
ax[1].set(title='VER')

#%% line plots
plt.figure()
mean_o3.plot.line(y='z', add_legend=False, xscale='log')
plt.show()

#%% histogram of cost y
plt.figure()
(data_iris.cost_y + data_iris.cost_x).plot.hist(bins=np.logspace(0,2), yscale='log', xscale='log')
plt.axvline(x=cost_max)
plt.xlabel('total cost')


#%%
#plt.figure()
#mean = data_iris.y.groupby_bins('latitude', lat_bins, 
#                                 labels=lat_bins_center).mean(dim='mjd')
##mean_ver.plot.line(y='z', xscale='log', color='k', add_legend=False)
#mean.plot.line(y='z', xscale='log', add_legend=False)
#plt.show()

#%%
#plt.figure()
#im_lst = np.random.uniform(0,len(data_ver.mjd), 10).astype(int)
#mean = data_ver.ver.isel(mjd=im_lst).mean(dim='mjd')
#(np.sqrt(data_ver.ver_error.isel(mjd=im_lst).where(
#        data_ver.mr_rel.isel(mjd=im_lst)>0.8).sortby('mjd'))/mean).plot.line(
#                hue='mjd', y='z', xscale='linear', add_legend=False)
#plt.show()

#%% uncertainty scaled with apriori
fig = plt.figure()
(np.sqrt(data_ver.ver_error.where(
        data_ver.mr_rel>0.8)).mean('mjd')/data_ver.ver_apriori.mean('mjd')).plot(y='z',xscale='linear', label='ver')
(np.sqrt(data_o3.o3_rms.where(
        data_o3.o3_mr>0.8)).mean('mjd')/data_o3.o3_a.mean('mjd')).plot(y='z',xscale='linear', label='o3')
plt.legend()
plt.xlabel('error / a priori')
plt.title('O3 accuracy averaged in one year')
fig.savefig('/home/anqil/Documents/reportFigure/article2/iris_uncertainty.png',
            bbox_inches = "tight")

#%%
fig = plt.figure()
(np.sqrt(data_o3.o3_rms.where(
        data_o3.o3_mr>0.8)).mean('mjd')/data_o3.o3_a.mean('mjd')).plot(y='z',xscale='linear')
plt.xlabel('o3 error / o3 a priori')
plt.title('O3 accuracy in month {}'.format(month[0]))
fig.savefig('/home/anqil/Documents/reportFigure/article2/iris_o3_uncertainty.png',
            bbox_inches = "tight")

#%%
path = '/home/anqil/Documents/osiris_database/iris_ver_o3/ver/'
filenames = 'ver_*.nc'
data_ver = xr.open_mfdataset(path+filenames)
mean_ver_error = (np.sqrt(data_ver.ver_error.where(data_ver.mr_rel>0.8))/data_ver.ver_apriori).mean('mjd').load()
std_ver_error = (np.sqrt(data_ver.ver_error.where(data_ver.mr_rel>0.8))/data_ver.ver_apriori).std('mjd').load()
path = '/home/anqil/Documents/osiris_database/iris_ver_o3/o3/'
filenames = 'o3_*.nc'
data_o3 = xr.open_mfdataset(path+filenames)
mean_o3_error = (np.sqrt(data_o3.o3_rms.where(data_o3.o3_mr>0.8))/data_o3.o3_a).mean('mjd').load()
#mean_o3_error = (np.sqrt(data_o3.o3_rms.where(data_o3.o3_mr>0.8))/data_o3.o3_a).load().median('mjd')
std_o3_error = (np.sqrt(data_o3.o3_rms.where(data_o3.o3_mr>0.8))/data_o3.o3_a).std('mjd').load()

fig = plt.figure()
#plt.errorbar(mean_error, data_ver.z, xerr=std_error)
mean_ver_error.plot(y='z')
plt.fill_betweenx(data_ver.z, mean_ver_error+std_ver_error, mean_ver_error-std_ver_error, alpha=0.5)
mean_o3_error.plot(y='z')
plt.fill_betweenx(data_o3.z, mean_o3_error+std_o3_error, mean_o3_error-std_o3_error, alpha=0.5)
plt.legend(['VER', 'O3'])
plt.xlabel('error / a priori')
plt.title('Accuracy averaged in 12 months')
fig.savefig('/home/anqil/Documents/reportFigure/article2/iris_uncertainty.png',
            bbox_inches = "tight")
