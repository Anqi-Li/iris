#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:22:18 2019

@author: anqil
"""
import numpy as np
import numpy.ma as ma
import xarray as xr
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
from netCDF4 import num2date
units = 'days since 1858-11-17 00:00:00.000'
from astropy.time import Time
from multiprocessing import Pool

#%%
def dt64_to_mjd(da):
    from datetime import datetime as dt
    return [dt(da.dt.year[i], 
               da.dt.month[i], 
               da.dt.day[i], 
               da.dt.hour[i], 
               da.dt.minute[i], 
               da.dt.second[i]).mjd for i in range(len(da))]
    
#%% find mjd list of 1 orbit
file = '/home/anqil/Documents/osiris_database/ir_stray_light_corrected.nc'
ir = xr.open_dataset(file)
#orbit = 37580
#ir = ir.where(ir.orbit==orbit, drop=True)
orbit_index = 105#45
orbit_no = np.unique(ir.orbit)[orbit_index]
ir = ir.where(ir.orbit==orbit_no, drop=True)
#print(num2date(ir.mjd,units))
ir_fullorbit = ir
day_mjd_lst = ir.mjd[ir.sza<90]
#ir = ir.sel(mjd=day_mjd_lst)

#%% Limb 1 orbit plot
alts_interp = np.arange(10e3, 150e3, .25e3)
data_interp = []
for (data, alt) in zip(ir.data, ir.altitude):
    f = interp1d(alt, data, bounds_error=False)
    data_interp.append(f(alts_interp))
data_interp = xr.DataArray(data_interp, coords=[ir.mjd, alts_interp], dims=['mjd', 'altitude'])
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(15,5))
data_interp.plot(ax=ax[0], y='altitude', robust=True, add_colorbar=False, norm=LogNorm(vmin=1e9, vmax=1e13), cmap='viridis')
ir.sza.plot(ax=ax[1])
ax[1].axhline(y=90)
ax[0].axvline(x=ir.mjd[828])

#%% load IRIS result
file = '/home/anqil/Documents/osiris_database/iris_ver_o3/semi-old/{}{}_v5.nc'
ds = xr.open_dataset(file.format(num2date(day_mjd_lst, units)[0].year,
                                 str(num2date(day_mjd_lst, units)[0].month).zfill(2)))
data_iris = ds.sel(mjd=slice(day_mjd_lst[0], day_mjd_lst[-1]))
#data_iris['z'] = data_iris.z*1e-3 # to km 

#%% load IRIS result from 2 files
file = '/home/anqil/Documents/osiris_database/iris_ver_o3/ver_{}{}_v5p0.nc'
ds1 = xr.open_dataset(file.format(num2date(day_mjd_lst, units)[0].year,
                                 str(num2date(day_mjd_lst, units)[0].month).zfill(2))) 
file = '/home/anqil/Documents/osiris_database/iris_ver_o3/o3_{}{}_mr08_o3false.nc'
ds2 = xr.open_dataset(file.format(num2date(day_mjd_lst, units)[0].year,
                                 str(num2date(day_mjd_lst, units)[0].month).zfill(2))) 
data_iris = ds1.merge(ds2)
data_iris = data_iris.sel(mjd=slice(day_mjd_lst[0], day_mjd_lst[-1]))

#%% load smr data (needed to be interpolated)
path = '/home/anqil/Documents/osiris_database/ex_data/'
filename = 'smr_200*_sza_interp.nc' 
data_smr = xr.open_mfdataset(path+filename)
data_smr = data_smr.where(data_smr.mjd>ir.mjd[0], drop=True)
data_smr = data_smr.where(data_smr.mjd<ir.mjd[-1], drop=True)
data_smr['altitude'] = data_smr.altitude*1e-3 # to km
data_smr['alt'] = data_smr.alt *1e-3 # to km
#data_smr = data_smr.assign_coords(z=data_smr.altitude[0,:].values) #temp
mr_threshold = 0.8
o3_vmr_smr = data_smr.o3_vmr.where(data_smr.mr>mr_threshold)
Av = 6.023e23 #Avogadro's number: molec/mol
R = 8.31 # gas constant: J/mol/K
m_smr = Av * data_smr.pressure / (R * data_smr.temperature) * 1e-6 # number density of air cm-3
o3_smr = m_smr * o3_vmr_smr # cm-3

#%% load os data
path = '/home/anqil/Documents/osiris_database/odin-osiris.usask.ca/Level2/CCI/OSIRIS_v5_10/'
file = path+'ESACCI-OZONE-L2-LP-OSIRIS_ODIN-SASK_V5_10_HARMOZ_ALT-{}{}-fv0002.nc'.format(
        num2date(ir.mjd[0], units).year, str(num2date(ir.mjd[0], units).month).zfill(2))
data_os = xr.open_dataset(file)
data_os = data_os.sel(time=slice(data_os.sel(time=num2date(day_mjd_lst[0],units), method='nearest').time, 
                                 data_os.sel(time=num2date(day_mjd_lst[-1],units), method='nearest').time))

data_os = data_os.update({'mjd': (['time'], Time(data_os.time, format='datetime64').mjd)}).swap_dims({'time':'mjd'})
o3_os = data_os.ozone_concentration * Av*1e-6 #molec cm-3
o3_os_error = data_os.ozone_concentration_standard_error * Av*1e-6 #molec cm-3
m_os = data_os.pressure/data_os.temperature/8.314e4*Av #air mass density cm-3
o3_vmr_os = o3_os / m_os
o3_vmr_os_error = o3_os_error / m_os

#%% load climatology
path = '/home/anqil/Documents/osiris_database/ex_data/'
file = 'msis_cmam_climatology_z200_lat8576.nc'
clima = xr.open_dataset(path+file)#.interp(z=z*1e-3)
clima = clima.update({'m':(clima.o + clima.o2 + clima.n2)*1e-6}) #cm-3
o3_clima = clima.o3_vmr * clima.m #cm-3

#%% VER 1 orbit plot
#fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(15,5))
fig = plt.figure(figsize=(15,5))
#fig.suptitle('Volume emission rate at orbit {}'.format(ir_fullorbit.orbit[0].values))
gs = GridSpec(1, 2, width_ratios=[3, 1], wspace=0.02)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

data_iris.ver.where(data_iris.mr>0.8, drop=True).plot(ax=ax1, y='z', 
                   robust=True, 
                  norm=LogNorm(vmin=1e5, vmax=1e6),
                  cbar_kwargs={'label':'VER / ($cm^{-3}$ $s^{-1}$)'})
idx = [40, 570, 915, 1250]
for i in idx:
    ax1.axvline(x=data_iris.mjd[i], color='k')
    ax1.text(data_iris.mjd[i]+5e-4, 75, 'Image {}'.format(i), rotation=90, color='w')
ax1.set(title='All images',
        ylabel='Altitude / km',
        xlabel='Latitude / degree N')
ax1.set_xticklabels(np.round(data_iris.latitude.interp(
        mjd=ax1.get_xticks(), kwargs={'fill_value': 'extrapolate'}).values, 1))
ax11 = ax1.twiny()
ax11.xaxis.set_label_position('bottom')
ax11.xaxis.set_ticks_position('bottom')
ax11.spines['bottom'].set_position(('outward', 40))
ax11.set(xlabel='Longitude / degree E')
ax11.set_xlim(ax1.get_xlim())
ax11.set_xticklabels(np.round(data_iris.longitude.interp(
        mjd=ax1.get_xticks(), kwargs={'fill_value': 'extrapolate'}).values, 1))


data_iris.ver.where(data_iris.mr>mr_threshold, drop=True).isel(mjd=idx).plot.line(
        ax=ax2, y='z', xscale='log', marker='.', add_legend=False)
#ax2.legend(['Image 1', 'Image 2', 'Image 3', 'Image 4'])
ax2.legend(['Im {}'.format(i) for i in idx])
ax2.set_yticklabels([])
ax2.set(ylabel='',
        xlabel='VER / ($cm^{-3}$ $s^{-1}$)',
        title='Selected images')
ax2.set_xlim([1e3, 1e7])


plt.rcParams.update({'font.size': 14})
plt.show()
#fig.savefig('/home/anqil/Documents/reportFigure/article2/iris_ver_sample_{}.png'.format(orbit_no),
#            bbox_inches = "tight")

#%% plot measurement response
#fig = plt.figure()
#data_iris.mr_rel.isel(mjd=idx).plot.line(y='z',marker='.', add_legend=False)
#plt.gca().set(xlabel='relative measurement response',
#               ylabel='Altitude / km')
#plt.legend(['Im {}'.format(i) for i in idx])


#%%
plt.figure()
data_iris.o3.where(data_iris.mr>0.8).plot(y='z', robust=True, norm=LogNorm(vmin=1e7, vmax=1e12), ylim=(60,100))

#%% Ozone compare 1 orbit

grid_2d = plt.GridSpec(ncols=2 ,nrows=3, hspace=0, wspace=0.05, bottom=0.25)
grid_1d = plt.GridSpec(ncols=2 ,nrows=3, hspace=0.3, wspace=0.05)
plt.rcParams.update({'font.size': 14})
#for i in range(800,900):#len(day_mjd_lst)):
i = 800
i_mjd = day_mjd_lst[i]

fig = plt.figure(figsize=(15,10))
ax0 = fig.add_subplot(grid_2d[0,:])
ax1 = fig.add_subplot(grid_2d[1,:], sharey=ax0)
ax2 = fig.add_subplot(grid_1d[2,0])
ax3 = fig.add_subplot(grid_1d[2,1], sharey=ax2)

cmap = 'RdBu_r'#'twilight_shifted'#'RdBu_r'
vmin = 1e7
vmax=1e13

ax0.pcolor(np.tile(data_smr.mjd,(len(data_smr.z),1)),
           data_smr.altitude.T, 
           o3_smr.T, 
           norm=LogNorm(vmin=vmin, vmax=vmax),
           cmap=cmap)
ax0.axvline(x = data_smr.sel(mjd=i_mjd, method='nearest').mjd, color='k')
ax0.set(ylim=[20, 100], 
        xlim=[data_smr.mjd[0], data_smr.mjd[-1]],
#        title='SMR',
        ylabel='Altitude / km')
ax0.set_xticklabels([])
ax0.text(0.25, 0.87, 'SMR', fontsize=18, transform=ax0.transAxes)

im = ax1.pcolor(day_mjd_lst,
                data_iris.z, 
                data_iris.o3.where(data_iris.mr>mr_threshold).reindex({'mjd': day_mjd_lst}).T, 
                norm=LogNorm(vmin=vmin, vmax=vmax),
                cmap=cmap)
#im = data_iris.o3.where(data_iris.mr>mr_threshold).plot(y='z', 
#                       ax=ax1, norm=LogNorm(vmin=1e7, vmax=1e12),
#                       ylim=(60,100), add_colorbar=False, cmap='RdBu_r')

o3_os[1:].plot(y='altitude', ax=ax1, norm=LogNorm(vmin=vmin, vmax=vmax), 
           cmap=cmap, add_colorbar=False)
ax1.axvline(x=data_iris.sel(mjd=i_mjd, method='nearest').mjd, color='k', ymin=0.5)
ax1.axvline(x=data_os.sel(mjd=i_mjd, method='nearest').mjd, color='k', ymax=0.5)
ax1.set(ylim=[10,100],
        xlim=[data_smr.mjd[0], data_smr.mjd[-1]],
        #title='OS (z<60 km) + IRIS (z>60 km) ',
        ylabel='Altitude / km',
        xlabel='Latitude / degree N')
ax1.text(0.01, 0.7, 'IRIS', fontsize=18, transform=ax1.transAxes)
ax1.text(0.02, 0.3, 'OS', fontsize=18, transform=ax1.transAxes)
ax1.set_xticklabels(np.round(data_smr.latitude.interp(
        mjd=ax1.get_xticks(), kwargs={'fill_value': 'extrapolate'}).values, 1))
ax11 = ax1.twiny()
ax11.xaxis.set_label_position('bottom')
ax11.xaxis.set_ticks_position('bottom')
ax11.spines['bottom'].set_position(('outward', 40))
ax11.set(xlabel='Longitude / degree E')
ax11.set_xlim(ax1.get_xlim())
ax11.set_xticklabels(np.round(data_smr.longitude.interp(
        mjd=ax1.get_xticks(), kwargs={'fill_value': 'extrapolate'}).values, 1))

fig.colorbar(im, ax=[ax0,ax1, ax2,ax3], label='Ozone number density /cm-3')


ax2.semilogx(o3_os.sel(mjd=i_mjd, method='nearest'), 
             data_os.altitude, '*',
              label='OS')
ax2.semilogx(data_iris.o3.where(data_iris.mr>mr_threshold).sel(mjd=i_mjd, method='nearest'),
             data_iris.z, '*', 
              label='IRIS')# (mr>{})'.format(mr_threshold))
ax2.semilogx(o3_smr.sel(mjd=i_mjd, method='nearest'), 
             data_smr.altitude.sel(mjd=i_mjd, method='nearest'), '*',
              label='SMR')# (mr>{})'.format(mr_threshold))
#ax2.semilogx(o3_clima.sel(lat=data_iris.sel(mjd=i_mjd, method='nearest').latitude, 
#                          month=num2date(i_mjd, units).month, 
#                          lst=data_iris.sel(mjd=i_mjd, method='nearest').lst, 
#                          method='nearest'), o3_clima.z)

ax2.set_xlim(left=1e4, right=1e13)
ax2.set(#title='Number density',
          xlabel='Number density / cm-3',
          ylabel='Altitude / km',
          ylim=[10, 100])
ax2.legend(loc='lower left')

m_iris = interp1d(data_smr.altitude.sel(mjd=i_mjd, method='nearest'), 
                 data_smr.density.sel(mjd=i_mjd, method='nearest'),
                fill_value="extrapolate")(data_iris.z)

ax3.semilogx(o3_vmr_os.sel(mjd=i_mjd, method='nearest')*1e6,
             data_os.altitude, '*',
             label='OS')
ax3.semilogx(data_iris.o3.where(data_iris.mr>mr_threshold).sel(
             mjd=i_mjd, method='nearest')/m_iris*1e6, data_iris.z, '*',
             label='IRIS')# (mr>{})'.format(mr_threshold))
ax3.semilogx(o3_vmr_smr.sel(mjd=i_mjd, method='nearest')*1e6, 
             data_smr.altitude.sel(mjd=i_mjd, method='nearest'), '*',
             label='SMR')# (mr>{})'.format(mr_threshold))
#ax3.semilogx((clima.o3_vmr*1e6).sel(lat=data_iris.sel(mjd=i_mjd, method='nearest').latitude, 
#                                      month=num2date(i_mjd, units).month, 
#                                      lst=data_iris.sel(mjd=i_mjd, method='nearest').lst, 
#                                      method='nearest'), clima.z)
ax3.set_yticklabels([])

ax3.legend(loc='lower left')
ax3.set_xlim(left=1e-2, right=1e1)
ax3.set(#title='volume mixing ratio',
          xlabel='VMR / ppmv')
  
fig.savefig('/home/anqil/Documents/reportFigure/article2/ozone_compare_{}.png'.format(orbit_no),
            bbox_inches = "tight")
    