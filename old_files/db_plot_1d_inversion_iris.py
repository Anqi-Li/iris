#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 17:14:44 2019

@author: anqil
"""

import sqlite3 as sql
import sys
sys.path.append('..')
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
from netCDF4 import num2date
units = 'days since 1858-11-17 00:00:00.000'
import astropy.coordinates as coord
from astropy.time import Time
import astropy.units as u
from oem_functions import linear_oem
from geometry_functions import pathl1d_iris
plt.ioff()


def extract_iris_db(orbit, ch, figure_dir):
    try:
        db = sql.connect(db_file)
        cur = db.cursor()
        return_column = ('data, mjd, look_ecef, sc_position_ecef, latitude, longitude, altitude')              
        select_str = 'SELECT {} FROM IRI JOIN channel{} ON IRI.stw = channel{}.stw WHERE orbit={}'
        result = cur.execute(select_str.format(return_column, ch, ch, orbit))
        all_image = result.fetchall()
        if len(all_image) == 0:
            print('No data for orbit {}'.format(orbit))
        print(orbit,' : num of images: {}'.format(len(all_image)))
        db.close()
        
        
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
        l1 = xr.DataArray(l1, coords=(mjd, pixel), 
                          dims=('mjd', 'pixel'), 
                          attrs={'units':'Rayleigh??'})
        sc_look = xr.DataArray(sc_look, coords=(mjd, pixel, ['x', 'y', 'z']), 
                               dims=('mjd', 'pixel', 'xyz'))
        sc_pos = xr.DataArray(sc_pos, coords=(mjd, ['x', 'y', 'z']), 
                              dims=('mjd', 'xyz'))
        tan_lat = xr.DataArray(tan_lat, coords=(mjd, pixel),
                               dims=('mjd', 'pixel'), 
                               attrs={'units':'degree'})
        tan_lon = xr.DataArray(tan_lon, coords=(mjd, pixel),
                               dims=('mjd', 'pixel'), 
                               attrs={'units':'degree'})
        tan_alt = xr.DataArray(tan_alt, coords=(mjd, pixel),
                               dims=('mjd', 'pixel'), 
                               attrs={'units':'meter'})
    #    import pandas as pd
    #    from geometry_functions import lla2ecef
    #    tan_ecef = xr.concat(lla2ecef(tan_lat,tan_lon,tan_alt), 
    #                         pd.Index(['x','y','z'], name='xyz'))
    #    
        #====drop all dates which have nan in l1
        l1 = l1.dropna('mjd')
        sc_look = sc_look.sel(mjd=l1.dropna('mjd').mjd)
        sc_pos = sc_pos.sel(mjd=l1.dropna('mjd').mjd)
        tan_lat = tan_lat.sel(mjd=l1.dropna('mjd').mjd)
        tan_lon = tan_lon.sel(mjd=l1.dropna('mjd').mjd)
        tan_alt = tan_alt.sel(mjd=l1.dropna('mjd').mjd)
        mjd = l1.dropna('mjd').mjd.data
        print('num of images after removing nan: {}'.format(len(date)))
            
        
        #==== plot limb radiance
        alts_interp = np.arange(40e3, 120e3, .25e3)
        data_interp = []
        
        for (data, alt) in zip(l1, tan_alt):
            f = interp1d(alt, data, bounds_error=False)
            data_interp.append(f(alts_interp))
        data_interp = xr.DataArray(data_interp, 
                                   coords=(mjd, alts_interp), 
                                   dims=('mjd', 'alt'))
        data_interp.attrs['units'] = 'Rayleigh?'
        data_interp.attrs['long_name'] = 'interpolated data'
        
        #==== plot the full orbit
        fig = plt.figure()
        data_interp.plot(x='mjd', y='alt', 
                         norm=LogNorm(), 
                         vmin=1e9, vmax=1e13, 
                         size=5, aspect=3)
        ax = plt.gca()
        ax.set(title='From {} \n to {}, \n channel {}'.format(num2date(mjd[0],units),
               num2date(mjd[-1], units), ch),
              xlabel='image index')
    
        ax.set_xticks(mjd[np.arange(0,len(mjd),300, dtype=int)])
        ax.set_xticklabels(np.arange(0,len(mjd),300))
        plt.savefig(figure_dir+'limb_{}.png'.format(orbit), bbox_inches = "tight") #fig file name
        plt.close(fig)
        
        
#        #==== calculate sza
#        loc = coord.EarthLocation(lon=tan_lon.sel(pixel=64)*u.deg,
#                                  lat=tan_lat.sel(pixel=64)*u.deg)
#        time = Time(mjd, format='mjd')
#        altaz = coord.AltAz(location=loc, obstime=time)
#        sun = coord.get_sun(time)
#        
#        sza = sun.transform_to(altaz).zen
#        
#        #==== choose dayglow measurements
#        
#        im_lst = np.where(sza.deg<90)[0]
#        pix_lst = np.arange(22,128)
#        
#        #====1D inversion
#        z = np.arange(50e3, 130e3, 2e3)
#        z_top = 170e3
#        result_1d = np.zeros((len(im_lst), len(z)))
#        xa = np.ones(len(z)) *0 # temp
#        Sa = np.diag(np.ones(len(z))) *(1e7)**2 #temp
#        Se = np.diag(np.ones(len(pix_lst))) * (1e12)**2# 1e10 #30 #temporary
#        Ave = []
#        #residual = []
#        for i in range(len(im_lst)):
#            h = tan_alt.isel(mjd=im_lst[i], pixel=pix_lst).data
#            K = pathl1d_iris(h, z, z_top)    
#            y = l1.isel(mjd=im_lst[i], pixel=pix_lst).data    
#        #    Se = np.diag(error.data[i,:]**2)
#            x, A, Ss, Sm = linear_oem(K, Se, Sa, y, xa)
#            result_1d[i,:] = x
#            Ave.append(A.sum(axis=1)) #sum over rows 
#        #    residual.append(y-K.dot(x))
#        
#        
#        result_1d = xr.DataArray(result_1d, 
#                                 coords=(mjd[im_lst], z), 
#                                 dims=('mjd', 'z'))
#        result_1d.attrs['units'] = 'photons m-3 s-1 ?'
#        result_1d.attrs['long_name'] = '1d inversion VER'
#        Ave = np.array(Ave)
#        mr_threshold = 0.8
#        result_1d_mean = result_1d.where(Ave>mr_threshold).mean(dim='mjd')
#        #====invert once more
#        xa = result_1d_mean.data
#        for i in range(len(im_lst)):
#            h = tan_alt.isel(mjd=im_lst[i], pixel=pix_lst).data
#            K = pathl1d_iris(h, z, z_top)    
#            y = l1.isel(mjd=im_lst[i], pixel=pix_lst).data    
#        #    Se = np.diag(error.data[i,:]**2)
#            x, A, Ss, Sm = linear_oem(K, Se, Sa, y, xa)
#            result_1d[i,:] = x
#            Ave.append(A.sum(axis=1)) #sum over rows 
#        #    residual.append(y-K.dot(x))
#        
#        
#        result_1d = xr.DataArray(result_1d, 
#                                 coords=(mjd[im_lst], z), 
#                                 dims=('mjd', 'z'))
#        result_1d.attrs['units'] = 'photons m-3 s-1 ?'
#        result_1d.attrs['long_name'] = '1d inversion VER'
#        Ave = np.array(Ave)
#        mr_threshold = 0.8
#        result_1d_mean = result_1d.where(Ave>mr_threshold).mean(dim='mjd')        
#        
#        #==== plot VER contour
#        fig = plt.figure()
#        result_1d = abs(result_1d)
#        result_1d.plot(x='mjd', y='z', 
#                       #norm=LogNorm(), 
#                       vmin=0, vmax=0.6e7, 
#                       size=3, aspect=3)
#        ax = plt.gca()
#        ax.set(title='1d retrieved VER',
#              xlabel='mjd')
#        ax.set_xticks(mjd[im_lst[::300]])
#    #    ax.set_xticklabels([])
#        plt.savefig(figure_dir+'1d_day_contour_{}.png'.format(orbit), bbox_inches = "tight") #fig file name
#        plt.close(fig)
#        
#        #==== plot VER vertical profile
#        fig = plt.figure()
#        ax = plt.gca()
#        ax.plot(result_1d.T, z, '.')
#        result_1d_mean.plot(y='z', color='k',ls='-',
#                            label='averaged profile with sum(A)>.{}'.format(mr_threshold))
#        ax.set_xscale('linear')
#        ax.set(xlim=[0, 1.5e7],
#               xlabel='volumn emission rate photons cm-3 s-1', 
#               ylabel='altitdue grid',
#               title='1d retrieval')
#        ax.legend(loc='upper left')
#        plt.savefig(figure_dir+'1d_day_vertical_{}.png'.format(orbit), bbox_inches = "tight") #fig file name
#        plt.close(fig)
#        
#        #==== choose nightglow measurements
#        im_lst = np.where(sza.deg>110)[0]
#        pix_lst = np.arange(22,128)
#        
#        #====1D inversion
#        z = np.arange(50e3, 130e3, 2e3)
#        z_top = 170e3
#        result_1d = np.zeros((len(im_lst), len(z)))
#        xa = np.ones(len(z)) *0 # temp
#        Sa = np.diag(np.ones(len(z))) *(1e7)**2 #temp
#        Se = np.diag(np.ones(len(pix_lst))) * (1e12)**2# 1e10 #30 #temporary
#        Ave = []
#        #residual = []
#        for i in range(len(im_lst)):
#            h = tan_alt.isel(mjd=im_lst[i], pixel=pix_lst).data
#            K = pathl1d_iris(h, z, z_top)    
#            y = l1.isel(mjd=im_lst[i], pixel=pix_lst).data    
#        #    Se = np.diag(error.data[i,:]**2)
#            x, A, Ss, Sm = linear_oem(K, Se, Sa, y, xa)
#            result_1d[i,:] = x
#            Ave.append(A.sum(axis=1)) #sum over rows 
#        #    residual.append(y-K.dot(x))
#        
#        
#        result_1d = xr.DataArray(result_1d, 
#                                 coords=(mjd[im_lst], z), 
#                                 dims=('mjd', 'z'))
#        result_1d.attrs['units'] = 'photons m-3 s-1 ?'
#        result_1d.attrs['long_name'] = '1d inversion VER'
#        Ave = np.array(Ave)
#        mr_threshold = 0.8
#        result_1d_mean = result_1d.where(Ave>mr_threshold).mean(dim='mjd')
#        #====invert once more
#        xa = result_1d_mean.data
#        for i in range(len(im_lst)):
#            h = tan_alt.isel(mjd=im_lst[i], pixel=pix_lst).data
#            K = pathl1d_iris(h, z, z_top)    
#            y = l1.isel(mjd=im_lst[i], pixel=pix_lst).data    
#        #    Se = np.diag(error.data[i,:]**2)
#            x, A, Ss, Sm = linear_oem(K, Se, Sa, y, xa)
#            result_1d[i,:] = x
#            Ave.append(A.sum(axis=1)) #sum over rows 
#        #    residual.append(y-K.dot(x))
#        
#        
#        result_1d = xr.DataArray(result_1d, 
#                                 coords=(mjd[im_lst], z), 
#                                 dims=('mjd', 'z'))
#        result_1d.attrs['units'] = 'photons m-3 s-1 ?'
#        result_1d.attrs['long_name'] = '1d inversion VER'
#        Ave = np.array(Ave)
#        mr_threshold = 0.8
#        result_1d_mean = result_1d.where(Ave>mr_threshold).mean(dim='mjd')        
#                
#        #==== plot VER contour
#        fig = plt.figure()
#        result_1d = abs(result_1d)
#        result_1d.plot(x='mjd', y='z', 
#                       #norm=LogNorm(), 
#                       vmin=0, vmax=5e5, 
#                       size=3, aspect=3)
#        ax = plt.gca()
#        ax.set(title='1d retrieved VER',
#              xlabel='mjd')
#        ax.set_xticks(mjd[im_lst[::300]])
#        plt.savefig(figure_dir+'1d_night_contour_{}.png'.format(orbit), bbox_inches = "tight") #fig file name
#        plt.close(fig)
#        #==== plot VER vertical profile
#        fig = plt.figure()
#        ax = plt.gca()
#        ax.plot(result_1d.T, z, '.')
#        result_1d_mean.plot(y='z', color='k',ls='-',
#                            label='averaged profile with sum(A)>.{}'.format(mr_threshold))
#        ax.set_xscale('linear')
#        ax.set(xlim=[-3e4, 5e5],
#               xlabel='volumn emission rate photons cm-3 s-1', 
#               ylabel='altitdue grid',
#               title='1d retrieval')
#        ax.legend(loc='upper left')
#        plt.savefig(figure_dir+'1d_night_vertical_{}.png'.format(orbit), bbox_inches = "tight") #fig file name
#        plt.close(fig)
    except:
        pass
    
        return


figure_dir = '/home/anqil/Documents/osiris_database/limb_radiance_plots/2006/'
db_file = '/home/anqil/Documents/osiris_database/OSIRIS_2006.db'
db = sql.connect(db_file)
cur = db.cursor()
select_str = 'SELECT orbit from IRI'
result = cur.execute(select_str)
all_image = result.fetchall()
db.close()
orb_lst = np.unique(all_image)
orb_from = 26400
idx_from = (np.abs(orb_lst - orb_from)).argmin()
for orbit in orb_lst[idx_from:]:
    extract_iris_db(orbit, 3, figure_dir)
    
    

