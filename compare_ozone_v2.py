#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:27:07 2019

@author: anqil
"""

import numpy as np
import xarray as xr
import matplotlib.pylab as plt
#from osirisl1services.readlevel1 import open_level1_ir
#from osirisl1services.services import Level1Services
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
from netCDF4 import num2date
units = 'days since 1858-11-17 00:00:00.000'

#%% load IRIS data
#channel = 3
#orbit = 37608
#
#ir = open_level1_ir(orbit, channel, valid=False)
#altitude = ir.l1.altitude.sel(pixel=slice(14, 128))
#latitude = ir.l1.latitude.sel(pixel=slice(14, 128))
#longitude = ir.l1.longitude.sel(pixel=slice(14, 128))
##sc_look = ir.l1.look_ecef.sel(pixel=slice(14, 128))
##sc_pos = ir.l1.position_ecef
#l1 = ir.data.sel(pixel=slice(14, 128)) 
#mjd = ir.mjd.data
#pixel = ir.pixel.sel(pixel=slice(14, 128)).data
#
## calculate sza
#import astropy.coordinates as coord
#from astropy.time import Time
#import astropy.units as u
#
#loc = coord.EarthLocation(lon=longitude.sel(pixel=60) * u.deg,
#                          lat=latitude.sel(pixel=60) * u.deg)
#now = Time(ir.mjd, format='mjd', scale='utc')
#altaz = coord.AltAz(location=loc, obstime=now)
#sun = coord.get_sun(now)
#sza = 90 - sun.transform_to(altaz).alt.deg
#sza = xr.DataArray(sza, coords=(mjd,), dims=('mjd',), name='sza')
#
#lst = np.mod(longitude.sel(pixel=60)/15 + np.modf(mjd)[0]*24, 24)
#lst = xr.DataArray(lst, coords=(mjd,), dims=('mjd',), name='sza')

orbit = 39004
path = '/home/anqil/Downloads/stray_light_removed/'
filename= 'ir_slc_{}_ch3.nc'.format(orbit)

ir = xr.open_dataset(path+filename)
mjd = ir.mjd.data
pixel = ir.pixel.sel(pixel=slice(14, 128)).data
l1 = ir.sel(pixel=slice(14, 128)).data
altitude = ir.altitude.sel(pixel=slice(14, 128))
latitude = ir.latitude
longitude = ir.longitude
sza = ir.sza
lst = ir.apparent_solar_time.sel(pixel=60)

#%% 1D inversion and retrieve ozone
from scipy.optimize import least_squares
from o2delta_model import cal_o2delta, gA
from multiprocessing import Pool

#%% oem for dense matrix
def linear_oem(K, Se, Sa, y, xa):
    if len(y.shape) == 1:
        y = y.reshape(len(y),1)
    if len(xa.shape) == 1:
        xa = xa.reshape(len(xa),1)
        
    
    if len(y)<len(xa): # m form
        G = Sa.dot(K.T).dot(np.linalg.inv(K.dot(Sa).dot(K.T) + Se))
        
        
    else: # n form
        Se_inv = np.linalg.inv(Se)
        Sa_inv = np.linalg.inv(Sa)
        G = np.linalg.inv(K.T.dot(Se_inv).dot(K) + Sa_inv).dot(K.T).dot(Se_inv)
#        G= np.linalg.solve(K.T.dot(Se_inv).dot(K) + Sa_inv, (K.T).dot(Se_inv))
        
    x_hat = xa + G.dot(y - K.dot(xa)) 
    A = G.dot(K)
    Ss = (A - np.identity(len(xa))).dot(Sa).dot((A - np.identity(len(xa))).T) # smoothing error
    Sm = G.dot(Se).dot(G.T) #retrieval noise 
    
    return x_hat.squeeze(), A, Ss, Sm

#%% path length 1d
def pathl1d_iris(h, z=np.arange(40e3, 110e3, 1e3), z_top=150e3):
    #z: retrieval grid in meter
    #z_top: top of the atmosphere under consideration 
    #h: tangent altitude of line of sight
    if z[1]<z[0]:
        z = np.flip(z) # retrieval grid has to be ascending
        print('z has to be fliped')
    
#    if h[1]<h[0]: # measred tangent alt grid has to be ascending
#        h = np.flip(h)
#        print('h has to be fliped')
    
    Re = 6370e3 # earth's radius in m
    z = np.append(z, z_top) + Re
    h = h + Re
    pl = np.zeros((len(h), len(z)-1))
    for i in range(len(h)):
        for j in range(len(z)-1):
            if z[j+1]> h[i]:
                pl[i,j] = np.sqrt(z[j+1]**2 - h[i]**2)
                
    pathl = np.append(np.zeros((len(h),1)), pl[:,:-1], axis=1)
    pathl = pl - pathl
    pathl = 2*pathl        
    
    return pathl 


#%% load climatology
path = '/home/anqil/Documents/osiris_database/ex_data/'
file = 'msis_cmam_climatology.nc'
clima = xr.open_dataset(path+file)#.interp(z=z*1e-3)
clima = clima.update({'m':(clima.o + clima.o2 + clima.n2)*1e-6}) #cm-3
o3_clima = clima.o3_vmr * clima.m #cm-3

#%% define oem inversion grid
#====drop data below and above some altitudes
bot = 60e3
top = 102e3

l1 = l1.where((altitude<top) & (altitude>bot))
#====retireval grid
z = np.arange(bot, top, 1e3) # m
z = np.append(z, 149e3)
z_top = z[-1] + 1e3
day_mjd_lst = mjd[sza<90]

#%%
def residual(o3, T, m, z, zenithangle, gA, o2delta_meas):
    o2delta_model = cal_o2delta(o3, T, m, z, zenithangle, gA)[0]  
#    plt.semilogx(o3, z) #temp
#    plt.show()
    return o2delta_meas - o2delta_model
A_o2delta = 2.23e-4
fr = 0.72 # filter fraction 
normalize = np.pi*4 / fr

def f(i):
#    try:
        print(i, 'out of', len(day_mjd_lst))
        
        o3_a = o3_clima.interp(month=num2date(day_mjd_lst[i], units).month,
                               lat=latitude.sel( mjd=day_mjd_lst[i], method='nearest'),
                               lst_bins=lst.sel(mjd=day_mjd_lst[i], method='nearest'),
                               )
        T_a = clima.T.interp(month=num2date(day_mjd_lst[i], units).month,
                             lat=latitude.sel(mjd=day_mjd_lst[i])
                             )
        m_a = clima.m.interp(month=num2date(day_mjd_lst[i], units).month,
                             lat=latitude.sel(mjd=day_mjd_lst[i])
                             )
        p_a = clima.p.interp(month=num2date(day_mjd_lst[i], units).month,
                             lat=latitude.sel(mjd=day_mjd_lst[i])
                             ) 

        sol_zen = sza.sel(mjd=day_mjd_lst[i], method='nearest').item()
        o2delta_a = cal_o2delta(o3_a.data, T_a.data, m_a.data, clima.z.data*1e3, 
                                sol_zen, gA(p_a, sol_zen))[0]
        xa = np.interp(z, clima.z*1e3, o2delta_a) * A_o2delta
        Sa = np.diag(xa**2)
        h = altitude.sel(mjd=day_mjd_lst[i]).where(l1.sel(mjd=day_mjd_lst[i]).notnull(), drop=True)
        K = pathl1d_iris(h, z, z_top) * 1e2 # m-->cm    
        y = l1.sel(mjd=day_mjd_lst[i]).where(l1.sel(mjd=day_mjd_lst[i]).notnull(),drop=True).data *normalize
        Se = np.diag(np.ones(len(y))) *(1e11*normalize)**2 #temp
        
        x, A, Ss, Sm = linear_oem(K, Se, Sa, y, xa)
        mr = A.sum(axis=1)
        
        #lsq fit to get ozone
        o2delta_meas = x / A_o2delta # cm-3?
        res_lsq = least_squares(residual, np.interp(z, clima.z*1e3, o3_a), 
                                bounds=(-np.inf, np.inf), verbose=0, 
#                                max_nfev=3, #temp
                                args=(np.interp(z, clima.z*1e3, T_a), 
                                      np.interp(z, clima.z*1e3, m_a), 
                                      z, sol_zen, 
                                      gA(np.interp(z, clima.z*1e3, p_a), sol_zen), 
                                      o2delta_meas))
        
        o3_iris = res_lsq.x
        
#    except:
#        pass

        return x, xa, np.diag(Sm), mr, o3_iris, o3_a #temp

#with Pool(processes=4) as pool:
#    result = np.array(pool.map(f, range(len(day_mjd_lst)))) 

result = []
for i in range(3):
    result.append(f(i))



#%% organize resulting arrays and save in nc file
result_1d = xr.DataArray(np.stack(result[:,0]), 
                         coords=(day_mjd_lst, z), 
                         dims=('mjd', 'z'), name='VER', attrs={'units': 'photons cm-3 s-1'})
da_o3_a = xr.DataArray(np.stack(result[:,5]), 
                       coords=(day_mjd_lst, clima.z*1e3), 
                       dims=('mjd', 'clima_z'), name='O3 apriori', attrs={'units': 'molecule cm-3'})
ds = xr.Dataset({'ver': result_1d, 
                 'ver_apriori': (['mjd', 'z'], np.stack(result[:,1]), {'units': 'photons cm-3 s-1'}),
                 'ver_error':(['mjd', 'z'], np.stack(result[:,2]), {'units': '(photons cm-3 s-1)**2'}), 
                 'mr':(['mjd', 'z'], np.stack(result[:,3])), 
                 'o3':(['mjd', 'z'], np.stack(result[:,4]), {'units': 'molecule cm-3'}),
                 'o3_apriori': da_o3_a,
                 'sza':('mjd', sza.sel(mjd=day_mjd_lst).values),
                 'lst':('mjd', lst.sel(mjd=day_mjd_lst).values),
                 'longitude':('mjd', longitude.sel(mjd=day_mjd_lst).values),
                 'latitude':('mjd', latitude.sel(mjd=day_mjd_lst).values)})
path = '/home/anqil/Documents/osiris_database/iris_ver_o3/'
#ds.to_netcdf(path+'ver_o3_{}.nc'.format(orbit))
    
#%%
from matplotlib.colors import LogNorm

dds = xr.open_dataset('test_ds.nc')
dds['mjd']=np.arange(len(dds.mjd))
dds.z.assign_attris({'units': 'm'})

#%%
imjd=[100, 600, 1100]
fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(15,5),
                       gridspec_kw={'width_ratios': [3, 1]})
dds.ver.where(dds.mr>0.8).rename({'mjd':'index'}).plot(y='z', ax=ax[0], cmap='viridis', 
             norm=LogNorm(vmin=1e4, vmax=2e7),  add_colorbar=False)
dds.ver.where(dds.mr>0.8).isel(mjd=imjd).rename({'mjd':'index'}).plot.line(y='z', 
             ls='-', marker='.', ax=ax[1], xscale='log')
ax[0].set(ylim=(60e3, 100e3))
ax[1].set(xlim=(1e4, 2e7))
for i in imjd:
    ax[0].axvline(x=dds.mjd[i], color='k')

#%% single profile ozone compareison
#% load smr
path = '/home/anqil/Documents/osiris_database/ex_data/'
file = 'smr_2008_sza_interp.nc'
smr = xr.open_dataset(path+file)
smr = smr.where((smr.mjd>mjd[0]) & (smr.mjd<mjd[-1]), drop=True)

# load os
path = '/home/anqil/Documents/osiris_database/odin-osiris.usask.ca/Level2/CCI/OSIRIS_v5_10/'
file = 'ESACCI-OZONE-L2-LP-OSIRIS_ODIN-SASK_V5_10_HARMOZ_ALT-{}{}-fv0002.nc'
os = xr.open_dataset(path+file.format(str(num2date(mjd[0], units).year), 
                                      str(num2date(mjd[0], units).month).zfill(2)))