#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 17:16:30 2019

@author: anqil
"""

import numpy as np
import xarray as xr
import matplotlib.pylab as plt
from osirisl1services.readlevel1 import open_level1_ir
from osirisl1services.services import Level1Services
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
from netCDF4 import num2date
units = 'days since 1858-11-17 00:00:00.000'
from astropy.time import Time
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


def residual(o3, T, m, z, zenithangle, gA, o2delta_meas):
    o2delta_model = cal_o2delta(o3, T, m, z, zenithangle, gA)[0]  
#    plt.semilogx(o3, z) #temp
#    plt.show()
    return o2delta_meas - o2delta_model


#%% Global variables
A_o2delta = 2.23e-4
fr = 0.72 # filter fraction 
normalize = np.pi*4 / fr
# define oem inversion grid
#====drop data below and above some altitudes
bot = 60e3
top = 102e3
top_extra = 149e3 # for apriori, cal_o2delta (J and g)
#====retireval grid
z = np.arange(bot, top, 1e3) # m
z = np.append(z, top_extra)
z_top = z[-1] + 1e3 # for jacobian

#%%
year = [2007, 2007]
month = [11, 12]
t_bounds = Time(['{}-{}-01T00:00:00'.format(year[0], str(month[0]).zfill(2)),
               '{}-{}-01T00:00:00'.format(year[1], str(month[1]).zfill(2))], 
                format='isot', scale='utc')
remote_file = 'https://arggit.usask.ca/opendap/ir_stray_light_corrected.nc'
ir = xr.open_dataset(remote_file)
mjd_lst = ir.mjd[np.logical_and(ir.mjd>=t_bounds.mjd[0], ir.mjd<t_bounds.mjd[1])]
ir = ir.sel(mjd=mjd_lst)
day_mjd_lst = ir.mjd[ir.sza<90]
ir = ir.sel(mjd=day_mjd_lst)
altitude = ir.altitude.sel(pixel=slice(14, 128))
latitude = ir.latitude
longitude = ir.longitude
sza = ir.sza
lst = ir.apparent_solar_time.sel(pixel=60)
l1 = ir.sel(pixel=slice(14, 128)).data
error = ir.sel(pixel=slice(14, 128)).error

#%% load climatology
path = '/home/anqil/Documents/osiris_database/ex_data/'
file = 'msis_cmam_climatology.nc'
clima = xr.open_dataset(path+file)#.interp(z=z*1e-3)
clima = clima.update({'m':(clima.o + clima.o2 + clima.n2)*1e-6}) #cm-3
clima = clima.sel(month=month[0])
o3_clima = clima.o3_vmr * clima.m #cm-3

#%%
def f(i):
    try:
        print(i, 'out of', len(day_mjd_lst))
        
        o3_a = o3_clima.interp(lat=latitude.isel(mjd=i),
                               lst_bins=lst.isel(mjd=i), z=z*1e-3)
        T_a = clima.T.interp(lat=latitude.isel(mjd=i), z=z*1e-3)
        m_a = clima.m.interp(lat=latitude.isel(mjd=i), z=z*1e-3)
        p_a = clima.p.interp(lat=latitude.isel(mjd=i), z=z*1e-3) 

        sol_zen = sza[i].item()
        o2delta_a = cal_o2delta(o3_a.data, T_a.data, m_a.data, z,
                                sol_zen, gA(p_a, sol_zen))[0]
        xa = o2delta_a * A_o2delta
        Sa = np.diag(xa**2)
        alt_chop_cond = (altitude.isel(mjd=i)>bot) & (altitude.isel(mjd=i)<top)
        h = altitude.isel(mjd=i).where(l1.isel(mjd=i).notnull(), drop=True
                          ).where(alt_chop_cond, drop=True)
        K = pathl1d_iris(h, z, z_top) * 1e2 # m-->cm    
        y = l1.isel(mjd=i).sel(pixel=h.pixel).data * normalize
        Se = np.diag((error.isel(mjd=i).sel(pixel=h.pixel).data * normalize)**2)
        x, A, Ss, Sm = linear_oem(K, Se, Sa, y, xa)
        mr = A.sum(axis=1)
        
        #lsq fit to get ozone
        o2delta_meas = x / A_o2delta # cm-3
        res_lsq = least_squares(residual, o3_a.data,
                                bounds=(-np.inf, np.inf), verbose=0, 
#                                max_nfev=3, #temp
                                args=(T_a.data,
                                      m_a.data,
                                      z, sol_zen, 
                                      gA(p_a, sol_zen),
                                      o2delta_meas))
        o3_iris = res_lsq.x
        
    except:
        pass

    return x, xa, np.diag(Sm), mr, o3_iris, o3_a.data, day_mjd_lst[i].data

#%%
if __name__ == '__main__': 
    with Pool(processes=4) as pool:
        result = np.array(pool.map(f, range(len(day_mjd_lst)))) 
    
#    result = []
#    for i in range(4):
#        result.append(f(i))

# organize resulting arrays and save in nc file
    result_1d = xr.DataArray(np.stack(result[:,0]), 
                             coords=(np.stack(result[:,6]), z), 
                             dims=('mjd', 'z'), name='VER', attrs={'units': 'photons cm-3 s-1'})

    ds = xr.Dataset({'ver': result_1d, 
                     'ver_apriori': (['mjd', 'z'], np.stack(result[:,1]), {'units': 'photons cm-3 s-1'}),
                     'ver_error':(['mjd', 'z'], np.stack(result[:,2]), {'units': '(photons cm-3 s-1)**2'}), 
                     'mr':(['mjd', 'z'], np.stack(result[:,3])), 
                     'o3':(['mjd', 'z'], np.stack(result[:,4]), {'units': 'molecule cm-3'}),
                     'o3_apriori':(['mjd', 'z'], np.stack(result[:,5]), {'units': 'molecule cm-3'}),
                     'sza':('mjd', sza.sel(mjd=np.stack(result[:,6])).values),
                     'lst':('mjd', lst.sel(mjd=np.stack(result[:,6])).values),
                     'longitude':('mjd', longitude.sel(mjd=np.stack(result[:,6])).values),
                     'latitude':('mjd', latitude.sel(mjd=np.stack(result[:,6])).values)})
    path = '/home/anqil/Documents/osiris_database/iris_ver_o3/'
    ds.to_netcdf(path+'ver_o3_{}{}.nc'.format(year[0], str(month[0]).zfill(2)))

