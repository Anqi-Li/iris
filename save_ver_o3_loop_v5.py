#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:41:04 2019

@author: anqili
"""

import numpy as np
import numpy.ma as ma
import xarray as xr
#import sys
#import os
import matplotlib.pylab as plt
#from osirisl1services.readlevel1 import open_level1_ir
#from osirisl1services.services import Level1Services
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
from netCDF4 import num2date
units = 'days since 1858-11-17 00:00:00.000'
from astropy.time import Time
from scipy.optimize import least_squares
from o2delta_model import cal_o2delta_new
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


#%%
def residual(o3, T, m, z, zenithangle, p, o2delta_meas):
    o2delta_model = cal_o2delta_new(o3, T, m, z, zenithangle, p)[0] 
#    print(id(o2delta_model))
#    print(round(o3[30]*1e-6))
#    plt.semilogx(o3, z) #temp
#    plt.show()
    return o2delta_meas - o2delta_model

#%%
def f(i):

        alt_chop_cond = (altitude.isel(mjd=i)>bot) & (altitude.isel(mjd=i)<top) #select altitude range 
        if alt_chop_cond.sum() < 2:
            print('wrong alt range ({}), month {}'.format(i, month[0]))
            pass
        elif l1.isel(mjd=i).notnull().sum() == 0:
            print('l1 is all nan ({}), month {}'.format(i, month[0]))
            pass
        elif error.isel(mjd=i).notnull().sum() == 0:
            print('error is all nan ({}), month {}'.format(i, month[0]))
            pass
        
        else:
            try:
                print(i, 'out of', len(day_mjd_lst), 'in month ', month[0])
#                print('get VER')
        
                o3_a = o3_clima.sel(lst=lst.isel(mjd=i),
                                    lat=latitude.isel(mjd=i), 
#                                    z=z*1e-3,
                                    method='nearest')
                T_a = clima.T.sel(lat=latitude.isel(mjd=i), 
#                                  z=z*1e-3, 
                                  method='nearest')
                m_a = clima.m.sel(lat=latitude.isel(mjd=i),
#                                  z=z*1e-3, 
                                  method='nearest')
                p_a = clima.p.sel(lat=latitude.isel(mjd=i), 
#                                  z=z*1e-3, 
                                  method='nearest') 
        
                sol_zen = sza[i].item()
                o2delta_a = cal_o2delta_new(o3_a.values, T_a.values, m_a.values, 
#                                            z,
                                            clima.z.values*1e3,
                                            sol_zen, p_a.values)[0]
                
#                xa = o2delta_a * A_o2delta
                xa = np.interp(z*1e-3, clima.z, o2delta_a) * A_o2delta
                Sa = np.diag(xa**2)
                h = altitude.isel(mjd=i).where(l1.isel(mjd=i).notnull(), drop=True
                                  ).where(alt_chop_cond, drop=True)
                K = pathl1d_iris(h, z, z_top) * 1e2 # m-->cm    
                y = l1.isel(mjd=i).sel(pixel=h.pixel).data * normalize                
                Se = np.diag((error.isel(mjd=i).sel(pixel=h.pixel).values * normalize)**2)
                x, A, Ss, Sm = linear_oem(K, Se, Sa, y, xa)
                y_fit = K.dot(x)
                cost_x = (x-xa).T.dot(np.linalg.inv(Sa)).dot(x-xa)
                cost_y = (y-y_fit).T.dot(np.linalg.inv(Se)).dot(y-y_fit)
                mr = A.sum(axis=1)
                y_fit = xr.DataArray(y_fit, coords={'pixel': h.pixel}, 
                                     dims='pixel').reindex(pixel=l1.pixel)/normalize                     
#                print('get ozone')
                #lsq fit to get ozone
                o2delta_meas = x / A_o2delta # cm-3
                o2delta_meas[o2delta_meas<0] = xa[o2delta_meas<0]/ A_o2delta / 1000    
#                o2delta_meas[o2delta_meas<0] = 0
                z_mr = z[mr>0.8]
                res_lsq = least_squares(residual,
#                                        o3_a.values[mr>0.8], #initial guess
                                        o3_a.interp(z=z_mr*1e-3).values,
#                                        method='lm',
#                                        bounds=(0, np.inf), verbose=0, 
#                                        loss='cauchy', #'cauchy'?
#                                        max_nfev=3, #temp
                                        args=(T_a.interp(z=z_mr*1e-3).values,
                                              m_a.interp(z=z_mr*1e-3).values,
                                              z_mr,  #in meter
                                              sol_zen, 
                                              p_a.interp(z=z_mr*1e-3).values,
                                              o2delta_meas[mr>0.8]))
                o3_iris = xr.DataArray(res_lsq.x, coords={'z': z_mr}, dims='z').reindex(z=z).data
                resi = xr.DataArray(res_lsq.fun, coords={'z': z_mr}, dims='z').reindex(z=z).data
    
                return (x, xa, np.diag(Sm), mr, o3_iris, o3_a.values, 
                        day_mjd_lst[i].values, res_lsq.status, res_lsq.nfev,
                        resi, np.diag(Ss), y_fit, cost_x, cost_y, res_lsq.cost)
         
            except:
                print('something is wrong ({})'.format(i))

#                raise
                pass

                
#%%
if __name__ == '__main__': 
    
    #%%
    year = [2008, 2008]
    month = [7,8]

    t_bounds = Time(['{}-{}-01T00:00:00'.format(year[0], str(month[0]).zfill(2)),
                   '{}-{}-01T00:00:00'.format(year[1], str(month[1]).zfill(2))], 
                    format='isot', scale='utc')
    #remote_file = 'https://arggit.usask.ca/opendap/ir_stray_light_corrected.nc'
    file = '/home/anqil/Documents/osiris_database/ir_stray_light_corrected.nc'
    ir = xr.open_dataset(file)
    mjd_lst = ir.mjd[np.logical_and(ir.mjd>=t_bounds.mjd[0], ir.mjd<t_bounds.mjd[1])]
    ir = ir.sel(mjd=mjd_lst)
    day_mjd_lst = ir.mjd[ir.sza<90]
    ir = ir.sel(mjd=day_mjd_lst)
    altitude = ir.altitude.sel(pixel=slice(14, 128))
    latitude = ir.latitude
    longitude = ir.longitude
    sza = ir.sza
    lst = ir.apparent_solar_time.sel(pixel=60).drop('pixel')
    l1 = ir.sel(pixel=slice(14, 128)).data
    error = ir.sel(pixel=slice(14, 128)).error
    
    #%% load climatology
    path = '/home/anqil/Documents/osiris_database/ex_data/'
    file = 'msis_cmam_climatology_z200_lat8576.nc'
    clima = xr.open_dataset(path+file)#.interp(z=z*1e-3)
    clima = clima.update({'m':(clima.o + clima.o2 + clima.n2)*1e-6}) #cm-3
    clima = clima.sel(month=month[0])
    o3_clima = clima.o3_vmr * clima.m #cm-3
    
 #%% Global variables
    A_o2delta = 2.23e-4
    fr = 0.72 # filter fraction 
    normalize = np.pi*4 / fr
    
    # define oem inversion grid
    #====drop data below and above some altitudes
    bot = 63e3
    top = 100e3
    top_extra = 110e3 # for apriori, cal_o2delta (J and g)
    #====retireval grid
    z = np.arange(bot, top, 1e3) # m
    z = np.append(z, top_extra)
    z_top = z[-1] + 1e3 # for jacobian

#%%

    result = []
    for i in range(len(day_mjd_lst)): 
        result.append(f(i))
    
##    index_lst = [251, 261, 8962, 13716, 24600]
#    index_lst = [2190,  2192, 2194, 3313, 9349, 10562, 10662, 21947]
#    index_lst = [56, 57]
#    result = []
#    for i in index_lst:
#        result.append(f(i))

#   organize resulting arrays and save in nc file
    result = [i for i in result if i] # filter out all None element in list
    mjd = np.stack([result[i][6] for i in range(len(result))])
    result_1d = xr.DataArray(np.stack([result[i][0] for i in range(len(result))]), 
                             coords=(mjd, z*1e-3), 
                             dims=('mjd', 'z'), name='VER', attrs={'units': 'photons cm-3 s-1'})
    limb_fit = xr.DataArray(np.stack([result[i][11] for i in range(len(result))]),
                            coords=(mjd, l1.pixel),
                           dims=('mjd', 'pixel'), name='Limb Radiance fitted')
    o3_apriori = xr.DataArray(np.stack([result[i][5] for i in range(len(result))]),
                            coords=(mjd, clima.z), 
                            dims=('mjd', 'clima_z'), name='climatology ozone', attrs={'units': 'photons cm-3 s-1'})
    ds = xr.Dataset({'ver': result_1d, 
                     'ver_apriori': (['mjd', 'z'], 
                                     np.stack([result[i][1] for i in range(len(result))]), 
                                     {'units': 'photons cm-3 s-1'}),
                     'ver_error':(['mjd', 'z'], 
                                  np.stack([result[i][2] for i in range(len(result))]), 
                                  {'units': '(photons cm-3 s-1)**2'}), 
                     'mr':(['mjd', 'z'], 
                           np.stack([result[i][3] for i in range(len(result))])), 
                     'o3':(['mjd', 'z'], 
                           np.stack([result[i][4] for i in range(len(result))]), 
                           {'units': 'molecule cm-3'}),
#                     'o3_apriori':(['mjd', 'z'], 
#                                   np.stack([result[i][5] for i in range(len(result))]), 
#                                   {'units': 'molecule cm-3'}),
                     'o3_apriori': o3_apriori, 
                     'sza': sza.sel(mjd=mjd),
                     'lst': lst.sel(mjd=mjd),
                     'longitude': longitude.sel(mjd=mjd),
                     'latitude': latitude.sel(mjd=mjd),
                     'status': (['mjd'], np.stack([result[i][7] for i in range(len(result))]),
                                {'The reason for least_squares algorithm termination':
                                    '''
        -1 : improper input parameters status returned from MINPACK.

        0 : the maximum number of function evaluations is exceeded.

        1 : gtol termination condition is satisfied.

        2 : ftol termination condition is satisfied.

        3 : xtol termination condition is satisfied.

        4 : Both ftol and xtol termination conditions are satisfied.
                                '''}),
                     'nfev': (['mjd'], np.stack([result[i][8] for i in range(len(result))]),
                              {'long name': 'Number of function evaluations done.'}),
                     'lsq_residual':(['mjd','z'], np.stack([result[i][9] for i in range(len(result))])),
                     'ver_smoothing':(['mjd','z'], np.stack([result[i][10] for i in range(len(result))])),
                     'limb_fit': limb_fit,
                     'cost_ver': (['mjd'], np.stack([result[i][12] for i in range(len(result))])),
                     'cost_limb':(['mjd'], np.stack([result[i][13] for i in range(len(result))])),
                     'cost_lsq': (['mjd'], np.stack([result[i][14] for i in range(len(result))]))
                     })
                     

#%% saveing to nc file
    path = '/home/anqil/Documents/osiris_database/iris_ver_o3/'
    filename = '{}{}_v5.nc'
    ds.to_netcdf(path+filename.format(year[0], str(month[0]).zfill(2)))

#%%
#
#    alts_interp = np.arange(10e3, 150e3, .25e3)
#    data_interp = []
#    for (data, alt) in zip(ir.data.isel(mjd=index_lst), ir.altitude.isel(mjd=index_lst)):
#        f_int = interp1d(alt, data, bounds_error=False)
#        data_interp.append(f_int(alts_interp))
#    data_interp = xr.DataArray(data_interp, coords=[ir.mjd.isel(mjd=index_lst), 
#                                                    alts_interp], dims=['mjd', 'z'])
#    
#    fit_interp = []
#    for (data, alt) in zip(ds.limb_fit, ir.altitude.sel(mjd=ds.mjd, pixel=slice(14,128))):
#        f_int = interp1d(alt, data, bounds_error=False)
#        fit_interp.append(f_int(alts_interp))
#    fit_interp = xr.DataArray(fit_interp, coords=[ds.mjd, alts_interp], 
#                              dims=['mjd', 'z'])
#    
#    
#    
##% plotting things
#    plt.rcParams.update({'font.size': 20, 'figure.figsize': (12,6)})
#    figdir = '/home/anqil/Desktop/'
#    mjd_index_lst = ir.mjd[index_lst]
#    
#    fig = plt.figure()
##    data_interp.plot(y='z', robust=True, add_colorbar=False, norm=LogNorm(vmin=1e9, vmax=1e13), cmap='viridis')
#    data_interp.plot.line(y='z', add_legend=False, color='k')
#    fit_interp.plot.line(y='z', add_legend=True, ls='', marker='.')#, color='k')
##    plt.ylim([bot, top])
#    plt.title('fitted and original(k) limb profiles in z space')
##    fig.savefig(figdir+'limb_z.png')
#    
#    fig = plt.figure()
#    (data_interp - fit_interp).plot.line(y='z')
#    plt.title('y-Kx')
##    fig = plt.figure()
##    ir.data.sel(mjd=mjd_index_lst).plot.line(y='pixel', color='k', add_legend=False)
##    ds.limb_fit.plot.line(y='pixel',  marker='.', ls='', xscale='linear')#, color='k')
##    plt.title('fitted and original (k) limb profiles in pixel space')
###    fig.savefig(figdir+'limb_pixel.png')
#
#    measured_o2 = ds.ver / A_o2delta
#    fitted_o2 = measured_o2 - ds.lsq_residual
#    fig = plt.figure()
#    measured_o2.where(ds.mr>0.8).plot.line(y='z', add_legend=False, color='k', xscale='log')
#    fitted_o2.plot.line(y='z', xscale='log', ls='-', marker='.')
#    plt.title('o2delta_measured (VER/A,mr>0.8) vs o2delta_fitted (measured - residual)')
##    fig.savefig(figdir+'o2delta.png')
#    
#    fig = plt.figure()
#    ds.ver_apriori.plot.line(y='z', add_legend=False, ls='-', xscale='log', color='k')
#    ds.ver.plot.line(y='z', marker='.', ls='-', xscale='log')
#    plt.title('retrieved VER vs apriori profiles')
##    plt.ylim([60, 100])
##    fig.savefig(figdir+'ver.png')
#
#    fig = plt.figure()
#    ds.o3_apriori.plot.line(y='clima_z', add_legend=False, ls='-', xscale='log', color='k')
#    ds.o3.plot.line(y='z', marker='.', ls='-', xscale='log')
#    plt.title('retrieved O3 vs apriori profiles')
##    fig.savefig(figdir+'o3.png')    
#    
#    fig = plt.figure()
#    ds.lsq_residual.plot.line(y='z')
#    plt.title('residual from non-linear fit')
##    plt.xlim([-1e-1, 1e-1])
##    fig.savefig(figdir+'lsq_residual.png')
#    
#
#
#    plt.show()
#    
