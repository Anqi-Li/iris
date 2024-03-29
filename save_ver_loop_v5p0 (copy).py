#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:10:51 2019

@author: anqil
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
#def residual(o3, T, m, z, zenithangle, p, o2delta_meas):
#    o2delta_model = cal_o2delta_new(o3, T, m, z, zenithangle, p)[0] 
##    print(id(o2delta_model))
##    print(round(o3[30]*1e-6))
##    plt.semilogx(o3, z) #temp
##    plt.show()
#    return o2delta_meas - o2delta_model

#%%
def rel_avk(xa, A):
    #xa 1D array lenth n
    #A 2D array size (n, n)
    A_rel = np.zeros(A.shape)
    for i in range(len(xa)):
        for j in range(len(xa)):
            A_rel[i, j] = xa[j] * A[i, j] / xa[i]
    return A_rel

#%%
def f(i):

        alt_chop_cond = (altitude.isel(mjd=i)>bot) & (altitude.isel(mjd=i)<top) #select altitude range 
        if alt_chop_cond.sum() < 2:
            print('wrong alt range ({}), month {}'.format(i, month))
            pass
        elif l1.isel(mjd=i).notnull().sum() == 0:
            print('l1 is all nan ({}), month {}'.format(i, month))
            pass
        elif error.isel(mjd=i).notnull().sum() == 0:
            print('error is all nan ({}), month {}'.format(i, month))
            pass
        
        else:
            try:
                print(i, 'out of', len(day_mjd_lst), 'in month ', month)
#                print('get VER')
        
                o3_a = o3_clima.sel(lst=lst.isel(mjd=i),
                                    lat=latitude.isel(mjd=i), 
                                    method='nearest')
                T_a = clima.T.sel(lat=latitude.isel(mjd=i), 
                                  method='nearest')
                m_a = clima.m.sel(lat=latitude.isel(mjd=i),
                                  method='nearest')
                p_a = clima.p.sel(lat=latitude.isel(mjd=i), 
                                  method='nearest') 
        
                sol_zen = sza[i].item()
                o2delta_a = cal_o2delta_new(o3_a.values, T_a.values, m_a.values, 
#                                            z,
                                            clima.z.values*1e3,
                                            sol_zen, p_a.values)[0]
                
#                xa = o2delta_a * A_o2delta
                xa = np.interp(z*1e-3, clima.z, o2delta_a) * A_o2delta
                Sa = np.diag((xa*0.75)**2)
#                Sa = np.diag(xa**2)
                h = altitude.isel(mjd=i).where(l1.isel(mjd=i).notnull(), drop=True
                                  ).where(alt_chop_cond, drop=True)
                K = pathl1d_iris(h, z, z_top) * 1e2 # m-->cm    
                y = l1.isel(mjd=i).sel(pixel=h.pixel).data * normalize    
#                y = yy.copy()
#                y[y<0] = 1e9
                Se = np.diag((error.isel(mjd=i).sel(pixel=h.pixel).values * normalize)**2)
#                Se = np.diag(np.ones(len(y))* (1e11)**2)
                x, A, Ss, Sm = linear_oem(K, Se, Sa, y, xa)
                y_fit = K.dot(x)
                cost_x = (x-xa).T.dot(np.linalg.inv(Sa)).dot(x-xa)
                cost_y = (y-y_fit).T.dot(np.linalg.inv(Se)).dot(y-y_fit)
                mr = A.sum(axis=1)
                A_rel = rel_avk(xa, A)
                mr_rel = A_rel.sum(axis=1)
                
                y_fit = xr.DataArray(y_fit, coords={'pixel': h.pixel}, 
                                     dims='pixel').reindex(pixel=l1.pixel)/normalize                     
                
                return (x, xa, np.diag(Sm), np.diag(Ss), mr, mr_rel, 
                        o3_a.values, day_mjd_lst[i].values, 
                        y_fit, cost_x, cost_y, Sm, Ss, A_rel)
         
            except:
                print('something is wrong ({})'.format(i))

                raise
                pass

                
#%%
if __name__ == '__main__': 
    
    file = '/home/anqil/Documents/osiris_database/ir_stray_light_corrected.nc'
    ir = xr.open_dataset(file)
    #orbit = 37580
    orbit_index = 45# 105#45
    ir = ir.where(ir.orbit==np.unique(ir.orbit)[orbit_index], drop=True)
    #print(num2date(ir.mjd,units))
    ir_fullorbit = ir
    day_mjd_lst = ir.mjd[ir.sza<90]

    #%%
#    year = [2008, 2008]
#    month = [1, 2]
#
#    t_bounds = Time(['{}-{}-01T00:00:00'.format(year[0], str(month[0]).zfill(2)),
#                   '{}-{}-01T00:00:00'.format(year[1], str(month[1]).zfill(2))], 
#                    format='isot', scale='utc')
#    #remote_file = 'https://arggit.usask.ca/opendap/ir_stray_light_corrected.nc'
#    file = '/home/anqil/Documents/osiris_database/ir_stray_light_corrected.nc'
#    ir = xr.open_dataset(file)
#    mjd_lst = ir.mjd[np.logical_and(ir.mjd>=t_bounds.mjd[0], ir.mjd<t_bounds.mjd[1])]
#    ir = ir.sel(mjd=mjd_lst)
#    day_mjd_lst = ir.mjd[ir.sza<90]
    ir = ir.sel(mjd=day_mjd_lst)
    altitude = ir.altitude.sel(pixel=slice(14, 128))
    latitude = ir.latitude
    longitude = ir.longitude
    sza = ir.sza
    lst = ir.apparent_solar_time.sel(pixel=60).drop('pixel')
    l1 = ir.sel(pixel=slice(14, 128)).data
    error = ir.sel(pixel=slice(14, 128)).error
    month = num2date(ir.mjd[0], units).month
    
    #%% load climatology
    path = '/home/anqil/Documents/osiris_database/ex_data/'
    file = 'msis_cmam_climatology_z200_lat8576.nc'
    clima = xr.open_dataset(path+file)#.interp(z=z*1e-3)
    clima = clima.update({'m':(clima.o + clima.o2 + clima.n2)*1e-6}) #cm-3
#    clima = clima.sel(month=month[0])
    clima = clima.sel(month = num2date(day_mjd_lst, units)[0].month)
    o3_clima = clima.o3_vmr * clima.m #cm-3
    
 #%% Global variables
    A_o2delta = 2.23e-4
    fr = 0.72 # filter fraction 
    normalize = np.pi*4 / fr
    
    # define oem inversion grid
    #====drop data below and above some altitudes
    bot = 61e3
    top = 100e3
#    top_extra = 110e3 # for apriori, cal_o2delta (J and g)
    #====retireval grid
    z = np.arange(50e3, 130e3, 1e3) # m
    z_top = z[-1] + 1e3 # for jacobian

#%%
#    result = []
#    for i in range(len(day_mjd_lst)): 
#        result.append(f(i))
    index_lst = [609]#[570]
#    index_lst = [251, 261, 8962, 13716, 24600]
#    index_lst = [2190,  2192, 2194, 3313, 9349, 10562, 10662, 21947]
#    index_lst = [56, 57]
#    index_lst = list(range(256,260))
#    index_lst = [abs(day_mjd_lst - 54467.649017).argmin().item()]
    result = []
    for i in index_lst:
        result.append(f(i))

#   organize resulting arrays and save in nc file
    result = [i for i in result if i] # filter out all None element in list
    mjd = np.stack([result[i][7] for i in range(len(result))])
    result_1d = xr.DataArray(np.stack([result[i][0] for i in range(len(result))]), 
                             coords=(mjd, z*1e-3), 
                             dims=('mjd', 'z'), name='VER', attrs={'units': 'photons cm-3 s-1'})
    limb_fit = xr.DataArray(np.stack([result[i][8] for i in range(len(result))]),
                            coords=(mjd, l1.pixel),
                           dims=('mjd', 'pixel'), name='Limb Radiance fitted')
    o3_apriori = xr.DataArray(np.stack([result[i][6] for i in range(len(result))]),
                            coords=(mjd, clima.z), 
                            dims=('mjd', 'clima_z'), name='climatology ozone', attrs={'units': 'photons cm-3 s-1'})
#    oem_avk = xr.DataArray(np.stack([result[i][5] for i in range(len(result))]),
#                           )
    ds = xr.Dataset({'ver': result_1d, 
                     'ver_apriori': (['mjd', 'z'], 
                                     np.stack([result[i][1] for i in range(len(result))]), 
                                     {'units': 'photons cm-3 s-1 squared'}),
                     'ver_error':(['mjd', 'z'], 
                                  np.stack([result[i][2] for i in range(len(result))]), 
                                  {'units': 'photons cm-3 s-1 squared'}),
                     'ver_smoothing':(['mjd','z'], np.stack([result[i][3] for i in range(len(result))])),
                     'mr':(['mjd', 'z'], 
                           np.stack([result[i][4] for i in range(len(result))])), 
                     'mr_rel':(['mjd', 'z'], 
                           np.stack([result[i][5] for i in range(len(result))])), 
                     'o3_apriori': o3_apriori, 
                     'sza': sza.sel(mjd=mjd),
                     'lst': lst.sel(mjd=mjd),
                     'longitude': longitude.sel(mjd=mjd),
                     'latitude': latitude.sel(mjd=mjd),
                     'limb_fit': limb_fit,
                     'cost_ver': (['mjd'], np.stack([result[i][9] for i in range(len(result))])),
                     'cost_limb':(['mjd'], np.stack([result[i][10] for i in range(len(result))]))
                     })
                     
#%% saveing to nc file
#    path = '/home/anqil/Documents/osiris_database/iris_ver_o3/'
#    filename = 'ver_{}{}_v5p0.nc'
#    ds.to_netcdf(path+filename.format(year[0], str(month[0]).zfill(2)))

#%%
#    alts_interp = np.arange(10e3, 150e3, .25e3)
    alts_interp = z#np.arange(bot, top, 1e3)
    data_interp = []
    fit_interp = []
    error_interp = []
    for (err, data, fit, alt) in zip(error.isel(mjd=index_lst), 
         l1.isel(mjd=index_lst), ds.limb_fit, altitude.isel(mjd=index_lst)):
        f_int = interp1d(alt, data, bounds_error=False)
        data_interp.append(f_int(alts_interp))
        f_int = interp1d(alt, fit, bounds_error=False)
        fit_interp.append(f_int(alts_interp))
        f_int = interp1d(alt, err, bounds_error=False)
        error_interp.append(f_int(alts_interp))
    data_interp = xr.DataArray(data_interp, coords=[ds.mjd, alts_interp], 
                              dims=['mjd', 'z'])  
    fit_interp = xr.DataArray(fit_interp, coords=[ds.mjd, alts_interp], 
                              dims=['mjd', 'z'])
    error_interp = xr.DataArray(error_interp, coords=[ds.mjd, alts_interp], 
                              dims=['mjd', 'z'])
    
    #%%
#% plotting things
    plt.rcParams.update({'font.size': 15, 'figure.figsize': (8,5)})
    figdir = '/home/anqil/Desktop/'
    
    fig = plt.figure()
    data_interp.plot.line(y='z', xscale='log', add_legend=False, color='k')
    fit_interp.plot.line(y='z',  xscale='log', add_legend=True, ls='', marker='.')
    plt.ylim([z[0], z[-1]])
    plt.xlim([0, 1e13])
    plt.title('fitted and original(k) limb profiles in z space')
#    fig.savefig(figdir+'limb_z.png')
    
#    color = ['C{}'.format(i) for i in range(5)] 
#    fig = plt.figure()
##    measured_o2.plot.line(y='z', add_legend=False, color='k', xscale='log')
#    [plt.plot(data_interp[i], alts_interp, color=color[i]) for i in range(5)]
#    [plt.plot(fit_interp[i], alts_interp,'.', color=color[i]) for i in range(5)]
#    plt.ylim([60e3, 100e3])
#    plt.xscale('log')
    
#    fig = plt.figure()
#    (data_interp - fit_interp).plot.line(y='z')
#    error_interp.plot.line(y='z', color='k')
#    plt.title('y-Kx and error from calibraion')
#    plt.ylim([bot, top])
    color = ['C{}'.format(i) for i in range(5)] 
    fig = plt.figure()
    [plt.plot((data_interp - fit_interp)[i], alts_interp, color=color[i]) for i in range(len(ds.mjd))]
    [plt.plot((error_interp)[i], alts_interp, ls='--', color=color[i]) for i in range(len(ds.mjd))]    
    [plt.plot(-(error_interp)[i], alts_interp, ls='--', color=color[i]) for i in range(len(ds.mjd))]  
    plt.title('y-Kx (solid) and error from calibraion (dashed)')


#%%
#    fig = plt.figure()
#    ds.mr.plot.line(y='z')
#    plt.title('absolute measurement reponse')

#%%
    Ss = result[0][-2]
    Sm = result[0][-3]
    fig = plt.figure()
    plt.semilogx(np.sqrt(np.diag(Ss)), ds.z, np.sqrt(np.diag(Sm)), ds.z)
    plt.legend(['smoothing error', 'retrieval noise'])
    plt.title('diag element of Ss and Sm (sqrt)')
    plt.xlabel('VER unit')
    fig.savefig('/home/anqil/Documents/reportFigure/article2/iris_Ss_Sm_sample.png',
            bbox_inches = "tight")   

    fig = plt.figure()
    plt.plot(np.diag(Sm, k=0),label='k=0')
    plt.plot(np.diag(Sm, k=1), label='k=1')
    plt.plot(np.diag(Sm, k=2),  label='k=2')
    plt.plot(np.diag(Sm, k=3),  label='k=3')
    plt.legend()
    plt.title('diagonal elements of Sm')
    plt.ylabel('VER unit square')
    plt.xlabel('element index')
    fig.savefig('/home/anqil/Documents/reportFigure/article2/iris_Sm_matrix_sample.png',
            bbox_inches = "tight")
    
#%%
    fig = plt.figure()
    (np.sqrt(ds.ver_error)/ds.ver).where(ds.mr_rel>0.8).plot(y='z')
    plt.title('error/ver in percentage')
    plt.legend(['where $MR^{frac}$ > 0.8'])
    fig.savefig('/home/anqil/Documents/reportFigure/article2/iris_error_percent_sample.png',
            bbox_inches = "tight")

#%%
    AVK_rel = result[0][-1]
    fig = plt.figure()
    mr = ds.mr_rel.plot.line(y='z', add_legend=False, color='k', ls='--')
    below = plt.plot(AVK_rel[:50:2, :].T, ds.z, color='C1')
    above = plt.plot(AVK_rel[50::2, :].T, ds.z, color='C2')
#    plt.title('relative averaging kernel, mesurement response')
    plt.ylabel('Altitude / km')
    plt.xlabel('AVK or MR')
#    plt.legend(['$MR^{frac}$'] + ['$AVKs^{frac}$'])
    plt.legend([mr[0], below[0], above[0]], ['$MR^{frac}$', '$AVK^{frac}$ below 100 km', '$AVK^{frac}$ above 100 km'])
#    plt.legend(['$MR_{frac}$'] + ['{} km'.format(i) for i in (ds.z.values[::10]).astype(int)])
    plt.title('')
    fig.savefig('/home/anqil/Documents/reportFigure/article2/iris_ver_AVK_MR_sample.png',
            bbox_inches = "tight")   
    
#%%
    from scipy.signal import chirp, find_peaks, peak_widths
    P, W = [], []
    for i in range(len(AVK_rel)):
        x = AVK_rel[i,:]
        peaks, _ = find_peaks(x, threshold=0.1)
        results_half = peak_widths(x, peaks, rel_height=0.5)
        if len(peaks) == 0:
            P.append(np.nan)
        else:
            P.append(peaks.item())
        if len(results_half[0]) == 0:
            W.append(np.nan)
        else:
            W.append(results_half[0].item())
            
#%%
    fig = plt.figure()
    ds.ver_apriori.plot.line(y='z', add_legend=False, ls='-', xscale='log', color='k')
    ds.ver.plot.line(y='z', marker='.', ls='-', xscale='log')
    plt.title('retrieved VER vs apriori profiles')
#    plt.ylim([z[0]*1e-3, z[-1]*1e-3])
#    plt.ylim([60, 100])
#    fig.savefig(figdir+'ver.png')
    
#    ver_test = ds.ver.copy()
#    ver_test[ver_test<0] = ds.ver_apriori[ver_test<0]/1000
#    ver_test.plot.line(y='z', marker='.', xscale='log')
