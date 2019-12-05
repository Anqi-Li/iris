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
#                print('get ozone')
                #lsq fit to get ozone
                o2delta_meas = x / A_o2delta # cm-3
#                o2delta_meas[o2delta_meas<0] = xa[o2delta_meas<0]/ A_o2delta / 1000    
#                o2delta_meas[o2delta_meas<0]
                
#                if (o2delta_meas<0).sum() > 0:
#                    print('negative value found')
#                    z_neg_1 = z[o2delta_meas<0].min()-1
#                    f_interp = interp1d(np.append(z[z<z_neg_1], z[z>115e3]),
#                                    np.log(np.append(o2delta_meas[z<z_neg_1],
#                                                     xa[z>115e3]/A_o2delta)))
#                    o2delta = np.exp(f_interp(z))
#                else:
                o2delta = o2delta_meas
                res_lsq = least_squares(residual,
#                                        o3_a.values[mr>0.8], #initial guess
                                        o3_a.interp(z=z*1e-3).values,
                                        method='trf',
#                                        xtol = 1e-8,
#                                        bounds=(1e-1, np.inf),
#                                        jac='3-point',
#                                        diff_step=1e-12,
#                                        ftol=1e-12,
#                                        xtol = 1e-14,
                                        verbose=2, 
#                                        loss='soft_l1', #'cauchy'?
#                                        tr_solver='exact',
#                                        max_nfev=30, #temp
                                        args=(T_a.interp(z=z*1e-3).values,
                                              m_a.interp(z=z*1e-3).values,
                                              z,  #in meter
                                              sol_zen, 
                                              p_a.interp(z=z*1e-3).values,
                                              o2delta))
#                o3_iris = xr.DataArray(res_lsq.x, coords={'z': z_mr}, dims='z').reindex(z=z).data
#                resi = xr.DataArray(res_lsq.fun, coords={'z': z_mr}, dims='z').reindex(z=z).data
                o3_iris = res_lsq.x
                resi = res_lsq.fun
                return (x, xa, np.diag(Sm), mr, o3_iris, o3_a.values, 
                        day_mjd_lst[i].values, res_lsq.status, res_lsq.nfev,
                        resi, np.diag(Ss), y_fit, cost_x, cost_y, res_lsq.cost,
                        A_rel, mr_rel, o2delta, res_lsq.jac)
         
            except:
                print('something is wrong ({})'.format(i))

                raise
                pass

                
#%%
if __name__ == '__main__': 
    
    #%%
    year = [2008, 2008]
    month = [10, 11]

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
    bot = 61e3
    top = 100e3
#    top_extra = 110e3 # for apriori, cal_o2delta (J and g)
    #====retireval grid
    z = np.arange(50e3, 102e3, 1e3) # m
    z_top = z[-1] + 1e3 # for jacobian

#%%

#    result = []
#    for i in range(10): #(len(day_mjd_lst)): 
#        result.append(f(i))
    
    index_lst = [8962]#[251, 261, 8962, 13716, 24600]
#    index_lst = [2190,  2192, 2194, 3313, 9349, 10562, 10662, 21947]
#    index_lst = [56, 57]
#    index_lst = list(range(256,260))
    result = []
    for i in index_lst:
        result.append(f(i))

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
#    oem_avk = xr.DataArray(np.stack([result[i][5] for i in range(len(result))]),
#                           )
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
                     'cost_lsq': (['mjd'], np.stack([result[i][14] for i in range(len(result))])),
                     'mr_rel': (['mjd','z'], np.stack([result[i][16] for i in range(len(result))])),
                     'o2delta_lsq' : (['mjd', 'z'], np.stack([result[i][17] for i in range(len(result))]))
                     })
                     

#%% saveing to nc file
#    path = '/home/anqil/Documents/osiris_database/iris_ver_o3/'
#    filename = '{}{}_v5p0_test.nc'
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
    plt.rcParams.update({'font.size': 12, 'figure.figsize': (10,5)})
    figdir = '/home/anqil/Desktop/'
    
    fig = plt.figure()
    data_interp.plot.line(y='z', xscale='log', add_legend=False, color='k')
    fit_interp.plot.line(y='z',  xscale='log', add_legend=True, ls='', marker='.')
    plt.ylim([z[0], z[-1]])
#    plt.xlim([0, 1e13])
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
    
    fig = plt.figure()
    ds.mr.plot.line(y='z')
    plt.title('absolute measurement reponse')

#%%
    fig = plt.figure()
    AVK_rel = result[0][15]
#    np.argmax(AVK_rel,axis=1)
    MR_rel = result[0][16]
    plt.plot(AVK_rel, z)
    plt.plot(MR_rel, z)
    plt.title('relative averaging kernel, mesurement response')
#    
#%%
    fig = plt.figure()
    
    lsq_jac = result[0][18]
    plt.plot(lsq_jac[z==78e3].squeeze().T, z)
    plt.xlim([-1e-10, 1e-10])
    plt.show()

#%%
    i=slice(0,len(index_lst))
    fig = plt.figure()
    ds.ver_apriori.isel(mjd=i).plot.line(y='z', add_legend=False, ls='-', xscale='log', color='k')
    ds.ver.isel(mjd=i).plot.line(y='z', marker='.', ls='-', xscale='log')
    
#    (ds.o2delta_lsq.isel(mjd=i)*A_o2delta).plot.line(y='z', marker='.', ls='-')#, xscale='linear')
    plt.title('retrieved VER vs apriori profiles')
#    plt.plot(MR_rel*1e8, z*1e-3)
#    plt.ylim([z[0]*1e-3, z[-1]*1e-3])
#    plt.ylim([75, 95])
#    fig.savefig(figdir+'ver.png')
#    ax = plt.gca().twiny()
#    np.log(ds.ver.isel(mjd=i)).differentiate('z').plot.line(
#            y='z',marker='.', color='r', ax=ax)

#%% check interped and original profiles of o2delta
#    color = ['C{}'.format(i) for i in range(len(index_lst))] 
#    o2delta_interp = ds.o2delta_lsq
#    o2delta_org = ds.ver/A_o2delta
#    fig = plt.figure()
##    o2delta_org.plot.line(y='z', xscale='log')
##    o2delta_interp.plot.line(y='z', marker='.')
#    [plt.semilogx(o2delta_interp[i], o2delta_interp.z, '--', color=color[i]) for i in range(len(index_lst))]
#    [plt.semilogx(o2delta_org[i], o2delta_org.z, '-', color=color[i]) for i in range(len(index_lst))]
#
#    plt.ylim([70, 110])

#%% 
    measured_o2 = ds.o2delta_lsq #ds.ver / A_o2delta
    
#    fitted_o2 = measured_o2 - ds.lsq_residual
    fitted_o2 = []
    for i in range(len(index_lst)):
        fitted_o2.append(cal_o2delta_new(ds.o3.isel(mjd=i).values, 
                        clima.T.sel(lat=latitude.isel(mjd=i), 
                                    method='nearest').interp(z=z*1e-3).values, 
                        clima.m.sel(lat=latitude.isel(mjd=i), 
                                    method='nearest').interp(z=z*1e-3).values, 
                        z, ds.sza[i].values, 
                        clima.p.sel(lat=latitude.isel(mjd=i), 
                                    method='nearest').interp(z=z*1e-3).values)[0])
    fitted_o2 = np.array(fitted_o2)
    
    i = slice(0,len(index_lst))#0
    fig = plt.figure()
    measured_o2.isel(mjd=i).plot.line(y='z', add_legend=False, color='k', xscale='log')
#    fitted_o2.plot.line(y='z', xscale='log', ls='', marker='.')
    plt.plot(fitted_o2[i].T, z*1e-3, '.')    
#    (ds.ver_error**.5).plot.line(y='z', add_legend=False, xscale='log', color='k',ls='--')
#    (fitted_o2 + 1e3*ds.ver_error**.5).plot.line(y='z', add_legend=False, xscale='log', color='k',ls='--')
#    (fitted_o2 - 1e3*ds.ver_error**.5).plot.line(y='z', add_legend=False, xscale='log', color='k',ls='--')
    plt.title('o2d_meas vs o2d_fitted (measured - residual)')
#    plt.ylim([bot*1e-3, top*1e-3])
#    plt.ylim(z[0]*1e-3, z[-1]*1e-3)
#    plt.ylim(70,85)
#    fig.savefig(figdir+'o2delta.png')



#%%
    fig = plt.figure()
    ds.o3_apriori.plot.line(y='clima_z', add_legend=False, ls='-', xscale='log', color='k')
    ds.o3.plot.line(y='z', marker='.', ls='-', xscale='log')
    plt.title('retrieved O3 vs apriori profiles')
#    plt.ylim([bot*1e-3, top*1e-3])
#    plt.xlim([-1e9, 1e9])
#    fig.savefig(figdir+'o3.png') 
    
#%% 
    fig = plt.figure()
    ds.lsq_residual.plot.line(y='z')
    plt.title('residual from non-linear fit')
    plt.ylim([z[0]*1e-3, z[-1]*1e-3])
    plt.xlim([-1e-1, 1e-1])
#    fig.savefig(figdir+'lsq_residual.png')

