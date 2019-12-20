#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 14:12:50 2019

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
A_o2delta = 2.23e-4

def residual(o3, T, m, z, zenithangle, p, o2delta_meas):
    o2delta_model = cal_o2delta_new(o3, T, m, z, zenithangle, p, 
                                    correct_neg_o3=False)[0] 
#    print(id(o2delta_model))
#    print(round(o3[30]*1e-6))
#    plt.semilogx(o3, z) #temp
#    plt.show()
    return o2delta_meas - o2delta_model

def fit_ozone(i):
    #data: xarray
    #clima: xarray
    try:
        print(i, '/', len(data.mjd), 'in month', month)
        o2delta_meas = data.ver.isel(mjd=i) / A_o2delta # cm-3
    #    o2delta_meas[o2delta_meas<0] = data.ver_apriori.isel(mjd=i).values[o2delta_meas<0]/ A_o2delta / 1000    
    #    o2delta_meas[o2delta_meas<0] = 0
        o3_a = o3_clima.sel(lst=data.lst.isel(mjd=i),
                            lat=data.latitude.isel(mjd=i), 
                            method='nearest')
        T_a = clima.T.sel(lat=data.latitude.isel(mjd=i), 
                          method='nearest')
        m_a = clima.m.sel(lat=data.latitude.isel(mjd=i),
                          method='nearest')
        p_a = clima.p.sel(lat=data.latitude.isel(mjd=i), 
                          method='nearest') 
        
        z_mr = data.z.where(data.mr_rel.isel(mjd=i)>0.8, drop=True
                            )#.where(data.ver.isel(mjd=i)>0, drop=True)
        res_lsq = least_squares(residual,
                                o3_a.interp(z=z_mr).values,
    #                            method='lm',
    #                            xtol = 1e-8,
#                                bounds=(0, np.inf), 
    #                            verbose=2, 
    #                           loss='cauchy', #'cauchy'?
    #                            max_nfev=10, #temp
                                args=(T_a.interp(z=z_mr).values,
                                      m_a.interp(z=z_mr).values,
                                      z_mr.values*1e3,  #in meter
                                      data.sza.isel(mjd=i).values.item(), 
                                      p_a.interp(z=z_mr).values,
                                      o2delta_meas.sel(z=z_mr).values))
    #    o3 = res_lsq.x
    #    resi = res_lsq.fun
        o3 = xr.DataArray(res_lsq.x, coords={'z': z_mr}, dims='z').reindex(z=data.z).data
        resi = xr.DataArray(res_lsq.fun, coords={'z': z_mr}, dims='z').reindex(z=data.z).data
    
        mjd = data.isel(mjd=i).mjd.values.item()
        return (mjd, o3, resi, res_lsq.status, res_lsq.nfev, 
                res_lsq.cost, o2delta_meas)
    except:
        pass

#%% load oem data
year = 2008
month = 6

path = '/home/anqil/Documents/osiris_database/iris_ver_o3/'
filenames = 'ver_{}{}_v5p0.nc'.format(year, str(month).zfill(2))
data = xr.open_dataset(path+filenames)

#%% load climatology
path = '/home/anqil/Documents/osiris_database/ex_data/'
file = 'msis_cmam_climatology_z200_lat8576.nc'
clima = xr.open_dataset(path+file)#.interp(z=z*1e-3)
clima = clima.update({'m':(clima.o + clima.o2 + clima.n2)*1e-6}) #cm-3
clima = clima.sel(month=month)
o3_clima = clima.o3_vmr * clima.m #cm-3
clima = clima.update({'o3': o3_clima})

#%%
#z_o3 = np.arange(60, 150, 1) #km

#%%
#index_lst = [251, 261, 8962, 13716, 24600]
#index_lst = [2190,  2192, 2194, 3313, 9349, 10562, 10662, 21947]
#index_lst = [56, 57]
#index_lst = list(range(256,258))
#index_lst = (np.random.uniform(0,len(data.mjd), 10)).astype(int)
result=[]
for i in range(len(data.mjd)):#index_lst:
    result.append(fit_ozone(i))
    
result = [i for i in result if i]    
mjd = np.stack([result[i][0] for i in range(len(result))])
o3_iris = xr.DataArray(np.stack([result[i][1] for i in range(len(result))]), 
                         coords=(mjd, data.z), 
                         dims=('mjd', 'z'), name='o3', attrs={'units': 'photons cm-3 s-1'})
ds = xr.Dataset({'o3': o3_iris, 
                 'lsq_residual': (['mjd', 'z'], 
                                  np.stack([result[i][2] for i in range(len(result))]), 
                                 {'units': 'photons cm-3 s-1'}),
                'status':(['mjd'], 
                                  np.stack([result[i][3] for i in range(len(result))])),
                'nfev':(['mjd'], 
                                  np.stack([result[i][4] for i in range(len(result))])),
                'cost_lsq':(['mjd'], 
                                  np.stack([result[i][5] for i in range(len(result))])),
                'o2delta':(['mjd', 'z'], 
                                  np.stack([result[i][6] for i in range(len(result))])),
                'datetime':(['mjd'], num2date(mjd,units))
                }, attrs={'ozone fitting':
                    '''correction on negative ozone in forward model, 
                        only select mr>0.8 in VER data.'''})
                
##%%
#plt.rcParams.update({'font.size': 14})
#fig, ax = plt.subplots(3, 1, sharey=True, figsize=(8,10))
#fig.tight_layout()
##(data.ver_apriori/A_o2delta).isel(mjd=index_lst).plot.line(y='z', color='k', ax=ax[0])
##data.o3_apriori.isel(mjd=index_lst).sel(clima_z=slice(60,130)).plot.line(y='clima_z', color = 'k', ax=ax[1])
#
#ds.o2delta.plot.line(y='z', ax=ax[0], xscale='log', add_legend=False)                                  
#ds.o3.plot.line(y='z', ax=ax[1], ls='-', marker='.', xscale='log', add_legend=False)
#ds.lsq_residual.plot.line(y='z', ax=ax[2], xscale='linear')

#%%
path = '/home/anqil/Documents/osiris_database/iris_ver_o3/'
ds.to_netcdf(path+'o3_{}{}_mr08_o3false.nc'.format(year, str(month).zfill(2)))
