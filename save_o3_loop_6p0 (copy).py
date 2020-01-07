#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 13:10:40 2019

@author: anqil
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from o2delta_model import cal_o2delta_new
plt.rcParams.update({'font.size': 14})

#%%
def jacobian_num(fun, x, dx, args=()):
#    y, *temp = fun(x, *args) #fun must return more than 1 variable
    y = fun(x, args)
    if isinstance(dx, (int, float, complex)):
        dx = np.ones(len(x)) * dx
        
    jac = np.empty((len(y), len(x)))
    x_perturb = x.copy()
    for i in range(len(x)):
        x_perturb[i] = x_perturb[i] + dx[i]
        y_perturb = fun(x_perturb, args)
        jac[:,i] = (y_perturb - y)/dx[i]
        x_perturb[i] = x[i]            
    return jac

def weighted_lsq(y, K, Se):
    if len(y.shape) == 1:
        y = y.reshape(len(y),1)
    Se_inv = np.linalg.inv(Se)
    x_hat = np.linalg.inv(K.T.dot(Se_inv).dot(K)).dot(K.T).dot(Se_inv).dot(y)
    return x_hat.squeeze()

def lsq(y, K):
    if len(y.shape) == 1:
        y = y.reshape(len(y),1)
    x_hat = np.linalg.inv(K.T.dot(K)).dot(K.T).dot(y)
    return x_hat.squeeze()

def linear_oem(y, K, Se, Sa, xa):
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

def oem_cost(x_hat, xa, y, Sa, Se, K):
    Sa_inv = np.linalg.inv(Sa)
    Se_inv = np.linalg.inv(Se)
    cost_x = (x_hat - xa).T.dot(Sa_inv).dot(x_hat - xa)
    cost_y = (y-K.dot(x_hat)).T.dot(Se_inv).dot(y-K.dot(x_hat))
    return cost_x, cost_y

#%%
def interation(forward_fun, fit_fun, y, x_initial, forward_args=(), fit_args=()):
    dx = 1e-2 * x_initial #determin step size for numerical jacobian
    x = x_initial.copy()
    residual = y.copy()
    x_change = np.ones(len(x))
    status = 0
    n_evaluate = 0
    while x_change.max() > 1e-3: #check Patrick Sheese
        
        K = jacobian_num(forward_fun, x, dx, forward_args)
        x_hat = fit_fun(y, K, fit_args)
        y_fit = forward_fun(x_hat, forward_args)
        residual = y-y_fit
        x_change = np.divide(abs(x_hat-x), abs(x_hat)) #check Patrick Sheese
        
        ########temp plotting ######################3
        print('evaluate ', n_evaluate)
        plt.figure(figsize=(10,3))
        plt.suptitle(n_evaluate)

        plt.subplot(141)
#        plt.plot(y,z, y_fit,z)
#        plt.legend(['y org', 'y fitted'])
        plt.plot(residual/y_fit, np.arange(len(y)))
        plt.title('y - y_fit / y_fit')
#        plt.xlim([-8e4, 0])

        plt.subplot(142)
        plt.semilogx(x_change, np.arange(len(y)))
        plt.xlim([1e-10, 1e1])
        plt.title('x change')

        plt.subplot(143)
        plt.plot(K, np.arange(len(y)))
        plt.title('K')
        plt.xlim([0, 1e-1])

        plt.subplot(144)
        plt.semilogx(x, np.arange(len(y)))
        plt.semilogx(x_hat, np.arange(len(y)))
        plt.title('x_hat')
        plt.xlim([1e6, 1e10])
        
        plt.show()
        ###############################3
        
        x = x_hat
        n_evaluate += 1
        if n_evaluate > 100:
            status = 1
            break
    return x_hat, y_fit, K, residual, x_change, status


#%% load ver data
year = 2008
month = 3

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

#%% define which forward model and which fitting algorithm and their outputs
def forward(x, forward_args):
    A_o2delta = 2.23e-4 #s-1
    y, *frac = cal_o2delta_new(x, *forward_args)
    return y*A_o2delta

#fit_args = ()
def fit(y, K, fit_args):
#    x_hat = lsq(y, K)
    x_hat, *temp = linear_oem(y, K, *fit_args)
    return x_hat

#%%
def loop_over_images(i):
    
    try:
        print('image', i, ' in month ', month)
        y_org = data.ver.isel(mjd=i).where(data.mr_rel.isel(mjd=i)>0.8, drop=True).values
        z_org = data.z[data.mr_rel.isel(mjd=i)>0.8].values #km
        #z = np.sort(np.append(z_org, z_org[:-1]+np.diff(z_org)/2)) #km
#        y = np.interp(z, z_org, y_org) #photon  cm-3 s-1
        y = y_org
        z = z_org #km
        x_initial = clima.o3.sel(lat=data.latitude.isel(mjd=i),
                                 lst=data.lst.isel(mjd=i), 
                                 method='nearest').interp(z=z_org).values #cm-3
        forward_args = (clima.T.sel(lat=data.latitude.isel(mjd=i),
                                    method='nearest').interp(z=z_org).values, #K
                        clima.m.sel(lat=data.latitude.isel(mjd=i),
                                    method='nearest').interp(z=z_org).values, #cm-3
                        z_org*1e3, #m
                        data.sza.isel(mjd=i).values, #degree
                        clima.p.sel(lat=data.latitude.isel(mjd=i), #Pa
                                    method='nearest').interp(z=z_org).values,
                        True) #m
        xa = x_initial
        Sa = np.diag((1*xa)**2)
        Se = np.diag(data.ver_error.isel(mjd=i).where(data.mr_rel.isel(mjd=i)>0.8, drop=True))
        fit_args = (Se, Sa, xa)
        
        result = interation(forward, fit, y, x_initial, forward_args=forward_args, fit_args=fit_args)
        x_hat, y_fit, K, residual, x_change, status = result
        o3 = xr.DataArray(x_hat, coords={'z': z}, dims='z').reindex(z=data.z)
        residual = xr.DataArray(residual, coords={'z': z}, dims='z').reindex(z=data.z)
        cost_x, cost_y = oem_cost(x_hat, xa, y, Sa, Se, K)
        return (data.mjd[i], data.sza[i], data.lst[i], data.longitude[i], data.latitude[i],
                o3, residual, status, cost_x, cost_y)
    except:
        print('something went wrong for image ', i)
#        raise
        pass
    
#%%
#image_lst = [3986,3969]
image_lst = [15, 33, 42]
result=[]
for i in image_lst:#range(len(data.mjd)):#image_lst: #range(100):
    result.append(loop_over_images(i))
  
result = [i for i in result if i] 
mjd = np.stack([result[i][0] for i in range(len(result))])    
o3_iris = xr.DataArray(np.stack([result[i][5] for i in range(len(result))]), 
                         coords=(mjd, data.z), 
                         dims=('mjd', 'z'), name='o3', attrs={'units': 'cm-3'})
ds = xr.Dataset({'o3': o3_iris, 
                 'residual': (['mjd', 'z'], 
                              np.stack([result[i][6] for i in range(len(result))]), 
                              {'units': 'photons cm-3 s-1'}),
                'status':(['mjd'], 
                          np.stack([result[i][7] for i in range(len(result))]),
                          {'meaning': '0: converged, 1: maximum iteration exceeded'}),
                'cost_x':(['mjd'], 
                          np.stack([result[i][8] for i in range(len(result))])),
                'cost_y':(['mjd'], 
                          np.stack([result[i][9] for i in range(len(result))])),
                'sza':(['mjd'], 
                        np.stack([result[i][1] for i in range(len(result))])),
                'lst':(['mjd'], 
                        np.stack([result[i][2] for i in range(len(result))]),
                        {'units':'hour'}),
                'longitude':(['mjd'], 
                            np.stack([result[i][3] for i in range(len(result))]),
                            {'units':'degree E'}),
                'latitude':(['mjd'], 
                            np.stack([result[i][4] for i in range(len(result))]),
                            {'units':'degree N'}),
                }, attrs={'ozone fitting':
                    '''no correction on negative ozone in forward model, 
                        only select mr>0.8 in VER data.'''
                })    
                    
#%%
#path = '/home/anqil/Documents/osiris_database/iris_ver_o3/'
#ds.to_netcdf(path+'o3_{}{}_v6p0.nc'.format(year, str(month).zfill(2)))

#%%
#y_org = data.ver.isel(mjd=0).where(data.mr_rel.isel(mjd=0)>0.8, drop=True).values
#z_org = data.z[data.mr_rel.isel(mjd=0)>0.8].values #km
##z = np.sort(np.append(z_org, z_org[:-1]+np.diff(z_org)/2)) #km
#z = z_org #km
#y = np.interp(z, z_org, y_org) #photon  cm-3 s-1
#x_initial = clima.o3.sel(lat=data.latitude.isel(mjd=0),
#                         lst=data.lst.isel(mjd=0), 
#                         method='nearest').interp(z=z_org).values #cm-3
#forward_args = (clima.T.sel(lat=data.latitude.isel(mjd=0),
#                            method='nearest').interp(z=z_org).values, #K
#                clima.m.sel(lat=data.latitude.isel(mjd=0),
#                            method='nearest').interp(z=z_org).values, #cm-3
#                z*1e3, #m
#                data.sza.isel(mjd=0).values, #degree
#                clima.p.sel(lat=data.latitude.isel(mjd=0), #Pa
#                            method='nearest').interp(z=z_org).values,
#                False)
#xa = x_initial
#Sa = np.diag((1*xa)**2)
#Se = np.diag(data.ver_error.where(data.mr_rel>0.8, drop=True).isel(mjd=0))
#
#def forward(x, forward_args):
#    A_o2delta = 2.23e-4 #s-1
#    y, *frac = cal_o2delta_new(x, *forward_args)
#    return y*A_o2delta
#
#fit_args = ()
##fit_args = (Se, Sa, xa)
#def fit(y, K, fit_args):
#    x_hat = lsq(y, K)
##    x_hat, *temp = linear_oem(y, K, *fit_args)
#    return x_hat
#
#result = interation(forward, fit, y, x_initial, forward_args=forward_args, fit_args=fit_args)
#x_hat, y_fit, K, residual, x_change, status = result
