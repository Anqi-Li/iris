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

#%%
def jacobian_num(fun, x, dx, args=()):
    if isinstance(fun(x, *args), tuple): # function returns more than 1 variables
        y, *temp = fun(x, *args)
        
        if isinstance(dx, (int, float, complex)):
            dx = np.ones(len(x)) * dx
            
        jac = np.empty((len(y), len(x)))
        x_perturb = x.copy()
        for i in range(len(x)):
            x_perturb[i] = x_perturb[i] + dx[i]
            y_perturb, *temp = fun(x_perturb, *args)
            jac[:,i] = (y_perturb - y)/dx[i]
            x_perturb[i] = x[i]
            
    else: # function returns only 1 variables
        y = fun(x, *args)
        if isinstance(dx, (int, float, complex)):
            dx = np.ones(len(x)) * dx
            
        jac = np.empty((len(y), len(x)))
        x_perturb = x.copy()
        for i in range(len(x)):
            x_perturb[i] = x_perturb[i] + dx[i]
            y_perturb = fun(x_perturb, *args)
            jac[:,i] = (y_perturb - y)/dx[i]
            x_perturb[i] = x[i]
                
    return jac

def weighted_lsq(y, K, Se):
    if len(y.shape) == 1:
        y = y.reshape(len(y),1)
    Se_inv = np.linalg.inv(Se)
    return np.linalg.inv(K.T.dot(Se_inv).dot(K)).dot(K.T).dot(Se_inv).dot(y)

def lsq(y, K):
    if len(y.shape) == 1:
        y = y.reshape(len(y),1)
    return np.linalg.inv(K.T.dot(K)).dot(K.T).dot(y)

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

#%%
def interation(forward_fun, fit_fun, y, x_initial, forward_args=(), fit_args=()):
    dx = 1e-2 * x_initial #determin step size for numerical jacobian
    x = x_initial.copy()
    residual = y.copy()
    i = 0
    while residual.sum() < 1:
        print('iter ', i)
        K = jacobian_num(forward_fun, x, dx, *forward_args)
        x_hat = fit_fun(y, K, *fit_args)
        y_fit = forward_fun(x_hat, *forward_args)
        residual = y-y_fit
        x_change = abs(x_hat-x)/x_hat #check Patrick Sheese
        x = x_hat
        i += 1
    return x_hat, y_fit, K, residual, x_change


#%% load ver data
year = 2008
month = 4

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


