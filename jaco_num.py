#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 21:02:09 2019

@author: anqil
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from o2delta_model import cal_o2delta_new

#%%
def test_fun(x, arg=()):
    y = np.array([x[0]**2 * x[1],
                  5 * x[0] + np.sin(x[1]),
                  x[0]**2 + x[1]**2 - 1,
                  5*x[0]**2 + 21*x[1]**2 - 9
                  ])
    return y, None

def test_fun2(x, T):
    return x *1e4 *T, None
    
def jacobian_num(fun, x, dx, args=()):
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
#    dy_dx = (fun(x, *args) + fun(x+dx, *args))/dx
    
    return jac


#%%
if __name__ == '__main__':    

    file = '/home/anqil/Documents/Python/external_data/msis_cmam_climatology_z200_lat8576.nc'
    ds = xr.open_dataset(file).interp(month=1, lat=13.5)
    m = (ds.o2+ds.n2+ds.o) * 1e-6 #cm-3
    o3 = ds.o3_vmr * m
    x = o3.isel(lst=0).values
    dx = 1e-8 * x
#    dx = 1e-8
    args = (ds.T.values, m.values, ds.z.values,  30, ds.p.values)
    result = jacobian_num(cal_o2delta_new, x, dx, args=args)
#    args = ds.T.values,
#    result = jacobian_num(test_fun2, x, dx, args=args)

#    plt.figure()
#    plt.pcolormesh(result)
    
    plt.figure()
    plt.plot(result.T[:, ::10], ds.z) #plotting columns
#    plt.xlim([0, 7.5e6])
    plt.title('jacobian')
    plt.xscale('log')
    
    plt.figure()
    plt.plot(x, ds.z, label='x')
    plt.plot(dx * np.ones(len(x)), ds.z, label='dx')
    plt.legend()
    plt.title('x and x_perturb')
    plt.xscale('log')

#%%
#if __name__ == '__main__':
##    x = np.ones(2)
#    x = np.array([1, 1000]).astype(float)
##    dx = 1e-1
#    dx = 1e-1 * x
#    
#    result = jacobian_num(test_fun, x, dx)
#    print(result.round(0))
##    
#    true = np.array([[2*x[0]*x[1], x[0]**2],
#                    [5, np.cos(x[1])],
#                    [2*x[0], 2*x[1]],
#                    [10*x[0], 42*x[1]]
#                     ])
#    print(true.round(0))   
#    plt.figure()
#    plt.plot(result)
#    
#    
##    print(test_fun(x))
##    print(true.dot(x))
    