#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 13:10:40 2019

@author: anqil
"""

import numpy as np
import scipy.sparse as sp
import xarray as xr
import matplotlib.pyplot as plt
from o2delta_model import cal_o2delta_new
plt.rcParams.update({'font.size': 14})

#%%
def jacobian_num(fun, x_in, dx, args=()):
    x = x_in.copy()
    x[x<0] = 1e-8
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

def LM_oem(y, K, x_old, y_fit_pro, Se_inv, Sa_inv, xa, D, gamma):
    if len(y.shape) == 1:
        y = y.reshape(len(y),1)
    if len(y_fit_pro.shape) == 1:
        y_fit_pro = y_fit_pro.reshape(len(y_fit_pro),1)
    if len(xa.shape) == 1:
        xa = xa.reshape(len(xa),1)
    if len(x_old.shape) == 1:
        x_old = x_old.reshape(len(xa),1)
        
    A = K.T @ (Se_inv) @ (K) + Sa_inv + gamma*D
    b = K.T @ (Se_inv @ (y-y_fit_pro) - Sa_inv @ (x_old-xa))
#    x_hat = x + np.linalg.inv(A) @ (b) #Rodgers page 93 eq 5.35    
    x_hat_minus_x_old = np.linalg.solve(A, b)
    x_hat = x_hat_minus_x_old + x_old
    return x_hat.squeeze()

def oem_cost_pro(y, y_fit, x_hat, Se_inv, Sa_inv, xa, *other_args):
    if len(y.shape) == 1:
        y = y.reshape(len(y),1)
    if len(y_fit.shape) == 1:
        y_fit = y_fit.reshape(len(y_fit),1)
    if len(xa.shape) == 1:
        xa = xa.reshape(len(xa),1)
    if len(x_hat.shape) == 1:
        x_hat = x_hat.reshape(len(xa),1)    
    cost_x = (x_hat - xa).T.dot(Sa_inv).dot(x_hat - xa) / len(y)
    cost_y = (y-y_fit).T.dot(Se_inv).dot(y-y_fit) / len(y)
    return cost_x.squeeze(), cost_y.squeeze()

def d_square_test(x_new, x_old, K, Se_inv, Sa_inv, *other_args):
    if len(x_new.shape) == 1:
        x_new = x_new.reshape(len(x_new),1)
    if len(x_old.shape) == 1:
        x_old = x_old.reshape(len(x_old),1)
    S_hat_inv = Sa_inv + K.T.dot(Se_inv).dot(K)
    d_square = (x_old-x_new).T.dot(S_hat_inv).dot(x_old-x_new)
    return d_square.squeeze()/len(x_new)

def mr_and_rms(x_hat, K, Sa_inv, Se_inv):
    if len(x_hat.shape) == 1:
        x_hat = x_hat.reshape(len(x_hat),1)
    A = Sa_inv + K.T.dot(Se_inv).dot(K)
    b = K.T.dot(Se_inv)
    G = np.linalg.solve(A, b) # gain matrix
    MR = G.dot(K).sum(axis=1)
#    Se = np.linalg.inv(Se_inv)
    Se = np.diag(1/np.diag(Se_inv)) #only works on diagonal matrix with no off-diagonal element
    Sm = G.dot(Se).dot(G.T) #retrieval noise covariance
    rms = np.diag(Sm)
    return MR, rms

def Sa_inv_off_diag(sigma, dz, h):
    # Rodgers P218 B.71
    n = len(sigma)
    alpha = np.exp(-dz/h)
    c0 = (1+alpha**2)/(1-alpha**2)
    c1 = -alpha / (1-alpha**2)
    S = sp.diags([c1, c0, c1], [-1, 0, 1], shape=(n,n)).tocsr()
    S[0,0] = 1/(1-alpha**2)
    S[-1,-1] = 1/(1-alpha**2)
    sigma_inv = 1/sigma
    Sa_inv = S.multiply(sigma_inv[:,None].dot(sigma_inv[None,:]))
    return Sa_inv

def Sa_off_diag(sigma, dz, h):
    # Rodgers P38 Eq2.82
    #diag_elements: sigma squared
    n = len(sigma)
    alpha = np.exp(-dz/h)
    Sa = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            Sa[i,j] = sigma[i] * sigma[j] * alpha**(abs(i-j))
    return Sa
    
#%% define which forward model and which fitting algorithm and their outputs
def forward(x, forward_args):
    A_o2delta = 2.23e-4 #s-1
    y, *frac = cal_o2delta_new(x, *forward_args)
    return y*A_o2delta

def fit(y, K, fit_args):
    x_hat = LM_oem(y, K, *fit_args)
    return x_hat

#%%
def interation(forward_fun, fit_fun, y, x_initial, forward_args=(), fit_args=(), max_iter=30):
    dx = 1e-3 * x_initial #determin step size for numerical jacobian
    gamma = 10
    status = 0
    K = jacobian_num(forward_fun, x_initial, dx, forward_args)
    y_fit_pro = forward_fun(x_initial, forward_args)
    cost_x_old, cost_y_old = oem_cost_pro(y, y_fit_pro, x_initial, *fit_args)
    cost_tot_old = (cost_x_old + cost_y_old)
    x_old = x_initial.copy()
    
    for n_evaluate in range(max_iter):
        K = jacobian_num(forward_fun, x_old, dx, forward_args)
        LM_fit_args = (x_old, y_fit_pro) + fit_args + (gamma,)
        x_new = fit_fun(y, K, LM_fit_args)
        y_fit_pro = forward_fun(x_new, forward_args)
#        y_fit = K.dot(x_new)
        residual = y - y_fit_pro
        cost_x, cost_y = oem_cost_pro(y, y_fit_pro, x_new, *fit_args)
        cost_tot = (cost_x + cost_y)
        
        if cost_tot <= cost_tot_old:
            d2 = d_square_test(x_new, x_old, K, *fit_args)
            x_change = np.divide(abs(x_new-x_old), abs(x_new))
            gamma /= 10
            if x_change.max() < 1e-2: #converged, stop iteration
                status = 1
            elif d2 < 0.5: #converged, stop iteration
                status = 2

            x_old = x_new.copy() 
            cost_x_old = cost_x.copy()
            cost_y_old = cost_y.copy()
            cost_tot_old = cost_tot.copy()
#            dx = 1e-3 * x_new
        while cost_tot > cost_tot_old:# cost increases--> sub iteration
            gamma *= 10
            if gamma>1e5:
                status = 3
                break
            LM_fit_args = (x_old, y_fit_pro) + fit_args + (gamma,)
            x_hat = fit_fun(y, K, LM_fit_args)
            cost_x, cost_y = oem_cost_pro(y, y_fit_pro, x_hat, *fit_args)
            cost_tot = (cost_x + cost_y)

        if status != 0:
            break

    return x_new, K, gamma, residual, status, cost_x, cost_y, n_evaluate

#%%
def loop_over_images(i):
    
    try:
        print('image', i, ' in month ', month)
        y_org = data.ver.isel(mjd=i).where(data.mr_rel.isel(mjd=i)>0.8, drop=True)
        y_org = y_org.where(y_org>0).interpolate_na('z') #neglect negative y and interpolate
        y_org = y_org.dropna('z') #in case the negative y is located at the edge of the array
        z_org = y_org.z #km
        y = y_org.values
        z = z_org.values #km
        x_initial = clima.o3.sel(lat=data.latitude.isel(mjd=i),
                                 lst=data.lst.isel(mjd=i), 
                                 method='nearest').interp(z=z).values #cm-3
        forward_args = (clima.T.sel(lat=data.latitude.isel(mjd=i),
                                    method='nearest').interp(z=z).values, #K
                        clima.m.sel(lat=data.latitude.isel(mjd=i),
                                    method='nearest').interp(z=z).values, #cm-3
                        z*1e3, #m
                        data.sza.isel(mjd=i).values, #degree
                        clima.p.sel(lat=data.latitude.isel(mjd=i), #Pa
                                    method='nearest').interp(z=z).values,
                        True) #m
        xa = x_initial
        error_2 = data.ver_error.isel(mjd=i).sel(z=z)
        Se_inv = np.diag(error_2**(-1))
#        Sa_inv = np.diag((0.75*xa)**(-2))
        Sa_inv = Sa_inv_off_diag(0.75*xa, dz=1, h=7).toarray()
        D = Sa_inv #for Levenberg Marquardt
        fit_args = (Se_inv, Sa_inv, xa, D)
        
        result = interation(forward, fit, y, x_initial, forward_args=forward_args, fit_args=fit_args)
        x_hat, K, gamma, residual, status, cost_x, cost_y, n_evaluate = result
        mr, rms = mr_and_rms(x_hat, K, Sa_inv, Se_inv)
        
        xa_out = clima.o3.sel(lat=data.latitude.isel(mjd=i),
                              lst=data.lst.isel(mjd=i), 
                              method='nearest').interp(z=z).reindex(z=data.z)
        o3 = xr.DataArray(x_hat, coords={'z': z}, dims='z').reindex(z=data.z)
        residual = xr.DataArray(residual, coords={'z': z}, dims='z').reindex(z=data.z)
        y = xr.DataArray(y, coords={'z': z}, dims='z').reindex(z=data.z)
        mr = xr.DataArray(mr, coords={'z': z}, dims='z').reindex(z=data.z)
        rms = xr.DataArray(rms, coords={'z': z}, dims='z').reindex(z=data.z)
        return (data.mjd[i], data.sza[i], data.lst[i], data.longitude[i], data.latitude[i],
                o3, y, residual, status, cost_x, cost_y, n_evaluate, gamma, mr, rms, xa_out)
    except:
        print('something went wrong for image ', i)
#        raise
        pass
    
#%% load ver data (y, Se)
year = 2008
month = 2

path = '/home/anqil/Documents/osiris_database/iris_ver_o3/'
filenames = 'ver_{}{}_v5p0.nc'.format(year, str(month).zfill(2))
data = xr.open_dataset(path+filenames)

#%% load climatology (xa, x_initial, Sa, and forward_args)
path = '/home/anqil/Documents/osiris_database/ex_data/'
file = 'msis_cmam_climatology_z200_lat8576.nc'
clima = xr.open_dataset(path+file)#.interp(z=z*1e-3)
clima = clima.update({'m':(clima.o + clima.o2 + clima.n2)*1e-6}) #cm-3
clima = clima.sel(month=month)
o3_clima = clima.o3_vmr * clima.m #cm-3
clima = clima.update({'o3': o3_clima})

#%%
#image_lst = (np.random.uniform(0,len(data.mjd), 10)).astype(int)
result=[]
for i in range(len(data.mjd)): #image_lst: #range(100):
    result.append(loop_over_images(i))
  
result = [i for i in result if i] 
mjd = np.stack([result[i][0] for i in range(len(result))])    
o3_iris = xr.DataArray(np.stack([result[i][5] for i in range(len(result))]), 
                         coords=(mjd, data.z), 
                         dims=('mjd', 'z'), name='o3', attrs={'units': 'cm-3'})
ds = xr.Dataset({'o3': o3_iris, 
                 'o3_y': (['mjd', 'z'], 
                      np.stack([result[i][6] for i in range(len(result))]), 
                      {'units': 'photons cm-3 s-1'}),
                 'residual': (['mjd', 'z'], 
                              np.stack([result[i][7] for i in range(len(result))]), 
                              {'units': 'photons cm-3 s-1'}),
                'status':(['mjd'], 
                          np.stack([result[i][8] for i in range(len(result))]),
                          {'meaning': '''0: maximum iteration exceeded (not_converged), 
                                         1: x_change is small enough,
                                         2: d2 is small enough
                                         '''}),
                'cost_x':(['mjd'], 
                          np.stack([result[i][9] for i in range(len(result))])),
                'cost_y':(['mjd'], 
                          np.stack([result[i][10] for i in range(len(result))])),
                'n_evaluate':(['mjd'], 
                          np.stack([result[i][11] for i in range(len(result))])),
                'gamma':(['mjd'], 
                          np.stack([result[i][12] for i in range(len(result))])),
                'o3_mr':(['mjd', 'z'], 
                          np.stack([result[i][13] for i in range(len(result))])),
                'o3_rms':(['mjd', 'z'], 
                          np.stack([result[i][14] for i in range(len(result))]),
                          {'units': 'cm-3 squared'}),
                'o3_a':(['mjd', 'z'], 
                          np.stack([result[i][15] for i in range(len(result))]),
                          {'units': 'cm-3'}),
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
                    '''correction on negative ozone in forward model and in K, 
                        only select mr>0.8 in VER data,
                        Sa has off-diagonal elements'''
                })    
#%%
path = '/home/anqil/Documents/osiris_database/iris_ver_o3/'
ds.to_netcdf(path+'o3_{}{}_v6p2.nc'.format(year, str(month).zfill(2)))
                    
#%%
#plt.figure()
#ds.o3.where(ds.o3_mr>0.8).plot.line(y='z', xscale='log', add_legend=False)
#plt.legend([image_lst[i] for i in range(len(image_lst))])
#plt.title(month)
##plt.ylim([60, 100])
#
##plt.figure()
##data.ver.isel(mjd=image_lst).where(data.mr_rel.isel(mjd=image_lst)>0.8).plot.line(
##        y='z', xscale='log', add_legend=False, color='k')
##ds.o3_y.plot.line(y='z', xscale='log', add_legend=False)
###plt.legend([image_lst[i] for i in range(len(image_lst))])
##plt.title(month)
#
#plt.figure()
#ds.o3_mr.plot.line(y='z', add_legend=False)
#plt.legend([image_lst[i] for i in range(len(image_lst))])
#
#plt.figure()
#ds.o3_rms.plot.line(y='z', xscale='log', add_legend=False)
#plt.legend([image_lst[i] for i in range(len(image_lst))])


