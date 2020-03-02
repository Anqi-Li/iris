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

#def weighted_lsq(y, K, Se):
#    if len(y.shape) == 1:
#        y = y.reshape(len(y),1)
#    Se_inv = np.linalg.inv(Se)
#    x_hat = np.linalg.inv(K.T.dot(Se_inv).dot(K)).dot(K.T).dot(Se_inv).dot(y)
#    return x_hat.squeeze()
#
#def lsq(y, K):
#    if len(y.shape) == 1:
#        y = y.reshape(len(y),1)
#    x_hat = np.linalg.inv(K.T.dot(K)).dot(K.T).dot(y)
#    return x_hat.squeeze()
#
#def linear_oem(y, K, Se, Sa, xa):
#    if len(y.shape) == 1:
#        y = y.reshape(len(y),1)
#    if len(xa.shape) == 1:
#        xa = xa.reshape(len(xa),1)
#        
#    if len(y)<len(xa): # m form
#        G = Sa.dot(K.T).dot(np.linalg.inv(K.dot(Sa).dot(K.T) + Se))
#        
#    else: # n form
#        Se_inv = np.linalg.inv(Se)
#        Sa_inv = np.linalg.inv(Sa)
#        G = np.linalg.inv(K.T.dot(Se_inv).dot(K) + Sa_inv).dot(K.T).dot(Se_inv)
##        G= np.linalg.solve(K.T.dot(Se_inv).dot(K) + Sa_inv, (K.T).dot(Se_inv))
#        
#    x_hat = xa + G.dot(y - K.dot(xa)) 
#    A = G.dot(K)
#    Ss = (A - np.identity(len(xa))).dot(Sa).dot((A - np.identity(len(xa))).T) # smoothing error
#    Sm = G.dot(Se).dot(G.T) #retrieval noise 
#    
#    return x_hat.squeeze(), A, Ss, Sm

#def oem_cost(y, K, x_hat, Se_inv, Sa_inv, xa, *other_args):
#    if len(y.shape) == 1:
#        y = y.reshape(len(y),1)
#    if len(xa.shape) == 1:
#        xa = xa.reshape(len(xa),1)
#    if len(x_hat.shape) == 1:
#        x_hat = x_hat.reshape(len(xa),1)
#        
##    Sa_inv = np.linalg.inv(Sa)
##    Se_inv = np.linalg.inv(Se)
#    cost_x = (x_hat - xa).T.dot(Sa_inv).dot(x_hat - xa) / len(y)
#    cost_y = (y-K.dot(x_hat)).T.dot(Se_inv).dot(y-K.dot(x_hat)) / len(y)
#    return cost_x.squeeze(), cost_y.squeeze()
    
def LM_oem(y, K, x_old, y_fit_pro, Se_inv, Sa_inv, xa, D, gamma):
    if len(y.shape) == 1:
        y = y.reshape(len(y),1)
    if len(y_fit_pro.shape) == 1:
        y_fit_pro = y_fit_pro.reshape(len(y_fit_pro),1)
    if len(xa.shape) == 1:
        xa = xa.reshape(len(xa),1)
    if len(x_old.shape) == 1:
        x_old = x_old.reshape(len(xa),1)
        
#    Se_inv = np.linalg.inv(Se)
#    Sa_inv = np.linalg.inv(Sa)
    
#    a = np.linalg.inv(K.T.dot(Se_inv).dot(K) + Sa_inv + gamma*D)
#    b = K.T.dot(Se_inv.dot(y-K.dot(x)) - Sa_inv.dot(x-xa))
#    x_hat = x + a.dot(b) #Rodgers page 93 eq 5.35
    A = K.T @ (Se_inv) @ (K) + Sa_inv + gamma*D
#    b = K.T @ (Se_inv @ (y-K @ (x_old)) - Sa_inv @ (x_old-xa))
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

def rel_avk(xa, A):
    #xa 1D array lenth n
    #A 2D array size (n, n)
    A_rel = np.zeros(A.shape)
    for i in range(len(xa)):
        for j in range(len(xa)):
            A_rel[i, j] = xa[j] * A[i, j] / xa[i]
    return A_rel

def mr_and_rms(x_hat, K, Sa_inv, Se_inv):
    if len(x_hat.shape) == 1:
        x_hat = x_hat.reshape(len(x_hat),1)
    A = Sa_inv + K.T.dot(Se_inv).dot(K)
    b = K.T.dot(Se_inv)
    G = np.linalg.solve(A, b) # gain matrix
    AVK = G.dot(K)
    MR = AVK.sum(axis=1)
#    Se = np.linalg.inv(Se_inv)
    Se = np.diag(1/np.diag(Se_inv)) #only works on diagonal matrix with no off-diagonal element
    Sm = G.dot(Se).dot(G.T) #retrieval noise covariance
    rms = np.diag(Sm)
    
    return MR, rms, AVK

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
#    x_hat = lsq(y, K)
#    x_hat, *temp = linear_oem(y, K, *fit_args)
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
    
#    temp_x = [cost_x_old] # temporary for ploting changes through iterations
#    temp_y = [cost_y_old]
#    temp_tot = [cost_tot_old]
#    temp_gamma = [gamma]
#    temp_d2 = []
#    temp_xchange_max = []
    
    for n_evaluate in range(max_iter):
        print('iteration', n_evaluate)
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
#            print('reduced gamma', gamma)
            if x_change.max() < 1e-2: #converged, stop iteration
                print('converged 1')
                status = 1
#                break
            elif d2 < 0.5: #converged, stop iteration
                print('converged 2')
                status = 2
#                break
#            temp_x.append(cost_x)
#            temp_y.append(cost_y)
#            temp_tot.append(cost_tot)
#            temp_gamma.append(gamma)
#            temp_d2.append(d2)
#            temp_xchange_max.append(x_change.max())

        ########temp plotting for each iteration ######################3
#            plt.figure(figsize=(10,3))
#            plt.suptitle(n_evaluate)
#    
#            plt.subplot(141)
#    #        plt.plot(residual/y_fit, forward_args[2]*1e-3)
#            plt.plot(y, forward_args[2]*1e-3, label='y_org')
#            plt.plot(y_fit_pro, forward_args[2]*1e-3, label='y_fit')
#            plt.legend()
#            plt.title('y and y_fit')
#    #        plt.xlim([-8e4, 0])
#    
#            plt.subplot(142)
#            plt.plot(x_change, forward_args[2]*1e-3)
#    #        plt.xlim([0, 1e0])
#            plt.title('x change')
#    
#            plt.subplot(143)
#            plt.plot(K, forward_args[2]*1e-3)
#            plt.title('K')
#            plt.xlim([0, 1e-2])
#    
#            plt.subplot(144)
#            plt.semilogx(x_initial, forward_args[2]*1e-3, color='k')
#            plt.semilogx(x_old, forward_args[2]*1e-3)
#            plt.semilogx(x_new, forward_args[2]*1e-3)
#            plt.title('x')
#            plt.xlim([1e1, 1e10])
#            
#            plt.show()
            ###############################3
            
            x_old = x_new.copy() 
            cost_x_old = cost_x.copy()
            cost_y_old = cost_y.copy()
            cost_tot_old = cost_tot.copy()
#            dx = 1e-3 * x_new
        
        while cost_tot > cost_tot_old:# cost increases--> sub iteration
            gamma *= 10
            print('increased gamma', gamma)
            if gamma>1e5:
                print('gamma reached to max')
                status = 3
                break
            LM_fit_args = (x_old, y_fit_pro) + fit_args + (gamma,)
            x_hat = fit_fun(y, K, LM_fit_args)
            cost_x, cost_y = oem_cost_pro(y, y_fit_pro, x_hat, *fit_args)
            cost_tot = (cost_x + cost_y)
            
            
        if status != 0:
            break

        
    ####### temp plotting######################## 
#    plt.figure(figsize=(10,2))
#    plt.subplot(151)
#    plt.plot(temp_x,'.-')
#    plt.axhline(y=1)
#    plt.xlim([0, n_evaluate])
#    plt.xlabel('iter')
#    plt.title('cost_x')
#    plt.subplot(152)
#    plt.plot(temp_y,'.-')
#    plt.axhline(y=1)
#    plt.xlim([0, n_evaluate])
#    plt.xlabel('iter')
#    plt.title('cost_y')
#    plt.subplot(153)
#    plt.plot(temp_tot,'.-')
#    plt.axhline(y=1)
#    plt.xlim([0, n_evaluate])
#    plt.title('total cost')
#    plt.subplot(154)
#    plt.semilogy(temp_gamma,'.-')
#    plt.title('gamma')
#    plt.xlabel('iter')
#    plt.xlim([0, n_evaluate])
#    plt.subplot(155)
#    plt.plot(temp_d2,'.-')
#    plt.xlim([0, n_evaluate])
#    plt.title('d2')
#    plt.xlabel('iter')
#
#    plt.show()
    ####################################

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
        Se = np.diag(error_2)
#        Sa = np.diag((0.75*xa)**2)
#        Sa_inv = np.linalg.inv(Sa)
        Sa_inv = Sa_inv_off_diag(0.75*xa, dz=1, h=5).toarray()
        Se_inv = np.linalg.inv(Se)
        D = Sa_inv #for Levenberg Marquardt
        fit_args = (Se_inv, Sa_inv, xa, D)
        
        result = interation(forward, fit, y, x_initial, forward_args=forward_args, fit_args=fit_args)
        x_hat, K, gamma, residual, status, cost_x, cost_y, n_evaluate = result
        mr, rms, AVK = mr_and_rms(x_hat, K, Sa_inv, Se_inv)
        
        xa_out = clima.o3.sel(lat=data.latitude.isel(mjd=i),
                                 lst=data.lst.isel(mjd=i), 
                                 method='nearest').interp(z=z).reindex(z=data.z)
        o3 = xr.DataArray(x_hat, coords={'z': z}, dims='z').reindex(z=data.z)
        residual = xr.DataArray(residual, coords={'z': z}, dims='z').reindex(z=data.z)
        y = xr.DataArray(y, coords={'z': z}, dims='z').reindex(z=data.z)
        mr = xr.DataArray(mr, coords={'z': z}, dims='z').reindex(z=data.z)
        rms = xr.DataArray(rms, coords={'z': z}, dims='z').reindex(z=data.z)
        return (data.mjd[i], data.sza[i], data.lst[i], data.longitude[i], data.latitude[i],
                o3, y, residual, status, cost_x, cost_y, n_evaluate, gamma, mr, rms, xa_out, AVK)
    except:
        print('something went wrong for image ', i)
        raise
        pass
    
#%% load ver data (y, Se)
year = 2008
month = 1

path = '/home/anqil/Documents/osiris_database/iris_ver_o3/ver/'
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
#image_lst = [3986,3969]
#image_lst = [0,1,2,3,4, 15, 19, 33, 42, 94]
#image_lst = np.arange(1)
#image_lst = [10639,  4720,  6465, 26571,  3251]
#image_lst = [10316, 15108, 3466, 17142]
#image_lst = (np.random.uniform(0,len(data.mjd), 1)).astype(int)
image_lst = [570]
result=[]
for i in image_lst:#range(len(data.mjd)):#image_lst: #range(100):
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
                    '''no correction on negative ozone in forward model, 
                        only select mr>0.8 in VER data.'''
                })    
                    
#%%
plt.figure()
ds.o3.where(ds.o3_mr>0.8).plot.line(y='z', xscale='log', add_legend=False)
plt.legend([image_lst[i] for i in range(len(image_lst))])
plt.title(month)
#plt.ylim([60, 100])

#plt.figure()
#data.ver.isel(mjd=image_lst).where(data.mr_rel.isel(mjd=image_lst)>0.8).plot.line(
#        y='z', xscale='log', add_legend=False, color='k')
#ds.o3_y.plot.line(y='z', xscale='log', add_legend=False)
##plt.legend([image_lst[i] for i in range(len(image_lst))])
#plt.title(month)

plt.figure()
ds.o3_mr.plot.line(y='z', add_legend=False)
plt.legend([image_lst[i] for i in range(len(image_lst))])

#plt.figure()
#ds.o3_rms.plot.line(y='z', xscale='log', add_legend=False)
#plt.legend([image_lst[i] for i in range(len(image_lst))])

fig = plt.figure()
(np.sqrt(ds.o3_rms)/ds.o3_a).plot.line(y='z', add_legend=False)
plt.xlabel('O3 accuracy')
#fig.savefig('/home/anqil/Documents/reportFigure/article2/iris_o3_uncertainty_sample.png',
#            bbox_inches = "tight")

#%%
fig = plt.figure()
AVK = result[0][-1]
AVK_rel = rel_avk(ds.o3_a.where(ds.o3.notnull(),drop=True).squeeze(), AVK)
z_mr = ds.z.where(ds.o3.notnull().squeeze(), drop=True)

mr = ds.o3_mr.plot(y='z', color='k')
avk = plt.plot(AVK[::2].T, z_mr, color='C1')
plt.xlabel('AVK')
plt.legend([mr[0], avk[0]], ['$MR^{frac}$', '$AVK^{frac}$'])
fig.savefig('/home/anqil/Documents/reportFigure/article2/iris_o3_avk_sample.png',
            bbox_inches = "tight")

#%%
from scipy.signal import chirp, find_peaks, peak_widths
P, W = [], []
for i in range(len(AVK)):
    x = AVK[i,:]
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
#path = '/home/anqil/Documents/osiris_database/iris_ver_o3/'
#ds.to_netcdf(path+'o3_{}{}_v6p0.nc'.format(year, str(month).zfill(2)))

