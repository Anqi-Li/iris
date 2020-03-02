#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 10:53:41 2020

@author: anqil

combined O2(1delta) and ozone retrieval from IRIS stray-light corrected limb radiance
O2 VER uses standard OEM 
Ozone uses Levenberg-Marquardt
a priori taken from CMAM 
other parameters (T, p) taken from MSIS
Sa has off-diagonal elements

"""
import numpy as np
import scipy.sparse as sp
import xarray as xr
import matplotlib.pyplot as plt
from o2delta_model import cal_o2delta_new
plt.rcParams.update({'font.size': 14})
from netCDF4 import num2date
units = 'days since 1858-11-17 00:00:00.000'


#%% load IRIS limb radiance data 
orbit100_idx = -1
orbit100 = np.arange(364, 375)
filename = 'ir_slc_{}xx_ch3.nc'
#path = 'https://arggit.usask.ca/opendap/'
path = '/home/anqil/Documents/osiris_database/odin-osiris.usask.ca/IR_slc/'
ir = xr.open_dataset(path+filename.format(orbit100[orbit100_idx]), chunks={'mjd':100})
ir = ir.sel(mjd=~ir.indexes['mjd'].duplicated())
day_mjd_lst = ir.mjd[ir.sza<90] # select only dayglow
ir = ir.sel(mjd=day_mjd_lst)
altitude = ir.altitude.sel(pixel=slice(14, 128))
fr = 0.72 # filter fraction 
normalize = np.pi*4 / fr
lst = ir.apparent_solar_time.sel(pixel=60).drop('pixel')
l1 = ir.sel(pixel=slice(14, 128)).data * normalize
error = ir.sel(pixel=slice(14, 128)).error * normalize

#%% load climatology (xa, x_initial, Sa, and forward_args)
path = '/home/anqil/Documents/osiris_database/ex_data/'
file = 'msis_cmam_climatology_z200_lat8576.nc'
clima = xr.open_dataset(path+file)#.interp(z=z*1e-3)
clima = clima.update({'m':(clima.o + clima.o2 + clima.n2)*1e-6}) #cm-3
#clima = clima.sel(month=month)
clima_o3 = clima.o3_vmr * clima.m #cm-3
clima = clima.update({'o3': clima_o3})

#%% Global variables
A_o2delta = 2.23e-4
#====drop data below and above some altitudes
bot = 61e3
top = 100e3
#====retireval grid
z = np.arange(50e3, 130e3, 1e3) # m
z_top = z[-1] + 1e3 # for jacobian

#%%
#image_lst = (np.random.uniform(0,len(ir.mjd), 5)).astype(int)
#image_lst = [130]
image_lst = [609]

#%%
def f(i):
    alt_chop_cond = (altitude.isel(mjd=i)>bot) & (altitude.isel(mjd=i)<top) #select altitude range 
    if alt_chop_cond.sum() < 2:
        print('wrong alt range ({}), file {}'.format(i, orbit100[orbit100_idx]))
        pass
    elif l1.isel(mjd=i).notnull().sum() == 0:
        print('l1 is all nan ({}), file {}'.format(i, orbit100[orbit100_idx]))
        pass
    elif error.isel(mjd=i).notnull().sum() == 0:
        print('error is all nan ({}), file {}'.format(i, orbit100[orbit100_idx]))
        pass
    
    else:
        try:
            print(i, 'out of', len(day_mjd_lst), 'in file ', orbit100[orbit100_idx])
            month = num2date(ir.mjd[i], units).month
            o3_clima = clima.o3.sel(lst=lst.isel(mjd=i),
                                lat=ir.latitude.isel(mjd=i),
                                month=month,
                                method='nearest')
            T_a = clima.T.sel(lat=ir.latitude.isel(mjd=i),
                              month=month,
                              method='nearest')
            m_a = clima.m.sel(lat=ir.latitude.isel(mjd=i),
                              month=month,
                              method='nearest')
            p_a = clima.p.sel(lat=ir.latitude.isel(mjd=i), 
                              month=month,
                              method='nearest') 
    
            sol_zen = ir.sza[i].values.item()
            
            forward_args = (T_a.values, m_a.values,
                            clima.z.values*1e3, #in m
                            sol_zen, p_a.values)
            ver_clima = forward(o3_clima.values, forward_args)
            ver_a = np.interp(z*1e-3, clima.z, ver_clima)

            h = altitude.isel(mjd=i).where(l1.isel(mjd=i).notnull(), drop=True
                              ).where(alt_chop_cond, drop=True)
            K = pathl1d_iris(h.values, z, z_top) * 1e2 # m-->cm    
            limb = l1.isel(mjd=i).sel(pixel=h.pixel).values     
            
            error_2 = (error.isel(mjd=i).sel(pixel=h.pixel).values)**2
            Se_inv = np.diag(1/error_2)
#            Sa_inv = np.diag(1/(0.75*ver_a)**2)
#            print('diag only Sa')
            Sa_inv = Sa_inv_off_diag(0.75*ver_a, dz=1, h=5).toarray()
#            print('off Sa')
            ver_hat, G = linear_oem(limb, K, Se_inv, Sa_inv, ver_a)
            ver_mr, ver_Sm, ver_A = mr_and_Sm(ver_hat, K, Sa_inv, Se_inv)
            ver_error = np.diag(ver_Sm)
            limb_fit = K.dot(ver_hat)
            ver_cost_x, ver_cost_y = oem_cost_pro(limb, limb_fit, ver_hat, Se_inv, Sa_inv, ver_a)
            ver_A_rel = rel_avk(ver_a, ver_A)
            ver_mr_rel = ver_A_rel.sum(axis=1)
            
            #%%%%%%%%%%% select only high ver_mr_rel to retireve o3 %%%%%%%%%%%
            cond_ver_mr = np.where(ver_mr_rel>0.8)
            z_mr = z[cond_ver_mr] #in meter
            ver_cond = ver_hat[cond_ver_mr]
            ver = np.interp(z_mr, z_mr[np.where(ver_cond>0)], ver_cond[np.where(ver_cond>0)])
            o3_initial = o3_clima.interp(z=z_mr*1e-3).values #cm-3
            forward_args = (T_a.interp(z=z_mr*1e-3).values, #K
                            m_a.interp(z=z_mr*1e-3).values, #cm-3
                            z_mr, #m
                            sol_zen, #degree
                            p_a.interp(z=z_mr*1e-3).values,
                            True) #m
            o3_a = o3_initial.copy()
            Sa_inv = Sa_inv_off_diag(0.75*o3_a, dz=1, h=5).toarray()
#            Se = ver_Sm[cond_ver_mr][:,cond_ver_mr].squeeze()
#            Se_inv = np.linalg.inv(Se) #full Se matrix
            Se_inv = np.diag(1/ver_error[cond_ver_mr]) #only diaganol
            D = Sa_inv #for Levenberg Marquardt
            fit_args = (Se_inv, Sa_inv, o3_a, D)
            
            LM_result = interation(forward, fit, ver, o3_initial, forward_args=forward_args, fit_args=fit_args)
            o3_hat, K, o3_gamma, ver_residual, o3_status, o3_cost_x, o3_cost_y, o3_n_evaluate = LM_result
            o3_mr, o3_Sm, o3_A = mr_and_Sm(o3_hat, K, Sa_inv, Se_inv)
            o3_error = np.diag(o3_Sm)
            o3_A_rel = rel_avk(o3_a, o3_A)
            o3_mr_rel = o3_A_rel.sum(axis=1)

            #%%%%%%%% reindexing to an uniformed length %%%%%%%%%%%%%%%%%%%%%%%%%%
            o3_hat = xr.DataArray(o3_hat, coords={'z': z_mr}, dims='z').reindex(z=z)
            ver_residual = xr.DataArray(ver_residual, coords={'z': z_mr}, dims='z').reindex(z=z)
            o3_mr = xr.DataArray(o3_mr, coords={'z': z_mr}, dims='z').reindex(z=z)
            o3_mr_rel = xr.DataArray(o3_mr_rel, coords={'z': z_mr}, dims='z').reindex(z=z)
            o3_error = xr.DataArray(o3_error, coords={'z': z_mr}, dims='z').reindex(z=z)
                                    
            return (day_mjd_lst[i].values,
                    ver_hat, ver_mr, ver_mr_rel, np.sqrt(ver_error),
                    ver_cost_x, ver_cost_y,
                    o3_hat, o3_mr, o3_mr_rel, np.sqrt(o3_error), ver_residual,
                    o3_cost_x, o3_cost_y, o3_n_evaluate, o3_gamma, o3_status, 
                    o3_clima.values, ver_clima, 
                    ver_A_rel, o3_A_rel
                    )
     
        except:
            print('something is wrong ({})'.format(i))
    
            raise
            pass

#%%
def jacobian_num(fun, x_in, dx, args=()):
    x = x_in.copy()
    x[x<0] = 1e-8 #for negative ozone
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
    #Rodgers page 93 eq 5.35    
    x_hat_minus_x_old = np.linalg.solve(A, b)
    x_hat = x_hat_minus_x_old + x_old
    return x_hat.squeeze()

def d_square_test(x_new, x_old, K, Se_inv, Sa_inv, *other_args):
    if len(x_new.shape) == 1:
        x_new = x_new.reshape(len(x_new),1)
    if len(x_old.shape) == 1:
        x_old = x_old.reshape(len(x_old),1)
    S_hat_inv = Sa_inv + K.T.dot(Se_inv).dot(K)
    d_square = (x_old-x_new).T.dot(S_hat_inv).dot(x_old-x_new)
    return d_square.squeeze()/len(x_new)

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
#        print('iteration', n_evaluate)
        K = jacobian_num(forward_fun, x_old, dx, forward_args)
        LM_fit_args = (x_old, y_fit_pro) + fit_args + (gamma,)
        x_new = fit_fun(y, K, LM_fit_args)
        y_fit_pro = forward_fun(x_new, forward_args)
        residual = y - y_fit_pro
        cost_x, cost_y = oem_cost_pro(y, y_fit_pro, x_new, *fit_args)
        cost_tot = (cost_x + cost_y)
        
        if cost_tot <= cost_tot_old:
            d2 = d_square_test(x_new, x_old, K, *fit_args)
            x_change = np.divide(abs(x_new-x_old), abs(x_new))
            gamma /= 10
            if x_change.max() < 1e-2: #converged, stop iteration
#                print('converged 1')
                status = 1
            elif d2 < 0.5: #converged, stop iteration
#                print('converged 2')
                status = 2

            x_old = x_new.copy() 
            cost_x_old = cost_x.copy()
            cost_y_old = cost_y.copy()
            cost_tot_old = cost_tot.copy()
        
        while cost_tot > cost_tot_old:# cost increases--> sub iteration
            gamma *= 10
#            print('increased gamma', gamma)
            if gamma>1e5:
#                print('gamma reached to max')
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
def forward(x, forward_args):
    o2delta, *frac = cal_o2delta_new(x, *forward_args)
    return o2delta*A_o2delta

def fit(y, K, fit_args):
#    x_hat = lsq(y, K)
#    x_hat, *temp = linear_oem(y, K, *fit_args)
    x_hat = LM_oem(y, K, *fit_args)
    return x_hat

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

#%% convert ordinary to relative AVK
def rel_avk(xa, A):
    #xa 1D array lenth n
    #A 2D array size (n, n)
    A_rel = np.zeros(A.shape)
    for i in range(len(xa)):
        for j in range(len(xa)):
            A_rel[i, j] = xa[j] * A[i, j] / xa[i]
    return A_rel

def mr_and_Sm(x_hat, K, Sa_inv, Se_inv):
    if len(x_hat.shape) == 1:
        x_hat = x_hat.reshape(len(x_hat),1)
    A = Sa_inv + K.T.dot(Se_inv).dot(K)
    b = K.T.dot(Se_inv)
    G = np.linalg.solve(A, b) # gain matrix
    AVK = G.dot(K)
    MR = AVK.sum(axis=1)
    Se = np.linalg.inv(Se_inv)
#    Se = np.diag(1/np.diag(Se_inv)) #only works on diagonal matrix with no off-diagonal element
    Sm = G.dot(Se).dot(G.T) #retrieval noise covariance
    return MR, Sm, AVK

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

#%%
def linear_oem(y, K, Se_inv, Sa_inv, xa):
    if len(y.shape) == 1:
        y = y.reshape(len(y),1)
    if len(xa.shape) == 1:
        xa = xa.reshape(len(xa),1)
    G= np.linalg.solve(K.T.dot(Se_inv).dot(K) + Sa_inv, (K.T).dot(Se_inv))        
    x_hat = xa + G.dot(y - K.dot(xa)) 
    
    return x_hat.squeeze(), G

#%%
result = []
for i in image_lst:
    result.append(f(i))

result = [i for i in result if i] # filter out all None element in list
mjd = np.stack([result[i][0] for i in range(len(result))])
ver_hat = xr.DataArray(np.stack([result[i][1] for i in range(len(result))]), 
                         coords=(mjd, z*1e-3), 
                         dims=('mjd', 'z'), name='VER', attrs={'units': 'photons cm-3 s-1'})
o3_clima = xr.DataArray(np.stack([result[i][17] for i in range(len(result))]), 
                         coords=(mjd, clima.z), 
                         dims=('mjd', 'clima_z'), name='o3_cmam', attrs={'units': 'cm-3'})

ds = xr.Dataset({'ver': ver_hat, 
                 'ver_mr':(['mjd', 'z'], 
                       np.stack([result[i][2] for i in range(len(result))])), 
                 'ver_mr_rel':(['mjd', 'z'], 
                       np.stack([result[i][3] for i in range(len(result))])),
                 'ver_error':(['mjd', 'z'], 
                       np.stack([result[i][4] for i in range(len(result))])),
                 'ver_cost_x':(['mjd'], 
                       np.stack([result[i][5] for i in range(len(result))])), 
                 'ver_cost_y':(['mjd'], 
                       np.stack([result[i][6] for i in range(len(result))])),  
                 'o3':(['mjd', 'z'], 
                       np.stack([result[i][7] for i in range(len(result))])),              
                 'o3_mr':(['mjd', 'z'], 
                       np.stack([result[i][8] for i in range(len(result))])), 
                 'o3_mr_rel':(['mjd', 'z'], 
                       np.stack([result[i][9] for i in range(len(result))])),
                 'o3_error':(['mjd', 'z'], 
                       np.stack([result[i][10] for i in range(len(result))])),
                 'ver_residual':(['mjd', 'z'], 
                       np.stack([result[i][11] for i in range(len(result))])),            
                 'o3_cost_x':(['mjd'], 
                       np.stack([result[i][12] for i in range(len(result))])), 
                 'o3_cost_y':(['mjd'], 
                       np.stack([result[i][13] for i in range(len(result))])),
                 'o3_evaluate':(['mjd'], 
                       np.stack([result[i][14] for i in range(len(result))])),  
                 'o3_gamma':(['mjd'], 
                       np.stack([result[i][15] for i in range(len(result))])),  
                 'o3_status':(['mjd'], 
                       np.stack([result[i][16] for i in range(len(result))])), 
                 'o3_a': o3_clima,
                 'ver_a':(['mjd', 'clima_z'], 
                       np.stack([result[i][18] for i in range(len(result))])),
                'longitude': (['mjd'], ir.longitude.sel(mjd=mjd)),
                'latitude': (['mjd'], ir.latitude.sel(mjd=mjd)),
                'sza': (['mjd'], ir.sza.sel(mjd=mjd)),
                'lst': (['mjd'], lst.sel(mjd=mjd))
                               })
ds.z.attrs['units'] = 'km'
#ds = ds.update({'longitude': ir.longitude.sel(mjd=ds.mjd),
#                'latitude': ir.latitude.sel(mjd=ds.mjd),
#                'sza': ir.sza.sel(mjd=ds.mjd),
#                'lst': lst.sel(mjd=ds.mjd)})
                
#%%
fig, ax = plt.subplots(1,2, sharey=True)                
ds.ver.where(ds.ver_mr_rel>0.8).plot.line(y='z', xscale='log', ax=ax[0], add_legend=False)
#ds.ver_a.plot.line(y='clima_z', xscale='log', ax=ax[0], add_legend=False, color='grey')
ds.ver_mr_rel.plot.line(y='z', xlim=(0.7,1.2), ax=ax[1], add_legend=False)

fig, ax = plt.subplots(1,2, sharey=True)                
ds.o3.where(ds.o3_mr_rel>0.8).plot.line(y='z', xscale='log', ax=ax[0], add_legend=False)
#ds.o3_a.plot.line(y='clima_z', xscale='log', ax=ax[0], add_legend=False, color='grey')
ds.o3_mr_rel.plot.line(y='z', xlim=(0.7,1.2), ax=ax[1], add_legend=False)

#%%
from scipy.optimize import curve_fit
def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def resolution(x,y):        
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))
    popt, pcov = curve_fit(gaus,x,y, p0=[max(y), mean, sigma])
##    plt.figure()
#    plt.plot(x, y,'b-',label='data')
#    plt.plot(x,gaus(x, *popt),'ro:',label='fit')
##    plt.legend()
    return popt[2]*2*np.sqrt(2*np.log(2))

ver_A_rel = result[0][-2]
o3_A_rel = result[0][-1]
z_mr = ds.z.where(ds.o3.notnull().squeeze(), drop=True)



fig = plt.figure()
mr = ds.ver_mr_rel.plot.line(y='z', add_legend=False, color='k', ls='--')
below = plt.plot(ver_A_rel[:50:2, :].T, ds.z, color='C1')
above = plt.plot(ver_A_rel[50::2, :].T, ds.z, color='C2')
plt.title('VER retrieval')
plt.ylabel('Altitude / km')
plt.xlabel('AVK or MR')
plt.legend([mr[0], below[0], above[0]], ['$MR^{frac}$', '$AVK^{frac}$ below 100 km', '$AVK^{frac}$ above 100 km'])
fig.savefig('/home/anqil/Documents/reportFigure/article2/iris_ver_AVK_MR_sample.png',
        bbox_inches = "tight") 
x = ds.z.values
res = []
for i in range(len(x)):
    y = ver_A_rel[i, :]
    res.append(resolution(x, y))
#plt.figure()
plt.plot(res, ds.z, '.')


fig = plt.figure()
mr = ds.o3_mr_rel.plot.line(y='z', add_legend=False, color='k', ls='--')
below = plt.plot(o3_A_rel[::2, :].T, z_mr)
plt.title('Ozone retrieval')
plt.ylabel('Altitude / km')
plt.xlabel('AVK or MR')
plt.legend(['$MR^{frac}$'] + ['$AVKs^{frac}$'], loc='lower center')
fig.savefig('/home/anqil/Documents/reportFigure/article2/iris_o3_AVK_MR_sample.png',
        bbox_inches = "tight") 
x = z_mr
res = []
for i in range(1,len(x)-1):
    y = o3_A_rel[i, :]
    res.append(resolution(x, y))
#plt.figure()
plt.plot(res, z_mr[1:-1], '.')


#%%

#x = ds.z.values



#plt.plot(res, ds.z, '.')
#%%
#import os
#os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
#from dask.diagnostics import ProgressBar
#savedir = '/home/anqil/Documents/osiris_database/iris_ver_o3/ver_o3/'
#filename = 'ver_o3_test.nc'
#delayed_obj = ds.to_netcdf(savedir+filename, compute=False)
#with ProgressBar():
#    results=delayed_obj.compute()