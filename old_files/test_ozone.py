#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 13:04:08 2019

@author: anqil
"""

from osirisl1services.readlevel1 import open_level1_ir
from osirisl1services.services import Level1Services
import sys
sys.path.append('..')
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
from netCDF4 import num2date
units = 'days since 1858-11-17 00:00:00.000'

#%%
#channel = 2
#orbit = 6434
#orbit = 22873
channel = 3
orbit = 20900

ir = open_level1_ir(orbit, channel, valid=False)
tan_alt = ir.l1.altitude
tan_lat = ir.l1.latitude
tan_lon = ir.l1.longitude
sc_look = ir.l1.look_ecef
sc_pos = ir.l1.position_ecef
l1 = ir.data
mjd = ir.mjd.data
pixel = ir.pixel.data

#====drop data below and above some altitudes
l1 = l1.where(tan_alt<110e3).where(tan_alt>60e3)

#%%
#im_lst = np.arange(300,350,5)
im_lst = np.arange(2350,2425,5)
pix_lst = np.arange(22, 128)

label_interval = 10

print(num2date(ir.mjd[0],units))
print('number of images: ', len(im_lst))

#%%
alts_interp = np.arange(40e3, 120e3, .25e3)
data_interp = []

for (data, alt) in zip(l1, tan_alt):
    f = interp1d(alt, data, bounds_error=False)
    data_interp.append(f(alts_interp))
data_interp = xr.DataArray(data_interp, coords=(mjd, alts_interp), 
                           dims=('mjd', 'alt'))

plt.figure()
data_interp.plot(x='mjd', y='alt', 
                 norm=LogNorm(), 
                 vmin=1e9, vmax=1e13, 
                 size=5, aspect=3)
ax = plt.gca()
ax.set(title='From {} \n to {}, \n channel {}'.format(num2date(mjd[0],units),
       num2date(mjd[-1], units), channel))
plt.axvline(x=mjd[im_lst[0]], color='k', linewidth=1)
plt.axvline(x=mjd[im_lst[-1]], color='k', linewidth=1)
plt.show()

plt.figure()
data_interp.isel(mjd=im_lst).plot(x='mjd', y='alt', 
                 norm=LogNorm(), 
                 vmin=1e9, vmax=1e13, 
                 size=5, aspect=3)
ax = plt.gca()
ax.set(title='zoom in im_lst')
plt.show()

plt.figure()
data_interp.isel(mjd=im_lst).plot.line(y='alt')
plt.ylim([50e3, 100e3])
plt.legend([])
#%% 1D inversion 
from oem_functions import linear_oem
from geometry_functions import pathl1d_iris

z = np.arange(60e3, 110e3, 2e3) # m
z_top = z[-1] + 2e3

result_1d = np.zeros((len(im_lst), len(z)))
xa = np.ones(len(z)) *0 # temp
#Sa = np.diag(np.ones(len(z)))*(1e7)**2 #temp
Sa = np.diag(np.ones(len(z))) *1e-9 #temp
#Se = np.diag(np.ones(len(pix_lst))) * (2.5e10)**2# 1e10 #30 #temporary
Ave = []
resi = []
for i in range(len(im_lst)):
    h = tan_alt.isel(mjd=im_lst[i], pixel=tan_alt.pixel[l1.notnull().isel(mjd=im_lst[i])])
    K = pathl1d_iris(h, z, z_top)    
    y = l1.isel(mjd=im_lst[i], pixel=tan_alt.pixel[l1.notnull().isel(mjd=im_lst[i])]).data
#    Se = np.diag(np.ones(len(y))) * (1e12)**2
    Se = np.diag(np.ones(len(y))) *30
#    Se = np.diag(error.data[i,:]**2)
    x, A, Ss, Sm = linear_oem(K, Se, Sa, y, xa)
    result_1d[i,:] = x
    Ave.append(A.sum(axis=1)) #sum over rows 
    resi.extend(y-K.dot(x))


result_1d = xr.DataArray(result_1d, 
                         coords=(mjd[im_lst], z), 
                         dims=('mjd', 'z'))
result_1d.attrs['units'] = 'photons m-3 s-1 ?'
result_1d.attrs['long_name'] = '1d inversion VER'
Ave = np.array(Ave)
mr_threshold = 0.8
result_1d_mean = result_1d.where(Ave>mr_threshold).mean(dim='mjd')
result_1d_pre = result_1d
result_1d_mean_pre = result_1d_mean

xa = result_1d_mean_pre.data
Sa = np.diag(np.ones(len(z))) *1e14 #temp
#Se = np.diag(np.ones(len(pixel))) * 30# 1e10 #temporary
result_1d = np.zeros((len(im_lst), len(z)))
Ave = []
for i in range(len(im_lst)):
    h = tan_alt.isel(mjd=im_lst[i], pixel=tan_alt.pixel[l1.notnull().isel(mjd=im_lst[i])])
    K = pathl1d_iris(h, z, z_top)    
    y = l1.isel(mjd=im_lst[i], pixel=tan_alt.pixel[l1.notnull().isel(mjd=im_lst[i])]).data    
#    Se = np.diag(error.data[i,:]**2)
    Se = np.diag(np.ones(len(y))) * (1e12)**2
    x, A, Ss, Sm = linear_oem(K, Se, Sa, y, xa)
    result_1d[i,:] = x
    Ave.append(A.sum(axis=1)) #sum over rows 

result_1d = xr.DataArray(result_1d, 
                         coords=(mjd[im_lst], z), 
                         dims=('mjd', 'z'))
result_1d.attrs['units'] = 'photons cm-3 s-1 ?'
result_1d.attrs['long_name'] = '1d inversion VER'
Ave = np.array(Ave)
result_1d_mean = result_1d.where(Ave>mr_threshold).mean(dim='mjd')

#%%
#==== plot residual
plt.figure()
plt.plot(np.array(resi).ravel())
plt.ylabel('y-Kx')

#==== plot VER contour
#result_1d = abs(result_1d)
result_1d.plot(x='mjd', y='z', 
#               norm=LogNorm(), 
               vmin=0, vmax=8e6, 
               size=3, aspect=3)
ax = plt.gca()
ax.set(title='1d retrieved VER',
      xlabel='tangent point along track distance from iris')
#ax.set_xticks(mjd[im_lst])
#ax.set_xticklabels(np.round(tan_beta.sel(pixel=60).data*Re))
ax.set_xticks(mjd[im_lst[::label_interval]])
ax.set_xticklabels(im_lst[::label_interval])
plt.show()

#==== plot VER in 1D
plt.figure()
ax = plt.gca()
ax.plot(result_1d.T, z, '.')
result_1d_mean.plot(y='z', color='k',ls='-',
                    label='averaged profile with sum(A)>.{}'.format(mr_threshold))
ax.set_xscale('linear')
ax.set(#xlim=[1e4, 1e8],
       xlabel=result_1d.units, 
       ylabel='altitdue grid',
       title='1d retrieval')
ax.legend(loc='upper left')
plt.show()

#==== plot averaging kernel
plt.plot(Ave.T, z, '*')
plt.xlabel('Averaging kernel sum over rows')
plt.ylabel('altitude grid')
plt.title('Measurement response')
plt.xlim([mr_threshold, 1.2])
plt.axvline(x=mr_threshold, ls=':', color='k')
#plt.text(mr_threshold, z[-1], 'threshold')
plt.show()


#%% Ozone
from scipy.io import loadmat
MSIS = loadmat('msisdata.mat')
zMsis = MSIS['zMsis'].squeeze() # in km
TMsis = MSIS['TMsis'] # in K
NMsis = MSIS['NMsis'] # in cm-3 
monthMsis = MSIS['monthMsis'].squeeze()
latMsis = MSIS['latMsis'].squeeze()
month = 6 #temp
lat = 1 #temp
T = interp1d(zMsis*1e3, TMsis[:,month,lat], fill_value='extrapolate')(z)
m = interp1d(zMsis*1e3, NMsis[:,month,lat], fill_value='extrapolate')(z)

zenithangle = 30
from chemi import gfactor
gA = gfactor(0.21*m, T, z, zenithangle)

#%% onion peel
from chemi import ozone_sme, ozone_mlynczak, oxygen_atom
from scipy.io import loadmat
from geometry_functions import pathleng
sigma = loadmat('sigma.mat')
sO = sigma['sO'].squeeze() #sigma [cm2]
sO2 = sigma['sO2'].squeeze() #sigma [cm2]
sO3 = sigma['sO3'].squeeze() #sigma [cm2]
sN2 = sigma['sN2'].squeeze() #sigma [cm2]
irrad = sigma['irrad'].squeeze() #irradiance [cm-2 s-1]
wave = sigma['wave'].squeeze() #wavelength grid [nm]
hartrange = (wave > 210) & (wave < 310)
srcrange = (wave > 122) & (wave < 175)
lyarange = 28  # wavelength = 121.567 nm
zenithangle = 0 #temp
pathl = pathleng(z, zenithangle) * 1e2  # [m -> cm]

O2 = 0.21 * m # molec cm-3 
N2 = 0.78 * m # molec cm-3

im = 5
O2sd_ver = result_1d_pre[im]
O2sd_ver = np.abs(O2sd_ver)
O3 = np.zeros(z.shape)
Jhart = np.zeros(z.shape)
jhart_max = 8e-3
jlya_max = 4.5e-9
jsrc_max = 6.3e-6

# top layer with Jhart max
Jhart[-1] = jhart_max
O3[-1] = ozone_sme(m[-1], T[-1], O2sd_ver[-1], 
                         jhart=Jhart[-1], js=gA[-1])

# from the second top layers ......
for i in range(len(z)-2, -1, -1):
   
    tau_o2 = sO2 * O2.dot(pathl[i+1,:])
    tau_n2 = sN2 * N2.dot(pathl[i+1,:])
    tau_o3 = sO3 * O3.dot(pathl[i+1,:])
    tau = tau_o2 + tau_n2 + tau_o3
    #tau = tau_o3
    

    jO3 = irrad * sO3 * np.exp(-tau)
       
    jhart = jO3[hartrange].sum()
    Jhart[i] = jhart
    O3[i] = ozone_sme(m[i], T[i], O2sd_ver[i], jhart=Jhart[i], js=gA[i])
    
    if jhart < 1e-4:
        break
O3_thomas = O3

O3 = np.zeros(z.shape)
O = np.zeros(z.shape)
Jhart = np.zeros(z.shape)
Jlya = np.zeros(z.shape)
Jsrc = np.zeros(z.shape)
jhart_max = 8e-3
jlya_max = 4.5e-9
jsrc_max = 6.3e-6

# top layer with max
Jhart[-1] = jhart_max
Jlya[-1] = jlya_max
Jsrc[-1] = jsrc_max
O3[-1] = ozone_mlynczak(O2sd_ver[-1], T[-1], m[-1], 0, 0, 
                        Jhart[-1], Jlya[-1], Jsrc[-1], gA[-1])

# from the second top layers ......
for i in range(len(z)-2, -1, -1):
   
    tau_o2 = sO2 * O2.dot(pathl[i+1,:])
    tau_n2 = sN2 * N2.dot(pathl[i+1,:])
    tau_o3 = sO3 * O3.dot(pathl[i+1,:])
    tau_o = sO * O.dot(pathl[i+1,:])
    tau = tau_o2 + tau_n2 + tau_o3 + tau_o
    #tau = tau_o3    

    jO3 = irrad * sO3 * np.exp(-tau)
    jO2 = irrad * sO2 * np.exp(-tau)
    jhart = jO3[hartrange].sum()
    jlya = jO2[lyarange].sum()
    jsrc = jO2[srcrange].sum()
    j3 = jO3.sum()
        
    Jhart[i] = jhart
    Jlya[i] = jlya
    Jsrc[i] = jsrc
    O3[i] = ozone_mlynczak(O2sd_ver[i], T[i], m[i], O[i], O3[i], 
                           Jhart[i], Jlya[i], Jsrc[i], gA[i])
    
    O[i] = oxygen_atom(m[i], T[i], O3[i], j3)
    
    if jhart < 1e-5:
        break
        
O3_mlynczak = O3

plt.figure()
plt.plot(O3_thomas, z, '-', label='thomas (SME)')
plt.plot(O3_mlynczak, z, '--', label='mlynczak (SABER)')
plt.xlabel('ozone (cm-3)')
plt.ylabel('z')
ax = plt.gca()
ax.set_xscale('linear')
plt.legend()
plt.show()

#%% lsq fit
from chemi import cal_o2delta, cal_o2delta_thomas
def residual(o3, T, m, z, zenithangle, gA, o2delta_meas):
    o2delta_model = cal_o2delta_thomas(o3, T, m, z, zenithangle, gA)    
    return o2delta_meas - o2delta_model

from scipy.optimize import least_squares
#npzfile = np.load('notebooks/ozone_profiles.npz')
#o3_initial = npzfile['arr_0'] # cm-3
#z_initial = npzfile['arr_1'] # m
#o3_init = interp1d(z_initial, o3_initial, fill_value='extrapolate')(z) # cm-3
o3_init = O3_mlynczak
o2delta_meas = result_1d.isel(mjd=5) / 2.58e-4 #cm-3
res_lsq = least_squares(residual, o3_init, bounds=(0, np.inf), verbose=1, 
                        args=(T, m, z, zenithangle, gA, o2delta_meas))
o3_thomas = res_lsq.x


def residual(o3, T, m, z, zenithangle, gA, o2delta_meas):
    o2delta_model = cal_o2delta(o3, T, m, z, zenithangle, gA)    
    return o2delta_meas - o2delta_model

from scipy.optimize import least_squares
o3_init = O3_mlynczak
o2delta_meas = result_1d.isel(mjd=5) / 2.58e-4 #cm-3
res_lsq = least_squares(residual, o3_init, bounds=(0, np.inf), verbose=1, 
                        args=(T, m, z, zenithangle, gA, o2delta_meas))
o3_mlynczak = res_lsq.x

plt.figure()
plt.semilogx(o3_thomas,z, o3_mlynczak, z)
plt.xlim([1e7, 1e11])
plt.figure()
plt.semilogx(O3_thomas,z, O3_mlynczak,z)
plt.xlim([1e7, 1e11])