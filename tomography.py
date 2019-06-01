#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 11:47:48 2019

@author: anqil
"""

import requests 
import numpy as np
import json
import sqlite3 as sql
import matplotlib.pylab as plt
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
A_o2delta = 2.23e-4#2.58e-4#2.23e-4 # radiative lifetime of singlet delta to ground state
#%% load IRIS data
#channel = 2
#orbit = 6434
#orbit = 22873
channel = 3
orbit = 20900
#orbit = 22015
#orbit = 22643

ir = open_level1_ir(orbit, channel, valid=False)
tan_alt = ir.l1.altitude.sel(pixel=slice(14, 128))
tan_lat = ir.l1.latitude.sel(pixel=slice(14, 128))
tan_lon = ir.l1.longitude.sel(pixel=slice(14, 128))
#sc_look = ir.l1.look_ecef.sel(pixel=slice(14, 128))
sc_pos = ir.l1.position_ecef
l1 = ir.data.sel(pixel=slice(14, 128)) #/np.pi
mjd = ir.mjd.data
pixel = ir.pixel.sel(pixel=slice(14, 128)).data

# calculate sza
import astropy.coordinates as coord
from astropy.time import Time
import astropy.units as u

loc = coord.EarthLocation(lon=tan_lon.sel(pixel=60) * u.deg,
                          lat=tan_lat.sel(pixel=60) * u.deg)
now = Time(ir.mjd, format='mjd', scale='utc')
altaz = coord.AltAz(location=loc, obstime=now)
sun = coord.get_sun(now)
sza = 90 - sun.transform_to(altaz).alt.deg
sza = xr.DataArray(sza, coords=(mjd,), dims=('mjd',), name='sza')

# re-calculate the look vectors (sc_look)
from geometry_functions import lla2ecef
import pandas as pd
tan_x, tan_y, tan_z = lla2ecef(tan_lat, tan_lon, tan_alt)
tan_xyz = xr.concat([tan_x, tan_y, tan_z],
                    pd.Index(['x', 'y', 'z'], name='xyz')).transpose('mjd', 'pixel', 'xyz')

norm = np.sqrt((tan_x-sc_pos.sel(xyz='x'))**2 +
                (tan_y-sc_pos.sel(xyz='y'))**2 +
                (tan_z-sc_pos.sel(xyz='z'))**2)

look = (tan_xyz - sc_pos)/norm

sc_look = look

#%% load smr whole orbit
start = num2date(mjd[0]-1/24/60*5, units)
end = num2date(mjd[-1]-1/24/60*4, units)
start_year = start.year
start_month = start.month
start_day = start.day
start_hour = start.hour
start_minute = start.minute
end_year = end.year
end_month = end.month
end_day = end.day
end_hour = end.hour
end_minute = end.minute

start_date = '{}-{}-{}%20{}%3A{}%3A00'.format(start_year, start_month, 
              start_day, start_hour, start_minute)
end_date = '{}-{}-{}%20{}%3A{}%3A59'.format(end_year, end_month, end_day, 
            end_hour, end_minute)

dataset = 'ALL'
fm = 2
product = "O3 / 545 GHz / 20 to 85 km"
baseurl = "http://odin.rss.chalmers.se/rest_api/v5/level2/development/"
scansurl = baseurl+"{0}/{1}/scans/?limit=1000&offset=0&"
scansurl += "start_time={2}&end_time={3}"
a = requests.get(scansurl.format(dataset,fm,start_date,end_date))
aaa = json.loads(a.text)

scanno_lst = np.zeros(len(aaa['Data']))
o3_vmr_a = np.zeros((len(aaa['Data']), 51))
o3_vmr = np.zeros((len(aaa['Data']), 51))
z_smr = np.zeros((len(aaa['Data']), 51))
mjd_smr = np.zeros((len(aaa['Data'])))
p_smr = np.zeros((len(aaa['Data']), 51))
T_smr = np.zeros((len(aaa['Data']), 51))
mr_smr = np.zeros((len(aaa['Data']), 51))
error_vmr = np.zeros((len(aaa['Data']), 51))

for i in range(len(aaa['Data'])):
    scanno_lst[i] = aaa['Data'][i]['ScanID']
    scansurl = aaa['Data'][i]['URLS']['URL-level2'] + 'L2/?product={}'.format(product)
    a = requests.get(scansurl)
    result = json.loads(a.text)['Data'][0]
    
    o3_vmr_a[i,:] = np.array(result['Apriori'])
    o3_vmr[i,:] = np.array(result['VMR'])
    z_smr[i,:] = np.array(result['Altitude'])
    mjd_smr[i] = result['MJD']
    p_smr[i,:] = np.array(result['Pressure'])
    T_smr[i,:] = np.array(result['Temperature'])
    mr_smr[i,:] = np.array(result['AVK']).sum(axis=1)
    error_vmr[i,:] = np.array(result['ErrorTotal'])

Av = 6.023e23 #Avogadro's number: molec/mol
R = 8.31 # gas constant: J/mol/K
m = Av * p_smr / (R * T_smr) * 1e-6 # number density of air cm-3
o3_smr = m * o3_vmr  # cm-3
o3_smr_a = m * o3_vmr_a # cm-3
error_smr = m * error_vmr #cm-3

#%% clip iris data
#====drop data below and above some altitudes
top = 110e3
bot = 60e3
l1 = l1.where(tan_alt<top).where(tan_alt>bot)
#====retireval grid
z = np.arange(bot, top, 1e3) # m
z_top = z[-1] + 2e3

day_mjd_lst = mjd[sza<90]
#im_lst = np.arange(1150,1250,1)
im_lst = np.arange(2350,2450,1)
#im_lst = np.arange(800,900,1)
pix_lst = np.arange(len(pixel))

#%% interpolation 
alts_interp = np.arange(60e3, 110e3, .25e3)
data_interp = []


for (data, alt) in zip(l1, tan_alt):
    f = interp1d(alt, data, bounds_error=False)
    data_interp.append(f(alts_interp))
data_interp = xr.DataArray(data_interp, coords=[l1.mjd, alts_interp],
                           dims=['mjd', 'altitude'])

#%%plotting
plt.figure(figsize=(15,6))
data_interp.plot(x='mjd', y='altitude', cmap='Spectral',
                 norm=LogNorm(), vmin=1e9, vmax=1e13)
plt.title(str(num2date(ir.mjd[0],units))+' channel '+str(channel))
plt.axvline(x=mjd[im_lst[0]], color='k', linewidth=5)
plt.axvline(x=mjd[im_lst[-1]], color='k', linewidth=5)
ax = plt.gca()
ax.set_xticks(mjd[np.arange(0,len(mjd),300, dtype=int)])
ax.set_xticklabels(np.arange(0,len(mjd),300))
ax.set(xlabel='image index')
plt.show()


plt.figure(figsize=(15,6))
data_interp.isel(mjd=im_lst).plot(x='mjd', y='altitude',  cmap='Spectral',
                 norm=LogNorm(), 
                 vmin=1e9, vmax=1e13)#, 
                 #size=5, aspect=3)
ax = plt.gca()
ax.set(title='zoom in im_lst',
      xlabel='image index')
plt.show()

plt.figure()
plt.plot(data_interp.isel(mjd=im_lst).T, alts_interp)
ax = plt.gca()
ax.set(ylabel='tangent altitude', 
       xlabel='limb radiance',
       title='limb radiance for selected images')

plt.figure()
tan_lat.isel(mjd=im_lst, pixel=60).plot()
ax = plt.gca()

ax.set(ylabel='tangent latitude')
plt.show()

#%% Tomography
#%% change coordinate
#====define the new base vectors
n_crosstrack = np.cross(sc_look.isel(mjd=im_lst[0], pixel=60),
                        sc_pos.isel(mjd=im_lst[0]))
n_vel = np.cross(sc_pos.isel(mjd=im_lst[0]), n_crosstrack)
n_zenith = sc_pos.isel(mjd=im_lst[0])

#====tangent points in alpha, beta, rho coordinate
import pandas as pd
from geometry_functions import lla2ecef, cart2sphe, change_of_basis
tan_ecef = xr.concat(lla2ecef(tan_lat,tan_lon,tan_alt), 
                     pd.Index(['x','y','z'], name='xyz'))

tan_alpha = []
tan_beta = []
tan_rho = []
for i in im_lst:
    p_old = tan_ecef.isel(mjd=i, pixel=pix_lst)
    p_new = change_of_basis(n_crosstrack, n_vel, n_zenith, p_old)
    alpha, beta, rho = cart2sphe(p_new.sel(xyz='x'),
                                 p_new.sel(xyz='y'),
                                 p_new.sel(xyz='z'))
    tan_alpha.append(alpha)
    tan_beta.append(beta)
    tan_rho.append(rho)
tan_alpha = xr.DataArray(tan_alpha, 
                         coords=[mjd[im_lst], pixel[pix_lst]],
                         dims=['mjd', 'pixel'])
tan_beta = xr.DataArray(tan_beta, 
                        coords=[mjd[im_lst], pixel[pix_lst]],
                        dims=['mjd', 'pixel'])
tan_rho = xr.DataArray(tan_rho, 
                       coords=[mjd[im_lst], pixel[pix_lst]],
                       dims=['mjd', 'pixel'])

Re = 6371 + 80 #Earth radius in km

#%% Tomo grid
#====define atmosphere grid (the bin edges)
edges_alpha = np.linspace(tan_alpha.min()-0.01,
                          tan_alpha.max()+0.01, 2) #radian
edges_beta = np.arange(tan_beta.min()-0.1,
                         tan_beta.max()+0.15, 0.02) #radian
edges_rho = np.append(z,z_top) # meter
edges = edges_alpha, edges_beta, edges_rho

#====grid points for plotting
grid_alpha = np.append(edges_alpha - np.gradient(edges_alpha)/2, 
                       edges_alpha[-1]+np.gradient(edges_alpha)[-1]/2)
grid_beta = np.append(edges_beta - np.gradient(edges_beta)/2, 
                       edges_beta[-1]+np.gradient(edges_beta)[-1]/2)
grid_rho = np.append(edges_rho - np.gradient(edges_rho)/2, 
                       edges_rho[-1]+np.gradient(edges_rho)[-1]/2)

#%% cal Jacobian
shape_tomo = (len(grid_alpha), len(grid_beta), len(grid_rho))
#====num of columns & rows of jacobian
col_len = len(grid_alpha) * len(grid_beta) * len(grid_rho)
row_len = l1.isel(mjd=im_lst).notnull().sum().item()

#====measure pathlength in each bin
from geometry_functions import los_points_fix_dl
from oem_functions import jacobian_row

dl = 3e3 #fixed distance between all points
nop = 500 # choose number of points along the line
K_row_idx = []
K_col_idx = []
K_value = []
dll = dl * np.ones(nop) #temp
all_los_alpha, all_los_beta, all_los_rho = [], [], []
measurement_id = 0
for image in im_lst:
    #====generate points of los for all pixels in each image
    #====all points in cartesian coordinate relative to the space craft
    sc_look_new = change_of_basis(n_crosstrack, n_vel, n_zenith, 
                                  sc_look[image].T)
    sc_pos_new = change_of_basis(n_crosstrack, n_vel, n_zenith, 
                                 sc_pos[image])
    lx, ly, lz = los_points_fix_dl(sc_look_new, sc_pos_new, dl=dl, nop=nop)    
    #====convert xyz to alpha, beta, rho for all points
    los_alpha, los_beta, los_rho = cart2sphe(lx, ly, lz)
    all_los_alpha.append(los_alpha)
    all_los_beta.append(los_beta)
    all_los_rho.append(los_rho)
    
    #====build K
    for pix in l1.pixel[l1.notnull().isel(mjd=image)]: 
        #print(image, pix.data, measurement_id)
        los = los_alpha.sel(pixel=pix), los_beta.sel(pixel=pix), los_rho.sel(pixel=pix)
        measurement_idx, grid_idx, pathlength = jacobian_row(dll, edges, los, measurement_id)
        K_row_idx.append(measurement_idx)
        K_col_idx.append(grid_idx)
        K_value.append(pathlength)
        measurement_id += 1
        
        
K_row_idx = np.concatenate(K_row_idx).astype('int')
K_col_idx = np.concatenate(K_col_idx).astype('int')
K_value = np.concatenate(K_value) # in meter

#==== create sparse matrix
from scipy.sparse import coo_matrix
K_coo = coo_matrix((K_value, (K_row_idx, K_col_idx)), shape = (row_len, col_len))

#%% Tomo inversion
from oem_functions import linear_oem_sp
import scipy.sparse as sp
from chemi import cal_o2delta

y = l1.isel(mjd=im_lst).stack(msure=('mjd','pixel')).dropna('msure').data 
#y = abs(y)

gA_table = np.load('gA_table.npz')['gA']
z_table = np.load('gA_table.npz')['z']
sza_table = np.load('gA_table.npz')['sza']
month_table = np.load('gA_table.npz')['month']

closest_scan_idx = (np.abs(mjd_smr - mjd[im_lst[0]])).argmin()
o3_SMR_a = interp1d(z_smr[closest_scan_idx,:], o3_smr_a[closest_scan_idx,:],
                   fill_value="extrapolate")(grid_rho)
T_SMR = interp1d(z_smr[closest_scan_idx,:], T_smr[closest_scan_idx,:],
                 fill_value="extrapolate")(grid_rho)
m_SMR = interp1d(z_smr[closest_scan_idx,:], m[closest_scan_idx,:],
                 fill_value="extrapolate")(grid_rho)
#        gA = gfactor(0.21*m_SMR, T_SMR, z, sza.sel(mjd=day_mjd_lst[i]).item())
gA = interp1d(z_table, 
              gA_table[:,(np.abs(month_table - start_month)).argmin(), 0,
                       (np.abs(sza_table - sza.sel(mjd=mjd[im_lst[0]]).item())).argmin()])(grid_rho)
xa_1d = cal_o2delta(o3_SMR_a, T_SMR, m_SMR, grid_rho, sza.sel(mjd=mjd[im_lst[0]]).item(), gA) * A_o2delta
xa_1d = abs(xa_1d)
xa = np.tile(xa_1d, (len(grid_alpha),len(grid_beta),1)).ravel()
Sa = sp.diags((xa)**2, shape=(col_len, col_len))  #temp
Se = sp.diags([1], shape=(measurement_id, measurement_id))* (1e12)**2 #(5e12)**2 #temp
x_hat, G = linear_oem_sp(K_coo, Se, Sa, y, xa)


result_tomo = x_hat.reshape(shape_tomo)
result_tomo = xr.DataArray(result_tomo, 
                           coords=(grid_alpha, grid_beta, grid_rho), 
                           dims=('alpha', 'beta', 'rho')) #temp

A = G.dot(K_coo)
mr_tomo = np.zeros(shape_tomo)
for alpha_id in range(len(grid_alpha)):
    for beta_id in range(len(grid_beta)):
        for rho_id in range(len(grid_rho)):
            grid_id = np.ravel_multi_index((alpha_id, beta_id, rho_id), shape_tomo)
            mr_tomo[alpha_id, beta_id, rho_id] = A[grid_id,:].toarray().squeeze().reshape(shape_tomo).sum()
mr_tomo = xr.DataArray(mr_tomo, coords=result_tomo.coords, dims=result_tomo.dims)

#%% Plot tomo result
#====check residual
zoom = np.arange(100,150)
plt.figure()
plt.plot((K_coo.dot(x_hat)-y)[zoom])
plt.ylabel('residual')
plt.show()


#====contour plot 
plt.figure(figsize=(10,5))
ax = plt.gca()
result_tomo_masked = np.ma.masked_where(mr_tomo.isel(alpha=1)<0.8, result_tomo.isel(alpha=1))
main = ax.pcolor(grid_beta*Re, grid_rho, 
#                 result_tomo_masked.T,
                 result_tomo.isel(alpha=1).T, 
                 norm=LogNorm(vmin=1e5, vmax=1e7), cmap='Spectral') 
#                 size=3, aspect=3)
ax.set(xlabel='Distance along track / km',
       ylabel='Altitude / km')
#       title='slice along track')
plt.colorbar(main)
for i in range(0,len(im_lst),1):
    plt.axvline(x=tan_beta.sel(pixel=60)[i].data*Re, ymin=0.9, color='k')

CS = plt.contour(grid_beta*Re, grid_rho, mr_tomo.isel(alpha=1).T,
           levels=[0.8, 1, 1.5], colors=('w',), linestyles=('-',),linewidths=(2,))
plt.clabel(CS, inline=1, fontsize=10)
#ax.set(xlim=[(tan_beta.sel(pixel=60)*Re).min(), (tan_beta.sel(pixel=60)*Re).max()])
ax1 = ax.twiny()
ax1.set_xlim(ax.get_xlim())
ax1.set_xticklabels(np.round(ax.get_xticks()/Re,2))
ax1.set_xlabel('Angle along track / degrees')


#==== vertical profiles within the 3D matrix
plt.figure()
fig, ax = plt.subplots()
ax.set_xscale('linear')
ax.set(xlabel='VER',
      ylabel='altitude',
      title='tomography VER (all columns along track)')
for i in range(1,len(grid_alpha)-1):
    for j in range(1, len(grid_beta)-1):
        ax.semilogx(result_tomo[i,j,:], grid_rho*1e-3, '-')
ax.semilogx(xa_1d, grid_rho*1e-3, 'k-', label='a priori')
plt.legend()
plt.show()


