#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 14:23:13 2019

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
A_delta = 2.23e-4#2.58e-4#2.23e-4 # radiative lifetime of singlet delta to ground state

#%% load IRIS data
#channel = 2
#orbit = 6434
#orbit = 22873
channel = 2
#orbit = 20900
#orbit = 22015
orbit = 22643

ir = open_level1_ir(orbit, channel, valid=False)
tan_alt = ir.l1.altitude.sel(pixel=slice(14, 128))
tan_lat = ir.l1.latitude.sel(pixel=slice(14, 128))
tan_lon = ir.l1.longitude.sel(pixel=slice(14, 128))
sc_look = ir.l1.look_ecef.sel(pixel=slice(14, 128))
sc_pos = ir.l1.position_ecef
l1 = ir.data.sel(pixel=slice(14, 128))
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

#%% clip data
#====drop data below and above some altitudes
top = 110e3
bot = 60e3
l1 = l1.where(tan_alt<top).where(tan_alt>bot)

# retireval grid
z = np.arange(bot, top, 1e3) # m
#z = np.concatenate((np.arange(bot, 90e3, 1e3), 
#                    np.arange(90e3, 100e3, 3e3), 
#                    np.arange(100e3, top, 5e3)))
z_top = z[-1] + 2e3

#im_lst = np.arange(300,350,5)
#im_lst = np.arange(2350,2425,5)
#im_lst = np.arange(2496,2527,5)
#im_lst = np.arange(1950,2000,5)
#im_lst = np.arange(1520, 1570,5)
im_lst = np.arange(1000, 1050, 1)
pix_lst = np.arange(22, 128)
im = 0 
label_interval = 10

print(num2date(ir.mjd[0],units))
print('number of images: ', len(im_lst))

#%% pixel -> altitude space and plot
alts_interp = np.arange(bot, top, .25e3)
data_interp = []

for (data, alt) in zip(l1, tan_alt):
    f = interp1d(alt, data, bounds_error=False)
    data_interp.append(f(alts_interp))
data_interp = xr.DataArray(data_interp, coords=(mjd, alts_interp), 
                           dims=('mjd', 'alt'))

#%%====plotting
fig, ax = plt.subplots(figsize=(18,5))
data_interp.plot(x='mjd', y='alt', 
                 norm=LogNorm(), 
                 vmin=1e9, vmax=1e13)

ax.set(title='From {} \n to {}, \n channel {}'.format(num2date(mjd[0],units),
          num2date(mjd[-1], units), channel))
ax.axvline(x=mjd[im_lst[0]], color='k', linewidth=1)
ax.axvline(x=mjd[im_lst[-1]], color='k', linewidth=1)


fig, ax = plt.subplots(ncols=2, nrows=1, sharey=True, figsize=(15,5))
data_interp.isel(mjd=im_lst).plot(x='mjd', y='alt', 
                 norm=LogNorm(), 
                 vmin=1e9, vmax=1e13, 
                 ax=ax[0])

data_interp.isel(mjd=im_lst).plot.line(y='alt', ax=ax[1], marker='*', ls=' ')
#plt.ylim([50e3, 100e3])
ax[1].legend([])
ax[1].set(ylabel=' ',
          xlim=[1e9, 1e13])
ax[1].set_xscale('log')
plt.show()

plt.figure()
sza.plot(x='mjd')
plt.show()

#
#plt.figure()
#data_interp.isel(mjd=im_lst).plot.line(y='alt')
#plt.ylim([50e3, 100e3])
#plt.legend([])



#%% load SMR Ozone
import requests 
import numpy as np
import json
import sqlite3 as sql
import matplotlib.pylab as plt

start = num2date(mjd[im_lst[0]]-1/24/60*5, units)
end = num2date(mjd[im_lst[-1]]-1/24/60*4, units)
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
baseurl = "http://odin.rss.chalmers.se/rest_api/v5/level2/development/"
scansurl = baseurl+"{0}/{1}/scans/?limit=1000&offset=0&"
scansurl += "start_time={2}&end_time={3}"
a = requests.get(scansurl.format(dataset,fm,start_date,end_date))
aaa = json.loads(a.text)
scanno = aaa['Data'][0]['ScanID']
product = "O3 / 545 GHz / 20 to 85 km"
baseurl = "http://odin.rss.chalmers.se/rest_api/v5/level2/development/"
scansurl = baseurl+"{0}/{1}/{2}/L2/?product={3}"
a = requests.get(scansurl.format(dataset,fm,scanno,product))
result = json.loads(a.text)['Data'][0]

o3_vmr_a = np.array(result['Apriori'])
o3_vmr = np.array(result['VMR'])
z_smr = np.array(result['Altitude'])
p_smr = np.array(result['Pressure'])
T_smr = np.array(result['Temperature'])
AVK_smr = np.array(result['AVK'])
error_vmr = np.array(result['ErrorTotal'])
Av = 6.023e23 #Avogadro's number: molec/mol
R = 8.31 # gas constant: J/mol/K
m = Av * p_smr / (R * T_smr) * 1e-6# number density of air
o3_smr = m * o3_vmr  # cm-3
o3_smr_a = m * o3_vmr_a # cm-3
error_smr = m * error_vmr #cm-3

o3_SMR = interp1d(z_smr, o3_smr, fill_value="extrapolate")(z)
o3_SMR_a = interp1d(z_smr, o3_smr_a, fill_value="extrapolate")(z)
T_SMR = interp1d(z_smr, T_smr, fill_value="extrapolate")(z)
m_SMR = interp1d(z_smr, m, fill_value="extrapolate")(z)
error_SMR = interp1d(z_smr, error_smr, fill_value="extrapolate")(z)

#%% 1D inversion 
from oem_functions import linear_oem
from geometry_functions import pathl1d_iris
from chemi import cal_o2delta, gfactor
import astropy.coordinates as coord
from astropy.time import Time
import astropy.units as u

loc = coord.EarthLocation(lon=tan_lon.isel(mjd=im_lst[im],pixel=60) * u.deg,
                          lat=tan_lat.isel(mjd=im_lst[im],pixel=60) * u.deg)
now = Time(ir.mjd.isel(mjd=im_lst[im]), format='mjd', scale='utc')
altaz = coord.AltAz(location=loc, obstime=now)
sun = coord.get_sun(now)
zenithangle = 90 - sun.transform_to(altaz).alt.deg

gA = gfactor(0.21*m_SMR, T_SMR, z, zenithangle)
xa = cal_o2delta(o3_SMR_a, T_SMR, m_SMR, z, zenithangle, gA) * A_delta
#xa = np.ones(len(z)) *0 # temp
#Sa = np.diag(np.ones(len(z))) *1e-9 #temp
Sa = np.diag(xa**2) #temp

#Sa = np.diag(xa**2) #temp
resi = []
result_1d = np.zeros((len(im_lst), len(z)))
mr = np.zeros(result_1d.shape)
for i in range(len(im_lst)):
    h = tan_alt.isel(mjd=im_lst[i]).sel(pixel=pixel[l1.notnull().isel(mjd=im_lst[i])])
    K = pathl1d_iris(h, z, z_top)    
    y = l1.isel(mjd=im_lst[i]).sel(pixel=pixel[l1.notnull().isel(mjd=im_lst[i])]).data
#    Se = np.diag(np.ones(len(y))) * 30
    Se = np.diag(np.ones(len(y))) *(1e11)**2
#    Se = np.diag(error.data[i,:]**2)
    x, A, Ss, Sm = linear_oem(K, Se, Sa, y, xa)
    result_1d[i,:] = x
    mr[i,:] = A.sum(axis=1) #sum over rows 
    resi.extend(y-K.dot(x))


result_1d = xr.DataArray(result_1d, 
                         coords=(mjd[im_lst], z), 
                         dims=('mjd', 'z'))
result_1d.attrs['units'] = 'photons cm-3 s-1 ?'
result_1d.attrs['long_name'] = '1d inversion VER'
mr_threshold = 0.9
result_1d_mean = result_1d.where(mr>mr_threshold).mean(dim='mjd')

#%%
#==== plot residual
plt.figure()
plt.plot(np.array(resi).ravel()[100:300])
plt.ylabel('y-Kx')

#==== plot VER contour
#result_1d = abs(result_1d)
#result_1d.plot(x='mjd', y='z', 
#               norm=LogNorm(), 
#               vmin=1e4, vmax=8e6, 
#               size=3, aspect=3)
plt.figure(figsize=(9,3))
plt.pcolormesh(result_1d.mjd, result_1d.z, np.ma.masked_where(mr.T<mr_threshold, result_1d.T),
               norm=LogNorm(), vmin=1e4, vmax=8e6)
ax = plt.gca()
ax.set(title='IRIS 1d retrieved VER',
      xlabel='mjd')
#ax.set_xticks(mjd[im_lst])
#ax.set_xticklabels(np.round(tan_beta.sel(pixel=60).data*Re))
ax.set_xticks(mjd[im_lst[::label_interval]])
ax.set_xticklabels(im_lst[::label_interval])
plt.show()

#==== plot VER in 1D
plt.figure()
ax = plt.gca()
ax.plot(result_1d.T, z, '*')
ax.plot(xa, z, '-', label='apriori')
result_1d_mean.plot(y='z', color='k',ls='-',
                    label='averaged profile with mr>.{}'.format(mr_threshold))
ax.set_xscale('log')
ax.set(#xlim=[1e4, 1e8],
       xlabel=result_1d.units, 
       ylabel='altitdue grid',
       title='IRIS 1d retrieval')
ax.legend(loc='upper right')
plt.show()

#==== plot measurement response
plt.plot(mr.T, z, '-*')
plt.xlabel('Averaging kernel sum over rows')
plt.ylabel('altitude grid')
plt.title('Measurement response')
plt.xlim([mr_threshold, 1.2])
plt.axvline(x=mr_threshold, ls=':', color='k')
#plt.text(mr_threshold, z[-1], 'threshold')
plt.show()


#%% lsq fit

from chemi import cal_o2delta, cal_o2delta_thomas
from scipy.optimize import least_squares

def residual(o3, T, m, z, zenithangle, gA, o2delta_meas):
    o2delta_model = cal_o2delta(o3, T, m, z, zenithangle, gA)    
    return o2delta_meas - o2delta_model
o3_init = interp1d(z_smr, o3_smr_a, fill_value="extrapolate")(z)
o2delta_meas = result_1d.isel(mjd=im) / A_delta # cm-3?
res_lsq = least_squares(residual, o3_init, bounds=(-np.inf, np.inf), verbose=1, 
                        args=(T_SMR, m_SMR, z, zenithangle, gA, o2delta_meas))
o3_iris = res_lsq.x

#%% compare
#ozone
f = 1 #temp iris offset factor 
plt.figure()
plt.plot(o3_iris[mr[im,:]>mr_threshold]/f, z[mr[im,:]>mr_threshold], '.',
         label='IRIS (mr>{})'.format(mr_threshold))
plt.errorbar(o3_smr[AVK_smr.sum(axis=1)>mr_threshold], z_smr[AVK_smr.sum(axis=1)>mr_threshold], #'*-',
                    xerr=error_smr[AVK_smr.sum(axis=1)>mr_threshold], 
                    color='r', ecolor='orange', label='SMR (mr>{})'.format(mr_threshold))
plt.plot(o3_smr_a, z_smr, color='k', ls='--', label='SMR apriori')
plt.legend()
ax = plt.gca()
ax.set_xscale('log')
ax.set(xlabel='ozone/ cm-3',
       ylabel='altitude/ m',
       title='compare SMR and IRIS ozone retrieval')
plt.xlim([o3_smr_a.min(), o3_smr_a.max()])
#plt.ylim([z_smr.min(), z_smr.max()])
#ax.set(xlim=(o3_smr.min()/f, o3_iris.max()/f),
#       ylim=(z.min(), z.max()))
plt.show()

plt.figure()
plt.plot(o3_iris[mr[im,:]>mr_threshold]/f/m_SMR[mr[im,:]>mr_threshold], 
         z[mr[im,:]>mr_threshold], '.', label='IRIS (mr>{})'.format(mr_threshold))
plt.errorbar(o3_vmr[AVK_smr.sum(axis=1)>mr_threshold], z_smr[AVK_smr.sum(axis=1)>mr_threshold], 
                xerr=error_vmr[AVK_smr.sum(axis=1)>mr_threshold], 
                color='r', ecolor='orange', label='SMR (mr>{})'.format(mr_threshold))
plt.plot(o3_vmr_a, z_smr, ls='--', color='k', label='SMR apriori')
plt.legend()
ax = plt.gca()
ax.set_xscale('linear')
ax.set(xlabel='ozone VMR',
       ylabel='altitude/ m',
#       xlim=((o3_iris/m_SMR).min(),(o3_iris/m_SMR).max()),
#       ylim=(z.min(), z.max()),
       title='compare SMR and IRIS ozone retrieval')
plt.show()

#pointing

plt.figure()
plt.plot(tan_lon.isel(mjd=im_lst[im]), tan_lat.isel(mjd=im_lst[im]), '*', label='iris')
plt.plot(result['Longitude'], result['Latitude'], '*', label='smr')
plt.legend()
plt.xlabel('lon')
plt.ylabel('lat')
plt.title('check colocation')
plt.show()

print('iris data taken in', num2date(mjd[im_lst[im]], units))
print('smr data taken in', num2date(result['MJD'], units))

#plt.figure()
#plt.subplot(121)
#plt.plot(tan_lat.isel(mjd=im_lst[im]), tan_alt.isel(mjd=im_lst[im]), '*', label='iris')
#plt.plot(result['Latitude'], result['Altitude'], '*', label='smr')
#plt.xlabel('latitude')
#plt.ylabel('altitude')
#plt.legend()
#
#plt.subplot(122)
#plt.plot(tan_lon.isel(mjd=im_lst[im]), tan_alt.isel(mjd=im_lst[im]), '*', label='iris')
#plt.plot(result['Longitude'], result['Altitude'], '*', label='smr')
#plt.xlabel('longitude')
##plt.ylabel('altitude')
#ax = plt.gca()
#ax.set_yticks([])
#plt.legend()

