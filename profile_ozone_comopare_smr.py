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
#orbit = 22643
#orbit = 22101
orbit = 37585

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

#h_fov=2e3 #m
#v_fov=-tan_alt.diff(dim='pixel').mean()
#distance_sq = ((np.linalg.norm(sc_pos, axis=1)).mean()**2 - (6371e3+80e3)**2)
#solid_angle = h_fov*v_fov/distance_sq

#%% clip data
#====drop data below and above some altitudes
top = 110e3
bot = 60e3
l1 = l1.where(tan_alt<top).where(tan_alt>bot)

# retireval grid
z = np.arange(bot, top, 1e3) # m
z_top = z[-1] + 2e3

im_lst = np.arange(1000-50, 1050-50, 1)
#im_lst = np.arange(440,450,1)
pix_lst = np.arange(22, 128)
im = 0 
#label_interval = 10

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


#%% load SMR Ozone
import requests 
import numpy as np
import json
#import sqlite3 as sql
import matplotlib.pylab as plt

start = num2date(mjd[im_lst[0]]-1/24/60*5, units)
end = num2date(mjd[im_lst[-1]]-1/24/60*4, units)
#start = num2date(mjd[im_lst[0]], units)
#end = num2date(mjd[im_lst[-1]], units)
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
Sa = np.diag(xa**2) #temp
fr = 0.5 # filter fraction 
normalize = np.pi*4 / fr
resi = []
result_1d = np.zeros((len(im_lst), len(z)))
mr = np.zeros(result_1d.shape)
for i in range(len(im_lst)):
    h = tan_alt.isel(mjd=im_lst[i]).sel(pixel=pixel[l1.notnull().isel(mjd=im_lst[i])])
    K = pathl1d_iris(h, z, z_top) * 1e2 # m-->cm 
    y = l1.isel(mjd=im_lst[i]).sel(pixel=pixel[l1.notnull().isel(mjd=im_lst[i])]).data *normalize
    Se = np.diag(np.ones(len(y))) *(1e11*normalize)**2
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
plt.figure(figsize=(9,3))
plt.pcolormesh(result_1d.mjd, result_1d.z, np.ma.masked_where(mr.T<mr_threshold, result_1d.T),
               norm=LogNorm(), vmin=1e4, vmax=8e6)
ax = plt.gca()
ax.set(title='IRIS 1d retrieved VER',
      xlabel='mjd')
#ax.set_xticks(mjd[im_lst[::label_interval]])
#ax.set_xticklabels(im_lst[::label_interval])
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


#%% lsq fit for ozone
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

#%% load OS data

path = '/home/anqil/Documents/osiris_database/odin-osiris.usask.ca/Level2/CCI/OSIRIS_v5_10/'
filename = 'ESACCI-OZONE-L2-LP-OSIRIS_ODIN-SASK_V5_10_HARMOZ_ALT-{}{}-fv0002.nc'.format(start_year, str(start_month).zfill(2))
data_os = xr.open_dataset(path+filename)
o3_os = data_os.ozone_concentration * Av*1e-6 #molec cm-3
error_os = data_os.ozone_concentration_standard_error * Av*1e-6 # molec cm-3
m_os = data_os.pressure/data_os.temperature/8.314e4*Av #air mass density cm-3
vmr_os = o3_os / m_os
os_closest_scanno = abs(data_os.time - np.datetime64(Time(mjd[im_lst[im]], format='mjd').iso)).argmin()
data_os.longitude[np.where(data_os.longitude<0)]=data_os.longitude[np.where(data_os.longitude<0)]+360

#%% load mipas data
def sph_distance(lat1, lat2, lon1, lon2, r):
    d_central_angle=np.arccos(np.sin(lat1)*np.sin(lat2) + np.cos(lat1)*np.cos(lat2)*np.cos(abs(lon1-lon2)))
    return r*d_central_angle

path = '/misc/pearl/extdata/MIPAS/MA_UA_modes/MA/v5R/O3/V5R_O3_522/'
filename = 'MIPAS-E_IMK.{}{}.V5R_O3_522.nc'.format(start_year, str(start_month).zfill(2))
data_mipas = xr.open_dataset(path+filename)
m_mipas = data_mipas.pressure/data_mipas.temperature/8.314e4*Av #air mass density cm-3
#search measurements made within 1 hour 
mipas_scanno_range = np.where(abs(data_mipas.time 
                                   - np.datetime64(Time(mjd[im_lst[im]], format='mjd').iso)) <= np.timedelta64(59, 'm'))[0]

m_mipas = m_mipas.isel(timegrid=mipas_scanno_range, altgrid=slice(12,80))
vmr_o3_mipas = data_mipas.target.isel(timegrid=mipas_scanno_range, altgrid=slice(12,80))
vmr_error_mipas = data_mipas.target_noise_error.isel(timegrid=mipas_scanno_range, altgrid=slice(12,80))
o3_mipas = vmr_o3_mipas*1e-6*m_mipas
error_mipas = vmr_error_mipas*1e-6*m_mipas
z_mipas = data_mipas.altitude.isel(timegrid=mipas_scanno_range, altgrid=slice(12,80))
lon_mipas = data_mipas.longitude.isel(timegrid=mipas_scanno_range) % 360
lat_mipas = data_mipas.latitude.isel(timegrid=mipas_scanno_range)
t_mipas = data_mipas.time.isel(timegrid=mipas_scanno_range)


mipas_closest_scanno = sph_distance(lat_mipas, data_os.latitude.isel(time=os_closest_scanno),
                                     lon_mipas, data_os.longitude.isel(time=os_closest_scanno)%360, 1).argmin()
mipas_distance = sph_distance(lat_mipas, data_os.latitude.isel(time=os_closest_scanno),
                                     lon_mipas, data_os.longitude.isel(time=os_closest_scanno)%360, 6370).min()


#%% compare
#####################################3
#ozone number density
plt.figure()
plt.plot(o3_iris[mr[im,:]>mr_threshold], z[mr[im,:]>mr_threshold], '.',
         label='IRIS (mr>{})'.format(mr_threshold))
plt.plot(o3_smr_a, z_smr, color='gray', ls='--', label='SMR apriori')
plt.fill_betweenx(data_os.altitude*1e3, 
                  (o3_os-error_os).isel(time=os_closest_scanno),
                  (o3_os+error_os).isel(time=os_closest_scanno),
                  alpha=0.5, edgecolor='green', facecolor='green', label='OS')
plt.fill_betweenx(z_smr[AVK_smr.sum(axis=1)>mr_threshold],
                  (o3_smr-error_smr)[AVK_smr.sum(axis=1)>mr_threshold],
                  (o3_smr+error_smr)[AVK_smr.sum(axis=1)>mr_threshold],
                  alpha=0.5, edgecolor='orange', facecolor='orange', 
                  label='SMR (mr>{})'.format(mr_threshold))
plt.fill_betweenx(z_mipas.isel(timegrid=mipas_closest_scanno)*1e3,
                  (o3_mipas-error_mipas).isel(timegrid=mipas_closest_scanno),
                  (o3_mipas+error_mipas).isel(timegrid=mipas_closest_scanno),
                  alpha=0.5, edgecolor='k', facecolor='k', label='mipas {}km'.format(np.round(mipas_distance.data)))
plt.legend()
ax = plt.gca()
ax.set_xscale('log')
ax.set(xlabel='ozone/ cm-3',
       ylabel='altitude/ m',
       title='compare O3 number density')
plt.xlim(left=o3_smr_a.min())
plt.show()

###############################################
#ozone VMR
plt.figure()
plt.plot(1e6*o3_iris[mr[im,:]>mr_threshold]/m_SMR[mr[im,:]>mr_threshold], 
         z[mr[im,:]>mr_threshold], '.', label='IRIS (mr>{})'.format(mr_threshold))
plt.plot(1e6*o3_vmr_a, z_smr, ls='--', color='gray', label='SMR apriori')
plt.fill_betweenx(data_os.altitude*1e3,
                  1e6*(vmr_os-error_os/m_os).isel(time=os_closest_scanno),
                  1e6*(vmr_os+error_os/m_os).isel(time=os_closest_scanno),
                  alpha=0.5, edgecolor='green', facecolor='green', label='OS')
plt.fill_betweenx(z_smr[AVK_smr.sum(axis=1)>mr_threshold],
                  1e6*(o3_vmr-error_vmr)[AVK_smr.sum(axis=1)>mr_threshold],
                  1e6*(o3_vmr+error_vmr)[AVK_smr.sum(axis=1)>mr_threshold],
                  alpha=0.5, edgecolor='orange', facecolor='orange', 
                  label='SMR (mr>{})'.format(mr_threshold))
plt.fill_betweenx(z_mipas.isel(timegrid=mipas_closest_scanno)*1e3,
                  (vmr_o3_mipas-vmr_error_mipas).isel(timegrid=mipas_closest_scanno),
                  (vmr_o3_mipas+vmr_error_mipas).isel(timegrid=mipas_closest_scanno),
                  alpha=0.5, edgecolor='k', facecolor='k', label='mipas {}km'.format(np.round(mipas_distance.data)))
plt.legend(loc='upper right')
ax = plt.gca()
ax.set_xscale('linear')
ax.set(xlabel='ozone VMR (ppmv)',
       ylabel='altitude/ m',
       xlim=(-0.5,(1e6*o3_vmr_a).max()),
#       ylim=(z.min(), z.max()),
       title='compare SMR and IRIS ozone retrieval')
plt.show()


#%%
#% location
plt.figure()
plt.plot(tan_lon.isel(mjd=im_lst[im]), tan_lat.isel(mjd=im_lst[im]), 
         '*', color='C0', label='iris')
plt.plot(data_os.longitude.isel(time=os_closest_scanno) %360,
         data_os.latitude.isel(time=os_closest_scanno), 'g*', label='os')
plt.plot(result['Longitude'], result['Latitude'], '*', color='r', label='smr')
plt.plot(lon_mipas[mipas_closest_scanno],
         lat_mipas[mipas_closest_scanno], '*', color='k', label='mipas')

plt.legend()
plt.xlabel('lon')
plt.ylabel('lat')
plt.title('check location')
plt.show()

print('iris data taken in', num2date(mjd[im_lst[im]], units))
print('smr data taken in', num2date(result['MJD'], units))
print('os data taken in', data_os.time[os_closest_scanno].data)
print('mipas data taken in', t_mipas[mipas_closest_scanno].data)

