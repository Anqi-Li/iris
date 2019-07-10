#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:42:44 2019

@author: anqil
"""
import requests 
import numpy as np
import json
import matplotlib.pylab as plt
from osirisl1services.readlevel1 import open_level1_ir
from osirisl1services.services import Level1Services
import sys
sys.path.append('..')
import xarray as xr
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
#orbit = 20900
#orbit = 22015
#orbit = 22643
orbit = 37586

ir = open_level1_ir(orbit, channel, valid=False)
tan_alt = ir.l1.altitude.sel(pixel=slice(14, 128))
tan_lat = ir.l1.latitude.sel(pixel=slice(14, 128))
tan_lon = ir.l1.longitude.sel(pixel=slice(14, 128))
sc_look = ir.l1.look_ecef.sel(pixel=slice(14, 128))
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
    
#plt.figure()
#plt.contourf(np.tile(mjd_smr, (51,1)), z_smr.T, o3_vmr.T)
#plt.pcolor(mjd_smr, z_smr[0,:], o3_smr.T, norm=LogNorm(vmin=1e4, vmax=1e9))

#%% clip iris data
#====drop data below and above some altitudes
top = 110e3
bot = 60e3
l1 = l1.where(tan_alt<top).where(tan_alt>bot)
#====retireval grid
z = np.arange(bot, top, 1e3) # m
z_top = z[-1] + 2e3

day_mjd_lst = mjd[sza<90]


#%% 1D inversion and retrieve ozone
from chemi import cal_o2delta, cal_o2delta_thomas
from scipy.optimize import least_squares
from oem_functions import linear_oem
from geometry_functions import pathl1d_iris
from chemi import gfactor

def residual(o3, T, m, z, zenithangle, gA, o2delta_meas):
    o2delta_model = cal_o2delta(o3, T, m, z, zenithangle, gA)    
    return o2delta_meas - o2delta_model

gA_table = np.load('gA_table.npz')['gA']
z_table = np.load('gA_table.npz')['z']
sza_table = np.load('gA_table.npz')['sza']
month_table = np.load('gA_table.npz')['month']
#mr = np.zeros((len(day_mjd_lst), len(z)))
#resi = []
#result_1d = np.zeros((len(day_mjd_lst), len(z)))
#o3_iris = np.zeros((len(day_mjd_lst), len(z)))
#all_xa = np.zeros((len(day_mjd_lst), len(z)))
#ver_error = np.zeros((len(day_mjd_lst), len(z)))
#fr = 0.5 # filter fraction 
#normalize = np.pi*4 / fr
#for i in range(len(day_mjd_lst)):
#    try:
#        print(i, 'out of', len(day_mjd_lst))
#        #match the closest scan of smr 
#        closest_scan_idx = (np.abs(mjd_smr - day_mjd_lst[i])).argmin()
#        o3_SMR_a = interp1d(z_smr[closest_scan_idx,:], o3_smr_a[closest_scan_idx,:],
#                           fill_value="extrapolate")(z)
#        T_SMR = interp1d(z_smr[closest_scan_idx,:], T_smr[closest_scan_idx,:],
#                         fill_value="extrapolate")(z)
#        m_SMR = interp1d(z_smr[closest_scan_idx,:], m[closest_scan_idx,:],
#                         fill_value="extrapolate")(z)
##        gA = gfactor(0.21*m_SMR, T_SMR, z, sza.sel(mjd=day_mjd_lst[i]).item())
#        gA = interp1d(z_table, 
#                      gA_table[:,(np.abs(month_table - start_month)).argmin(), 0,
#                               (np.abs(sza_table - sza.sel(mjd=day_mjd_lst[i]).item())).argmin()])(z)
#        
#        xa = cal_o2delta(o3_SMR_a, T_SMR, m_SMR, z, sza.sel(mjd=day_mjd_lst[i]).item(), gA) * A_o2delta
#        Sa = np.diag(xa**2)
#        h = tan_alt.sel(mjd=day_mjd_lst[i], pixel=pixel[l1.notnull().sel(mjd=day_mjd_lst[i])])
#        K = pathl1d_iris(h, z, z_top) * 1e2 # m-->cm    
#        y = l1.sel(mjd=day_mjd_lst[i], pixel=pixel[l1.notnull().sel(mjd=day_mjd_lst[i])]).data *normalize
#        Se = np.diag(np.ones(len(y))) *(1e11*normalize)**2
#
#        x, A, Ss, Sm = linear_oem(K, Se, Sa, y, xa)
#        result_1d[i,:] = x
#        ver_error[i,:] = np.diag(Sm)
#        mr[i,:] = A.sum(axis=1) #sum over rows 
#        all_xa[i,:] = xa
#        resi.extend(y-K.dot(x))
#        #    resi[i,:] = (y-K.dot(x))
#        
#        #lsq fit to get ozone
#        o2delta_meas = x / A_o2delta # cm-3?
#        res_lsq = least_squares(residual, o3_SMR_a, bounds=(-np.inf, np.inf), verbose=1, 
#                                args=(T_SMR, m_SMR, z, sza.sel(mjd=day_mjd_lst[i]).item(), gA, o2delta_meas))
#        o3_iris[i,:] = res_lsq.x
#    except:
#        print('error occur in ', i)
#        result_1d[i,:] = np.ones(len(z)) * np.nan
#        mr[i,:] = np.ones(len(z)) * np.nan
#        o3_iris[i,:] = np.ones(len(z)) * np.nan
#        pass

# organize resulting arrays and save in nc file
#result_1d = xr.DataArray(result_1d, 
#                         coords=(day_mjd_lst, z), 
#                         dims=('mjd', 'z'), name='VER')
#result_1d.attrs['units'] = 'photons cm-3 s-1'
#mr = np.array(mr)
#mr_threshold = 0.9
##result_1d_mean = result_1d.where(mr>mr_threshold).mean(dim='mjd')
#ds = xr.Dataset({'ver': result_1d, 
#                 'ver_error':(['mjd', 'z'], ver_error), 
#                 'mr':(['mjd', 'z'], mr), 
#                 'o3_iris':(['mjd', 'z'], o3_iris),
#                 'ver_xa': (['mjd', 'z'], all_xa)})
#ds.to_netcdf('ver_o3_{}.nc'.format(orbit))

fr = 0.5 # filter fraction 
normalize = np.pi*4 / fr
from multiprocessing import Pool
def f(i):
    print(i, 'out of', len(day_mjd_lst))
    #match the closest scan of smr 
    closest_scan_idx = (np.abs(mjd_smr - day_mjd_lst[i])).argmin()
    o3_SMR_a = interp1d(z_smr[closest_scan_idx,:], o3_smr_a[closest_scan_idx,:],
                       fill_value="extrapolate")(z)
    T_SMR = interp1d(z_smr[closest_scan_idx,:], T_smr[closest_scan_idx,:],
                     fill_value="extrapolate")(z)
    m_SMR = interp1d(z_smr[closest_scan_idx,:], m[closest_scan_idx,:],
                     fill_value="extrapolate")(z)
#        gA = gfactor(0.21*m_SMR, T_SMR, z, sza.sel(mjd=day_mjd_lst[i]).item())
    gA = interp1d(z_table, 
                  gA_table[:,(np.abs(month_table - start_month)).argmin(), 0,
                           (np.abs(sza_table - sza.sel(mjd=day_mjd_lst[i]).item())).argmin()])(z)
    
    xa = cal_o2delta(o3_SMR_a, T_SMR, m_SMR, z, sza.sel(mjd=day_mjd_lst[i]).item(), gA) * A_o2delta
    Sa = np.diag(xa**2)
    h = tan_alt.sel(mjd=day_mjd_lst[i], pixel=pixel[l1.notnull().sel(mjd=day_mjd_lst[i])])
    K = pathl1d_iris(h, z, z_top) * 1e2 # m-->cm    
    y = l1.sel(mjd=day_mjd_lst[i], pixel=pixel[l1.notnull().sel(mjd=day_mjd_lst[i])]).data *normalize
    Se = np.diag(np.ones(len(y))) *(1e11*normalize)**2

    x, A, Ss, Sm = linear_oem(K, Se, Sa, y, xa)

    #lsq fit to get ozone
    o2delta_meas = x / A_o2delta # cm-3?
    res_lsq = least_squares(residual, o3_SMR_a, bounds=(-np.inf, np.inf), verbose=0, 
                            args=(T_SMR, m_SMR, z, sza.sel(mjd=day_mjd_lst[i]).item(), gA, o2delta_meas))
#    o3_iris[i,:] = res_lsq.x
    
    return x, xa, np.diag(Sm), A.sum(axis=1), res_lsq.x, o3_SMR_a

with Pool(processes=4) as pool:
    result = np.array(pool.map(f, range(len(day_mjd_lst)))) #len(day_mjd_lst)

# organize resulting arrays and save in nc file
result_1d = xr.DataArray(result[:,0,:], 
                         coords=(day_mjd_lst[:4], z), 
                         dims=('mjd', 'z'), name='VER', attrs={'units': 'photons cm-3 s-1'})
ds = xr.Dataset({'ver': result_1d, 
                 'ver_apriori': (['mjd', 'z'], result[:,1,:], {'units': 'photons cm-3 s-1'}),
                 'ver_error':(['mjd', 'z'], result[:,2,:], {'units': '(photons cm-3 s-1)**2'}), 
                 'mr':(['mjd', 'z'], result[:,3,:]), 
                 'o3_iris':(['mjd', 'z'], result[:,4,:], {'units': 'molecule cm-3'}),
                 'o3_xa': (['mjd', 'z'], result[:,5,:], {'units': 'molecule cm-3'})})
ds.to_netcdf('ver_o3_{}.nc'.format(orbit))

##==== plot residual
#label_interval = 300
#plt.figure()
#plt.plot(np.array(resi).ravel())
#plt.ylabel('y-Kx')
#
##==== plot VER contour
##result_1d = abs(result_1d)
#result_1d.plot(x='mjd', y='z', 
##               norm=LogNorm(), 
#               vmin=0, vmax=8e6, 
#               size=3, aspect=3)
#ax = plt.gca()
#ax.set(title='IRIS 1d retrieved VER',
#      xlabel='tangent point along track distance from iris')
#plt.show()
#
##==== plot VER in 1D
#plt.figure()
#ax = plt.gca()
#ax.plot(result_1d.T, z, '*')
#ax.plot(xa, z, '-', label='apriori')
#result_1d_mean.plot(y='z', color='k',ls='-',
#                    label='averaged profile with mr>.{}'.format(mr_threshold))
#ax.set_xscale('log')
#ax.set(#xlim=[1e4, 1e8],
#       xlabel=result_1d.units, 
#       ylabel='altitdue grid',
#       title='IRIS 1d retrieval')
#ax.legend(loc='upper right')
#plt.show()
#
##==== plot averaging kernel
#plt.plot(Ave.T, z, '-')
#plt.xlabel('Averaging kernel sum over rows')
#plt.ylabel('altitude grid')
#plt.title('Measurement response')
##plt.xlim([mr_threshold, 1.2])
#plt.axvline(x=mr_threshold, ls=':', color='k')
#plt.text(mr_threshold, z[-1], 'threshold')
#plt.show()

##==== plot ozone 
#plt.figure()
#plt.semilogx(o3_iris.T, z)

#%%
#i = 1000
#closest_smr_scan_idx = (np.abs(mjd_smr - day_mjd_lst[i])).argmin()
#
#fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True, figsize=(18,6))
##ax[0].pcolor(mjd_smr, z_smr[0,:], o3_smr.T, norm=LogNorm(vmin=1e4, vmax=1e9))
#ax[0].pcolor(np.tile(mjd_smr,(len(z_smr.T),1)), z_smr.T, o3_smr.T, norm=LogNorm(vmin=1e4, vmax=1e9))
#CS = ax[0].contour(mjd_smr, z_smr[0,:], mr_smr.T, levels=[0.8, 1.2])
##ax[0].clabel(CS, inline=1, fontsize=10)
#ax[0].set(ylim=[60e3, 110e3], 
#          title='SMR')
#ax[0].axvline(x=mjd_smr[closest_smr_scan_idx], color='r')
#
#im = ax[1].pcolor(day_mjd_lst[:-2], z, o3_iris[:-2].T, norm=LogNorm(vmin=1e4, vmax=1e9))
##ax[1].contour(day_mjd_lst, z, mr.T, levels=[0.6,0.8, 1.2])
#ax[1].set(title='IRIS')
#ax[1].axvline(x=day_mjd_lst[i], color='r')
#
#fig.colorbar(im, ax=ax.ravel().tolist(), label='Ozone number density /cm-3')
#plt.rcParams.update({'font.size': 12})
#
##    path = '
##    np.savefig()
#
##    plt.figure(figsize=(15,2))
##    plt.plot(mjd, sza)
##    plt.axhline(y=90)
##    plt.ylabel('sza')
#
#fig, ax = plt.subplots(ncols=2, nrows=1, sharey=True, figsize=(10,5))
#ax[0].semilogx(o3_iris[i,mr[i]>mr_threshold], z[mr[i]>mr_threshold], '*', 
#              label='iris (mr>{})'.format(mr_threshold))
#ax[0].semilogx(o3_smr[closest_smr_scan_idx,mr_smr[closest_smr_scan_idx]>mr_threshold], 
#              z_smr[closest_smr_scan_idx,mr_smr[closest_smr_scan_idx]>mr_threshold], '*',
#              label='smr (mr>{})'.format(mr_threshold))
#ax[0].set_xlim(left=1e4)
#ax[0].set(title='Number density',
#          xlabel='cm-3',
#          ylabel='Altitude')
#ax[0].legend()
##plt.title(num2date(day_mjd_lst[i], units))
#
#m_SMR = interp1d(z_smr[closest_smr_scan_idx,:], m[closest_scan_idx,:],
#                fill_value="extrapolate")(z[mr[i]>mr_threshold])
#ax[1].semilogx(o3_iris[i,mr[i]>mr_threshold]/m_SMR*1e6, z[mr[i]>mr_threshold], '*',
#              label='iris (mr>{})'.format(mr_threshold))
#ax[1].semilogx(o3_vmr[closest_smr_scan_idx,mr_smr[closest_smr_scan_idx]>mr_threshold]*1e6, 
#              z_smr[closest_smr_scan_idx,mr_smr[closest_smr_scan_idx]>mr_threshold], '*',
#              label='smr (mr>{})'.format(mr_threshold))
#ax[1].set_xlim(left=1e-2)
#ax[1].set(title='volume mixing ratio',
#          xlabel='ppmv')
#ax[1].legend()
#plt.rcParams.update({'font.size': 16})

#%%
#grid = plt.GridSpec(ncols=2 ,nrows=3, hspace=0.5, wspace=0.2)
#plt.rcParams.update({'font.size': 14})
#for i in range(800,900):#len(day_mjd_lst)):
#    print(i, 'out of', len(day_mjd_lst))
#    closest_smr_scan_idx = (np.abs(mjd_smr - day_mjd_lst[i])).argmin()
#    
#    fig = plt.figure(figsize=(15,10))
#    ax0 = fig.add_subplot(grid[0,:])
#    ax1 = fig.add_subplot(grid[1,:],  sharex=ax0)
#    ax2 = fig.add_subplot(grid[2,0])
#    ax3 = fig.add_subplot(grid[2,1],  sharey=ax2)
#
#    ax0.pcolor(np.tile(mjd_smr,(len(z_smr.T),1)), z_smr.T*1e-3, o3_smr.T, norm=LogNorm(vmin=1e5, vmax=1e9))
#    ax0.axvline(x=mjd_smr[closest_smr_scan_idx], color='r')
#    CS = ax0.contour(mjd_smr, z_smr[0,:]*1e-3, mr_smr.T, levels=[0.8, 1.2])
#    #ax[0].clabel(CS, inline=1, fontsize=10)
#    ax0.set(ylim=[60, 110], 
#              title='SMR',
#              ylabel='Altitude / km')
#    
#    im = ax1.pcolor(day_mjd_lst[:-2], z*1e-3, o3_iris[:-2].T, norm=LogNorm(vmin=1e5, vmax=1e9))
#    ax1.axvline(x=day_mjd_lst[i], color='r')
#    ax1.set(title='IRIS',
#            ylabel='Altitude / km')
#    fig.colorbar(im, ax=[ax0,ax1,ax2,ax3], label='Ozone number density /cm-3')
#    
#    ax2.semilogx(o3_smr_a[closest_smr_scan_idx], z_smr[closest_smr_scan_idx]*1e-3, 
#                 'k--', label='SMR a priori')
#    ax2.semilogx(o3_iris[i,mr[i]>mr_threshold], z[mr[i]>mr_threshold]*1e-3, '*', 
#                  label='IRIS (mr>{})'.format(mr_threshold))
#    ax2.semilogx(o3_smr[closest_smr_scan_idx,mr_smr[closest_smr_scan_idx]>mr_threshold], 
#                  z_smr[closest_smr_scan_idx,mr_smr[closest_smr_scan_idx]>mr_threshold]*1e-3, '*',
#                  label='SMR (mr>{})'.format(mr_threshold))
#    ax2.set_xlim(left=1e4, right=1e12)
#    ax2.set(title='Number density',
#              xlabel='cm-3',
#              ylabel='Altitude / km')
#    ax2.legend(loc='lower left')
#    
#    m_SMR = interp1d(z_smr[closest_smr_scan_idx,:], m[closest_smr_scan_idx,:],
#                    fill_value="extrapolate")(z[mr[i]>mr_threshold])
#    ax3.semilogx(o3_vmr_a[closest_smr_scan_idx]*1e6, z_smr[closest_smr_scan_idx]*1e-3, 'k--')
#    ax3.semilogx(o3_iris[i,mr[i]>mr_threshold]/m_SMR*1e6, z[mr[i]>mr_threshold]*1e-3, '*',
#                  label='IRIS (mr>{})'.format(mr_threshold))
#    ax3.semilogx(o3_vmr[closest_smr_scan_idx,mr_smr[closest_smr_scan_idx]>mr_threshold]*1e6, 
#                  z_smr[closest_smr_scan_idx,mr_smr[closest_smr_scan_idx]>mr_threshold]*1e-3, '*',
#                  label='SMR (mr>{})'.format(mr_threshold))
#    ax3.set_xlim(left=1e-2, right=1e1)
#    ax3.set(title='volume mixing ratio',
#              xlabel='ppmv')
##    ax3.legend(loc='lower left')
#    
#    
#    path = '/home/anqil/Documents/osiris_database/plots_for_presentations/Limb_workshop2019/Ozone/'
#    filename = 'ozone_{}_{}.png'.format(orbit, i)
#    plt.savefig(path+filename)
#    plt.close(fig)
    
#%%
#import cv2
#
#orbit = 22643
#path = '/home/anqil/Documents/osiris_database/plots_for_presentations/Limb_workshop2019/Ozone/'
#filename = 'ozone_pi_{}_{}.png'
#output_path = '/home/anqil/Documents/osiris_database/plots_for_presentations/Limb_workshop2019/'
#output_filename ='ozone_pi_{}.avi' 
#
#for i in range(800): 
#    file = path+filename.format(orbit, i)
#    img = cv2.imread(file)
#    
#    if i == 0:
#        height, width, layers = img.shape
#        fourcc = cv2.VideoWriter_fourcc(*'XVID')
#        video = cv2.VideoWriter(output_path+output_filename.format(orbit), fourcc, 60, (width, height))
#    video.write(img)
#cv2.destroyAllWindows()
#video.release()