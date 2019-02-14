#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 15:28:20 2019

@author: anqil
"""

import numpy as np 
from pyproj import Proj, transform
import xarray as xr

ecef = Proj(proj='geocent', ellps='WGS84', datum='WGS84')
lla = Proj(proj='latlong', ellps='WGS84', datum='WGS84')

#%% path length 1d
def pathl1d_iris(h, z=np.arange(40e3, 110e3, 1e3), z_top=150e3):
    #z: retrieval grid in meter
    #z_top: top of the atmosphere under consideration 
    #h: tangent altitude of line of sight
    if z[1]<z[0]:
        z = np.flip(z) # retrieval grid has to be ascending
        print('z has to be fliped')
    
#    if h[1]<h[0]: # measred tangent alt grid has to be ascending
#        h = np.flip(h)
#        print('h has to be fliped')
    
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
    pathl = 2*pathl # in meter        
    
    return pathl


#%% copy paste from matlab code
def pathl1d_iris_matlab(h, z, z_top):
    if z[2]>z[1]:
        z = np.flip(z)
        
    if h[2]>h[1]:
        h = np.flip(h)
        
    Re = 6370e3
    z = np.append(z_top, z) + Re
    h = h + Re
    pl = np.zeros((len(h), len(z)-1))
    for i in range(len(h)):
        for j in range(len(z)-1):
            if z[j] > h[i]:
                pl[i,j] = np.sqrt(z[j]**2 - h[i]**2)
                
    pathl = np.append(pl[:, 1:], np.zeros((len(h), 1)), axis=1)
    pathl = pl - pathl
    pathl = 2*pathl
    
    return pathl

#%% geodetic to ecef coordinates
def geo2ecef(lat, lon, alt):
    # lat lon  in degree
    # alt in meter
    
    d2r = np.pi/180
    lat = lat*d2r
    lon = lon*d2r
    
    a = 6378137 #equatorial radius in meter
    b = 6356752 #polar radius in meter
    
    N_lat = a**2 /(np.sqrt(a**2 * np.cos(lat)**2 
                           + b**2 * np.sin(lat)**2))
    x = (N_lat + alt) * np.cos(lat) * np.cos(lon)
    y = (N_lat + alt) * np.cos(lat) * np.sin(lon)
    z = (b**2/a**2 * N_lat + alt) * np.sin(lat)
    return x,y,z


#%% generates query points along los using k_min and k_max method
def los_points(sc_pos, lat, lon, alt, nop=300):
    #returns ecef cordinate of all query points along line of sight (los) in meter
    #sc_pos: satellite position in ecef coordinate in meter
    #lat, lon, alt: tangent point(s) in geodetic coordinate in degree
    #nop: number of points along each los
    
    #k: proportion of the distance from satellite to tangent point(s)
#    k=np.linspace(0.6, 1.3, num=300)
    k_min, k_max = k_min_max(sc_pos, lat, lon, alt)
    
    k = np.linspace(k_min.min(), k_max.max(), nop)
    
    #tx, ty, tz: tangent point(s) in ecef coordinate in meter
#    tx, ty, tz = geo2ecef(lat,lon,alt)
    tx, ty, tz = lla2ecef(lat, lon, alt)

    sx = sc_pos.sel(xyz='x')
    sy = sc_pos.sel(xyz='y')
    sz = sc_pos.sel(xyz='z')
    if tx.shape == ty.shape:
        if ty.shape == tz.shape:
            lx = np.zeros((len(k), len(tx)))
            ly = np.zeros((len(k), len(tx)))
            lz = np.zeros((len(k), len(tx)))
            for nk in range(len(k)):
                lx[nk, :] = sx + k[nk] * (tx - sx)
                ly[nk, :] = sy + k[nk] * (ty - sy)
                lz[nk, :] = sz + k[nk] * (tz - sz)
        else:
            print('shape of ty is not equal to tz')
    else:
        print('shape of tx is not equal to ty')
    
    lx = xr.DataArray(lx, coords=(k,lat.pixel), dims=('k', 'pixel'))
    ly = xr.DataArray(ly, coords=lx.coords, dims=lx.dims)
    lz = xr.DataArray(lz, coords=lx.coords, dims=lx.dims)
    
    dlx = np.gradient(lx, axis=0)
    dly = np.gradient(ly, axis=0)
    dlz = np.gradient(lz, axis=0)
    dl = np.sqrt(dlx**2 + dly**2 + dlz**2) 
    return lx, ly, lz, dl #in meter

 
def k_min_max(sc_pos, lat, lon, alt, r_top=150e3):
    #calculates the min/max values for ratio k required in function los_points 
    #to cover the entire atmosphere 
    sx = sc_pos.sel(xyz='x')
    sy = sc_pos.sel(xyz='y')
    sz = sc_pos.sel(xyz='z')
    
    r_sc = np.sqrt(sx**2 + sy**2 + sz**2) #spacecraft altitude in m
    R = 6370e3 #earth's radius
    d_sc_tan = np.sqrt(r_sc**2 - (alt+R)**2) #distance between sc and tangent point
    d_top_tan = np.sqrt((r_top+R)**2 - (alt+R)**2) #distance between los intersect with TOA and tangent point
    k_min = (d_sc_tan - d_top_tan)/d_sc_tan
    k_max = (d_sc_tan + d_top_tan)/d_sc_tan
    
    return k_min, k_max

#%% 
def los_points_fix_dl(look, pos, nop=300, dl=6e3, d_start=1730e3):
    lx = np.empty((nop, len(look)))
    ly = np.empty((nop, len(look)))
    lz = np.empty((nop, len(look)))
    for i in range(nop):
        lx[i,:] = pos.sel(xyz='x') + (i+d_start/dl)*look.sel(xyz='x')*dl 
        ly[i,:] = pos.sel(xyz='y') + (i+d_start/dl)*look.sel(xyz='y')*dl
        lz[i,:] = pos.sel(xyz='z') + (i+d_start/dl)*look.sel(xyz='z')*dl
    lx = xr.DataArray(lx, coords=(np.arange(nop), look.pixel), 
                      dims=('n', 'pixel'), attrs={'units':'meter'})
    ly = xr.DataArray(ly, coords=lx.coords, dims=lx.dims, attrs=lx.attrs)
    lz = xr.DataArray(lz, coords=lx.coords, dims=lx.dims, attrs=lx.attrs)

    return lx, ly, lz

#%%convert xyz to lon lat alt for all points
def ecef2lla(lx,ly,lz):
    #x, y, z must be in xarray format
    los_lon, los_lat, los_alt = transform(ecef, lla, lx.data, ly.data, lz.data)
#    los_lon[los_lon<0] = los_lon[los_lon<0] + 360 #wrap around 180 longitude
    los_lon = xr.DataArray(los_lon, coords=lx.coords, dims=lx.dims, attrs={'units':'degree'})
    los_lat = xr.DataArray(los_lat, coords=lx.coords, dims=lx.dims, attrs={'units':'degree'})
    los_alt = xr.DataArray(los_alt, coords=lx.coords, dims=lx.dims, attrs={'units':'meter'})
    return los_lat, los_lon, los_alt

#%%
def lla2ecef(lat, lon, alt):
    #lat, lon, alt must be in xarray format
    lx, ly, lz = transform(lla, ecef, lon.data, lat.data, alt.data)
    lx = xr.DataArray(lx, coords=lat.coords, dims=lat.dims, attrs={'units':'meter'})
    ly = xr.DataArray(ly, coords=lat.coords, dims=lat.dims, attrs=lx.attrs)
    lz = xr.DataArray(lz, coords=lat.coords, dims=lat.dims, attrs=lx.attrs)
    return lx, ly, lz