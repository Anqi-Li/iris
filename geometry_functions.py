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
    pathl = 2*pathl        
    
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
    lx = np.empty((nop, len(look.pixel)))
    ly = np.empty((nop, len(look.pixel)))
    lz = np.empty((nop, len(look.pixel)))
    
    
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

#%% Path length calculation for a given sun zenith angle (for photosynthesis rate calculation)
def pathleng(heights, Xi):
    # inputs: 
    # heights -- altitdue grid
    # Xi -- sun zenith angle
    deltaz = heights[1:] - heights[:-1]
    deltaz = np.append(deltaz,deltaz[-1])
    heights = np.append(heights, heights[-1]+deltaz[-1])

    nheights = len(heights)
    Re = 6370e3 # m Earth radius

    if Xi==90:
        	Zt = heights
    else:
        	Zt = (Re + heights) * np.sin(Xi*np.pi/180) - Re

    pathl = np.zeros((nheights, nheights))
    for j in range(nheights):
        h = heights[j:] 
        Ztj = Zt[j]
        pathl[j,j:] = np.sqrt(h**2 + 2*Re*(h-Ztj) - Ztj**2)
        
    pathl[:,:-1] = pathl[:, 1:] - pathl[:, 0:-1]
    pathl = pathl[:-1, :-1]
    pathl = np.triu(pathl)
    heights = heights[:-1]
    nheights=nheights-1

#    if Xi>90:
#        for j in range(nheights):
#            if Zt[j] > 0:
#                I = find(heights<heights[j] & heights>Zt[j])
#                
#                if ((isempty(I))):
#                    I = max(1,j-1)
#                else:
#                    I=np.append(max(I[1]-1,1),I)
#                    
#                h=heights(I)+deltaz(I)
#                Ztj=Zt(j);
#                pathl(j,I) = sqrt(h.*h+2*Re*(h-Ztj)-Ztj*Ztj)
#                                
#                if (isempty(find(I==1, 1))):
#                    pathl(j,I)=(pathl(j,I)-pathl(j,max(I-1,1)))
#                else:
#                    J=I(I~=1)
#                    pathl(j,J)=(pathl(j,J)-pathl(j,max(J-1,1)))
#                        
#                
#            else Zt(j)<=0:
#                pathl(j,:) = zeros(size(pathl(j,:)))
#                
#
#	pathl1=fliplr(pathl)
#	nanregion=find(isnan(pathl)==1)
#	pathl2=(triu((pathl'),1))';
#	pathl2(nanregion')=zeros(size(nanregion'))
#    
#	pathlength=pathl+pathl2
#	columnpathl=[pathl1 pathl2]
#    
    return pathl    

def plot_los(los_lla, sc_pos, tan_lla, edges_lla, im_lst, pix_lst):
    all_los_lat, all_los_lon, all_los_alt = los_lla
    tan_lat, tan_lon, tan_alt = tan_lla
    edges_lat, edges_lon, edges_alt = edges_lla
    
    #plot los in 3d to check
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams['legend.fontsize'] = 10
            
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    #plot los
    for im in im_lst:
        for pix in pix_lst[::50]:
            ax.plot(all_los_lon.sel(im=im,pixel=pix), 
                    all_los_lat.sel(im=im,pixel=pix),
                    all_los_alt.sel(im=im,pixel=pix))
    
    #plot satellite position
    sx, sy, sz = sc_pos.isel(date=im_lst, xyz=0), sc_pos.isel(date=im_lst, xyz=1), sc_pos.isel(date=im_lst, xyz=2)
    s_lat, s_lon, s_alt = ecef2lla(sx, sy, sz)
    ax.scatter(s_lon, s_lat, s_alt)
    ax.text(s_lon[0], s_lat[0], s_alt[0], 'sc loc')
    #plot tagent points
    tlat = tan_lat.isel(date=im_lst, pixel=pix_lst[::50])
    tlon = tan_lon.isel(date=im_lst, pixel=pix_lst[::50])
    talt = tan_alt.isel(date=im_lst, pixel=pix_lst[::50])
    ax.scatter(tlon, tlat, talt)
    ax.text(tlon[0][0], tlat[0][0], talt[0][0], 'tan point')
    
    #other settings of the plot
    ax.set_xticks(edges_lon)
    ax.set_yticks(edges_lat)
    #ax.set_zticks(edges_alt)
    #ax.xaxis.grid()
    #ax.yaxis.grid()
    #ax.zaxis.grid()
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    ax.set(xlabel='lon',
           ylabel='lat',
           zlabel='alt')
    
    ax.axis('equal')
    
    plt.show()
    

def change_of_basis(e1, e2, e3, v):
    #e1, e2, e3 are the new base vectors 
    #v is the coordinate with respect to the original base vectors
    #v_new is the coordinate with respect to the new base vectors
    #normalize all base vectors to unit length 1
    e1 = e1/np.linalg.norm(e1)
    e2 = e2/np.linalg.norm(e2)
    e3 = e3/np.linalg.norm(e3)
    
    Q = np.array([e1, e2, e3]).T
    #print('Transformed to original system with \n Q={}'.format(Q))
    
    v_new = np.linalg.solve(Q,v)
    #print('The vector in the new coordinates \n v_new={}'.format(v_new))

    if type(v) is xr.core.dataarray.DataArray:
        v_new = xr.DataArray(v_new, coords=v.coords, dims=v.dims)
    return v_new

def cart2sphe(e1, e2, e3):
    #from cartesian system to geocentric spherical corrdinate system
    #(ex. 
    #alpha: across-track angle, 
    #beta: along-track angle, 
    #rho: distance from origin)
    alpha = np.arctan(e1/e3)
    beta = np.arctan(e2/e3)
    rho = np.sqrt(e1**2 + e2**2 + e3**2) - 6371e3
    return alpha, beta, rho
