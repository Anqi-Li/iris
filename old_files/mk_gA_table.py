#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 17:30:01 2019

@author: anqil
"""
import numpy as np
import matplotlib.pylab as plt
from chemi import gfactor
from scipy.io import loadmat
MSIS = loadmat('msisdata.mat')
zMsis = MSIS['zMsis'].squeeze() *1e3 # in m
TMsis = MSIS['TMsis'] # in K
NMsis = MSIS['NMsis'] # in cm-3 
monthMsis = MSIS['monthMsis'].squeeze() + 1
latMsis = MSIS['latMsis'].squeeze()

#%%
#idx_mon = 9
#idx_lat = 9
#sza = 90
#
#gA = gfactor(0.21*NMsis[:, idx_mon, idx_lat], 
#             TMsis[:, idx_mon, idx_lat], 
#             zMsis, sza)
#
#gA_test = np.insert(gA_test, -1, gA, axis=1)
#plt.plot(gA_test, zMsis)
#

#%%

latMsis = latMsis[np.array([9, 14])]

sza = np.arange(0, 92, 2)
gA = np.zeros((len(zMsis), len(monthMsis)-1, len(latMsis), len(sza)))
for idx_mon in range(len(monthMsis)-1):
    for idx_lat in range(len(latMsis)):
        for idx_sza in range(len(sza)):
            print('month: ', monthMsis[idx_mon])
            print('latitude: ', latMsis[idx_lat])
            print('sza: ', sza[idx_sza])
            gA[:, idx_mon, idx_lat, idx_sza] = gfactor(
                    0.21*NMsis[:, idx_mon, idx_lat], 
                         TMsis[:, idx_mon, idx_lat], 
                         zMsis, sza[idx_sza])
    
#np.savez('gA_table', gA=gA, month=monthMsis[:-1], z=zMsis, lat=latMsis, sza=sza)

#%%
sza = 90
ggA = np.zeros((len(zMsis), len(monthMsis)-1, len(latMsis)))
for idx_mon in range(len(monthMsis)-1):
    for idex_lat in range(len(latMsis)):
        print('month: ', monthMsis[idx_mon])
        print('latitude: ', latMsis[idx_lat])
        ggA[:, idx_mon, idx_lat] = gfactor(0.21*NMsis[:, idx_mon, idx_lat],
           TMsis[:, idx_mon, idx_lat], zMsis, sza)

gA = np.insert(gA, -1, ggA, axis=3)
#np.savez('gA_table', gA=gA, month=monthMsis[:-1], z=zMsis, lat=latMsis, sza=np.append(np.arange(0,90,2), sza))
