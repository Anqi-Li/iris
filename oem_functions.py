#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 16:50:44 2019

@author: anqil
"""

import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import inv, spsolve

#%% oem for sparse matrix
def linear_oem_sp(K, Se, Sa, y, xa):
    K = K.tocsc()
    Sa = Sa.tocsc()
    Se = Se.tocsc()
    
    if len(y)<len(xa): # m form
        G = Sa.dot(K.T).dot(inv(K.dot(Sa).dot(K.T) + Se))

    else:
        Se_inv = inv(Se)
        Sa_inv = inv(Sa)
        G = inv(K.T.dot(Se_inv).dot(K) + Sa_inv).dot(K.T).dot(Se_inv)
#        G = spsolve(K.T.dot(Se_inv).dot(K) + Sa_inv, (K.T).dot(Se_inv))
    x_hat = xa + G.dot(y - K.dot(xa))
#    A = G.dot(K)
#    I = sp.identity(len(xa))
#    Ss = (A - I).dot(Sa).dot((A - I).T) # smoothing error
#    Sm = G.dot(Se).dot(G.T) #retrieval noise 
    return x_hat, G#, A, Ss, Sm

#%% oem for dense matrix
def linear_oem(K, Se, Sa, y, xa):
    if len(y.shape) == 1:
        y = y.reshape(len(y),1)
    if len(xa.shape) == 1:
        xa = xa.reshape(len(xa),1)
        
    
    if len(y)<len(xa): # m form
        G = Sa.dot(K.T).dot(np.linalg.inv(K.dot(Sa).dot(K.T) + Se))
        
        
    else: # n form
        Se_inv = np.linalg.inv(Se)
        Sa_inv = np.linalg.inv(Sa)
        G = np.linalg.inv(K.T.dot(Se_inv).dot(K) + Sa_inv).dot(K.T).dot(Se_inv)
#        G= np.linalg.solve(K.T.dot(Se_inv).dot(K) + Sa_inv, (K.T).dot(Se_inv))
        
    x_hat = xa + G.dot(y - K.dot(xa)) 
    A = G.dot(K)
    Ss = (A - np.identity(len(xa))).dot(Sa).dot((A - np.identity(len(xa))).T) # smoothing error
    Sm = G.dot(Se).dot(G.T) #retrieval noise 
    return x_hat.squeeze(), A, Ss, Sm

#%% generate rows in jacobian for each measurement (pixel)
import pandas as pd
def jacobian_row(dll, lla_edges, los, pix_idx):
    #compute pathlength along 1 los (1 pixel) 
    #dll: distance between each points along los 
    #los: lla coordinate of all points along los
    #lla_edges: atmospheric grid that you want to define in lla coordinate
    #(dll.shape = los.shape)
    #(be aware that dll unit will be the same as the pathlength unit)
        
    edges_lat, edges_lon, edges_alt = lla_edges
    los_lat, los_lon, los_alt = los
    
    #index of lat/lon/alt bins which the point belongs to 
    # edges[i-1] < sample <= edges[i] 
    # (returns idx = 0 or len(xxedges) if the points fall outside the bin range)
    bin_idx_lat = np.searchsorted(edges_lat, los_lat)
    bin_idx_lon = np.searchsorted(edges_lon, los_lon)
    bin_idx_alt = np.searchsorted(edges_alt, los_alt)
    
    flat_idx_each_points = np.ravel_multi_index((bin_idx_lat, bin_idx_lon, bin_idx_alt),
                                    (len(edges_lat)+1, len(edges_lon)+1, len(edges_alt)+1))
    
    #grid_idx, f = np.unique(flat_idx, return_inverse=True) 
    f, grid_idx = pd.factorize(flat_idx_each_points)
    pathlength = np.bincount(f, weights=dll).astype(dll.dtype)
    measurement_idx = pix_idx * np.ones(len(grid_idx))
    
    return measurement_idx, grid_idx, pathlength
                
#%% unravel measurement id
def unfold_measure_id(measurement_id, im_lst, pix_lst):
    dims=(len(im_lst), len(pix_lst))
    im_idx, pix_idx = np.unravel_index(measurement_id, dims)
    image = im_lst[im_idx]
    pix = pix_lst[pix_idx]
    return image, pix

#%% unravel grid id
def unfold_grid_id(grid_id, edges_lat, edges_lon, edges_alt):
    dims = (len(edges_lat)+1, len(edges_lon)+1, len(edges_alt)+1)
    bin_idx_lat, bin_idx_lon, bin_idx_alt = np.unravel_index(grid_id, dims)
    max_lat = edges_lat[bin_idx_lat]
    max_lon = edges_lon[bin_idx_lon]
    max_alt = edges_alt[bin_idx_alt]
    return max_lat, max_lon, max_alt

#%% oem handels sparse matrix
    