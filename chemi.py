#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 15:32:01 2019

@author: anqil
"""

import numpy as np
import matplotlib.pyplot as plt

def ozone_sme(m, T, V, jhart=8e-3, js=6.3e-9):
    #kinetic model presented in Thomas et al (1983)
    #inputs:
    #m: background air density
    #T: temperature in K
    #V: volumn emission 
    #jhart: the number of ozone dissociations (hartly band) per second per molecule
    #js: excitation rate of o2(1 sigma) -- per second per molecule
    epsilon = 0.9 #production efficiency of o2(1 delta) from o3
    As = 0.085 # sec-1 Einstein coefficients for o2(1 sigma)
    Ad = 2.58e-4 # sec-1 Einstein coefficients for o2(1 delta)
    kd = 2.22e-18 * (T/300)**0.78 #cm3 sec-1
    kn = 2e-11 * np.exp(-107/T) #cm3 sec-1
    ko = 2.9e-11 * np.exp(-67/T) #cm3 sec-1
    ks = 2e-15 #cm3 sec-1
    
    o2 = 0.8 * m #assume O2 is 80% of the total air
    
    R = ko/(ko + 3.76*kn) #fraction of O(1D) that becomes O2(1 sigma)
    L = V/Ad * (Ad + kd*o2) #singlet delta O2 loss rate
    K = ks*m /(As + ks*m)
    o3 = (L - js * o2 * K)/(jhart * epsilon * R * K + jhart * epsilon)
    return o3

def ozone_textbook(m, T, V, jhart=8e-3):
    #kinetic model presented in textbook (aeronomy of the middle atmosphere) p203
    epsilon = 0.9 #production efficiency of o2(1 delta) from o3
    Ad = 2.58e-4 # sec-1 Einstein coefficients for o2(1 delta)   
    kd = 2.22e-18*(T/300)**0.78 #cm3 sec-1
    o2 = 0.8*m
    
    F = epsilon*jhart*(Ad + kd*o2)
    o3 = V/Ad/F
    return o3

def oxygen_atom(m, T, o3, j3):
    #only works for day, not night
    #(smith et al. 2011)
    o2 = 0.2*m
    ka = 6e-34*(300/T)**2.4
    kb = 8e-12 * np.exp(-2060 / T)
    o = j3 * o3 / (ka * o2 * m - kb * o3)
    return o
    
def jfactors(O, O2, O3, N2, z, zenithangle):
    from scipy.io import loadmat
    from geometry_functions import pathleng
    
    O = O[None,:]
    O2 = O2[None,:]
    O3 = O3[None,:]
    N2 = N2[None,:]

    sigma = loadmat('sigma.mat')
    sO = sigma['sO']
    sO2 = sigma['sO2']
    sO3 = sigma['sO3']
    sN2 = sigma['sN2']
    irrad = sigma['irrad']
    wave = sigma['wave']
    pathl = pathleng(z, zenithangle) * 1e3  # [m -> cm]
    tau = np.matmul((np.matmul(sO, O) + np.matmul(sO2, O2) + np.matmul(sO3, O3) + np.matmul(sN2, N2)), pathl.T)

    jO3 = irrad * sO3 * np.exp(-tau)
    jO2 = irrad * sO2 * np.exp(-tau)
    jO3[tau == 0] = 0
    jO2[tau == 0] = 0

    hartrange = (wave > 210) & (wave < 310)
    srcrange = (wave > 122) & (wave < 175)
    lyarange = 28  # wavelength = 121.567 nm
    jhart = np.matmul(hartrange, jO3)
    jsrc = np.matmul(srcrange, jO2)
    jlya = jO2[lyarange][:]

    j3 = np.sum(jO3, axis=0)
    j2 = np.sum(jO2, axis=0)

    jhart = np.squeeze(jhart)
    jsrc = np.squeeze(jsrc)

    return jhart, jsrc, jlya, j3, j2


def gfactor(O2, T, z, zenithangle):
    from scipy.io import loadmat
    from geometry_functions import pathleng
    import sys
    sys.path.append('..')
    
    O2 = O2[None,:]

    alines = loadmat('alines.mat')
    freq = alines['freq']
    Sj = alines['Sj']
    Elow = alines['Elow']
    # Al = alines['Al']
    K = 1.3807e-23  # Boltzmann constant [m2kgs-2K-1]
    C = 299792458  # speed of light [m/s]
    AMU = 1.66e-27  # atomic mass unit [kg]

    Ad = freq / C * np.sqrt(2 * np.log(2) * K * 298 / 32 / AMU)
    grid = np.arange(12900, 13170, 0.01)  # frequency interval
    sigma = np.zeros((len(z), len(grid)))

    def doppler(Ad, niu_niu0):
        return np.sqrt(np.log(2) / np.pi) / Ad * np.exp(-np.log(2) * niu_niu0 ** 2 / Ad ** 2)

    for zi in range(len(z)):
        Sjlayer = Sj * 298 / T[zi] * np.exp(1.439 * Elow * (T[zi] - 298) / 298 / T[zi])
        Adlayer = Ad * np.sqrt(T[zi] / 298)

        for freqi in range(len(freq)):
            sigma[zi] += Sjlayer[freqi] * doppler(Adlayer[freqi], grid - freq[freqi])

    pathl = pathleng(z, zenithangle) * 1e5  # [km -> cm]
    tau = np.matmul(np.multiply(sigma.T, O2), pathl.T)
    gA = np.sum((2.742e13 * sigma.T * np.exp(-tau)), axis=0) / len(freq)
#    gA[tau[1] == 0] = 0

    return gA
    