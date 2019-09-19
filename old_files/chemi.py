#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 15:32:01 2019

@author: anqil
"""

import numpy as np
import matplotlib.pyplot as plt

def ozone_sme(M, T, V, jhart, js):
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
    
    o2 = 0.8 * M #assume O2 is 80% of the total air
    
    R = ko/(ko + 3.76*kn) #fraction of O(1D) that becomes O2(1 sigma)
    L = V/Ad * (Ad + kd*o2) #singlet delta O2 loss rate
    K = ks*M /(As + ks*M)
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
    o2 = 0.21*m
    ka = 6e-34*(300/T)**2.4
    kb = 8e-12 * np.exp(-2060 / T)
    o = j3 * o3 / (ka * o2 * m)# - kb * o3)
    return o
    
def jfactors(O, O2, O3, N2, z, zenithangle):
    from scipy.io import loadmat
    from geometry_functions import pathleng
    
    O = O[None,:]
    O2 = O2[None,:]
    O3 = O3[None,:]
    N2 = N2[None,:]

    path = '/home/anqil/Documents/osiris_database/ex_data/'
    sigma = loadmat(path+'sigma.mat')
    sO = sigma['sO']
    sO2 = sigma['sO2']
    sO3 = sigma['sO3']
    sN2 = sigma['sN2']
    irrad = sigma['irrad']
    wave = sigma['wave']
    pathl = pathleng(z, zenithangle) * 1e2  # [m -> cm]
    tau = np.matmul((np.matmul(sO, O) + np.matmul(sO2, O2) + np.matmul(sO3, O3) + np.matmul(sN2, N2)), pathl.T)

    jO3 = irrad * sO3 * np.exp(-tau)
    jO2 = irrad * sO2 * np.exp(-tau)
#    jO3[tau == 0] = 0
#    jO2[tau == 0] = 0

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
    path = '/home/anqil/Documents/osiris_database/ex_data/'
    alines = loadmat(path+'alines.mat')
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

    pathl = pathleng(z, zenithangle) * 1e2  # [m -> cm]
    tau = np.matmul(np.multiply(sigma.T, O2), pathl.T)
    gA = np.sum((2.742e13 * sigma.T * np.exp(-tau)), axis=0) / len(freq)
#    gA[tau[1] == 0] = 0

    return gA

def q_o2sig(n2, co2, o2, o, o3):
    k_n2 = 2.1e-15
    k_co2 = 4.2e-13
    k_o3 = 2.2e-11
    k_o = 8e-14
    k_o2 = 3.9e-17
    return n2*k_n2 + co2*k_co2 + o3*k_o3 + o*k_o + o2*k_o2

def q_o2delta(T, o2, n2, o):
    k_o2 = 3.6e-18*np.exp(-220/T)
    k_n2 = 1e-20
    k_o = 1.3e-16
    return k_o2*o2 + k_n2*n2 + k_o*o

def q_o1d(T, n2, o2):
    k_n2 = 1.8e-11*np.exp(110/T)
    k_o2 = 3.2e-11*np.exp(70/T)
    return k_n2*n2 + k_o2*o2

def ozone_mlynczak(ver_o2delta, T, m, o, o3, Jhart, Jlya, Jsrc, gA):
    o2 = 0.21*m
    n2 = 0.78*m
    co2 = 405e-6*m
    
    Q_o1d = q_o1d(T, n2, o2)
    Q_o2delta = q_o2delta(T, o2, n2, o)
    Q_o2sig = q_o2sig(n2, co2, o2, o, o3)
    
    A_o2sig = 0.0758
    A_o2delta = 2.58e-4
    A_o1d = 0 #?
    
    k = 3.2e-11*np.exp(70/T) # o1d + o2 -> o + o2sig
    
    o2delta = ver_o2delta/A_o2delta
    loss_o2delta = Q_o2delta + A_o2delta
    loss_o2sig = Q_o2sig + A_o2sig
    loss_o1d = Q_o1d + A_o1d
    
    qy_hart = 0.9 #quatumn yield
    qy_lya = 0.44
    qy_src = 1 
    eff_o1d_o2sig = 0.77 #efficiency 
    
    o2delta_from_gA = (Q_o2sig/loss_o2delta) * (gA*o2/loss_o2sig)
    o2delta_from_Jo2 = (Q_o2sig/loss_o2delta
                        ) * (eff_o1d_o2sig*k*o2/loss_o2sig
                        ) * (qy_lya*Jlya*o2 + qy_src*Jsrc*o2)/loss_o1d
    o2delta_from_Jo3_factor = (qy_hart*Jhart/loss_o2delta
                               ) + (Q_o2sig/loss_o2delta
                                    ) * (eff_o1d_o2sig*k*o2/loss_o2sig
                                    ) * (qy_hart*Jhart/loss_o1d)
    o3 = (o2delta - o2delta_from_gA - o2delta_from_Jo2)/o2delta_from_Jo3_factor
    return o3

def cal_o2delta(o3, T, m, z, zenithangle, gA):
    from chemi import jfactors, oxygen_atom
    o2 = 0.21 * m
    n2 = 0.78 * m
    co2 = 405e-6*m
    
    #gA = gfactor(o2, T, z, zenithangle)
    jhart, jsrc, jlya, j3, j2 = jfactors(np.zeros(z.shape), o2, o3, n2, z, zenithangle)
    o = oxygen_atom(m, T, o3, j3)

#    plt.semilogx(o,z, label='o')
#    plt.semilogx(o3,z, label='o3')
#    plt.legend()
    
    qy_hart = 0.9 #quatumn yield
    qy_lya = 0.44
    qy_src = 1 
    eff_o1d_o2sig = 0.77 #efficiency 
    
    from chemi import q_o1d, q_o2sig, q_o2delta
    Q_o1d = q_o1d(T, n2, o2)
    Q_o2delta = q_o2delta(T, o2, n2, o)
    Q_o2sig = q_o2sig(n2, co2, o2, o, o3)
    
    A_o2sig = 0.0758
    A_o2delta = 2.23e-4 # 2.58e-4
    A_o1d = 0 #6.81e-3 #from donal's code? 
    
    prod_o1d_from_o2 = o2 * (qy_src * jsrc + qy_lya * jlya)
    prod_o1d_from_o3 = qy_hart * o3 * jhart
    prod_o1d = prod_o1d_from_o3 + prod_o1d_from_o2
    
    loss_o1d = Q_o1d + A_o1d
    o1d = prod_o1d / loss_o1d
    
    k_o_o = 4.7e-33*(300/T)
    c_o2 = 6.6 #empirical quenchin coefficient
    c_o = 19 #empirical quenchin coefficient
    k_o1d_o2 = 3.2e-11*np.exp(70/T) #
    prod_o2sig_barth = k_o_o * o**2 * m * o2 / (c_o2*o2 + c_o*o)
    prod_o2sig = eff_o1d_o2sig * k_o1d_o2 * o2 * o1d + gA * o2 + prod_o2sig_barth
    loss_o2sig = Q_o2sig + A_o2sig
    o2sig = prod_o2sig / loss_o2sig
    
    prod_o2delta_from_o3 = qy_hart * o3 * jhart
    prod_o2delta_from_o2sig = Q_o2sig * o2sig
    prod_o2delta = prod_o2delta_from_o3 + prod_o2delta_from_o2sig
    loss_o2delta = Q_o2delta + A_o2delta
    o2delta = prod_o2delta/loss_o2delta
    
    return o2delta

def cal_o2delta_thomas(o3, T, m, z, zenithangle, gA):
    from chemi import jfactors, oxygen_atom
    o2 = 0.21 * m
    n2 = 0.78 * m
    #co2 = 405e-6*m 
    jhart, jsrc, jlya, j3, j2 = jfactors(np.zeros(z.shape), o2, o3, n2, z, zenithangle)
    #o = oxygen_atom(m, T, o3, j3)
    Q_o1d = 2e-11*np.exp(-107/T)*n2 + 2.9e-11*np.exp(-67/T)*o2
    Q_o2delta = 2.22e-18 * ((T/300)**0.78) *o2
    Q_o2sig = 2e-15*n2
    
    A_o1d = 0
    A_o2sig = 0.085
    A_o2delta = 2.58e-4
    
    qy_hart = 0.9 #quatumn yield
    qy_lya = 0.44
    qy_src = 1 
    eff_o1d_o2sig = 0.77 #efficiency
    
    prod_o1d_from_o3 = qy_hart * o3 * jhart
    prod_o1d = prod_o1d_from_o3 
    
    loss_o1d = Q_o1d + A_o1d
    o1d = prod_o1d / loss_o1d
    
    k_o1d_o2 = 3.2e-11*np.exp(70/T) #
    prod_o2sig = eff_o1d_o2sig * k_o1d_o2 * o2 * o1d + gA * o2
    loss_o2sig = Q_o2sig + A_o2sig
    o2sig = prod_o2sig / loss_o2sig
    
    prod_o2delta_from_o3 = qy_hart * o3 * jhart
    prod_o2delta_from_o2sig = Q_o2sig * o2sig
    prod_o2delta = prod_o2delta_from_o3 + prod_o2delta_from_o2sig
    loss_o2delta = Q_o2delta + A_o2delta
    o2delta = prod_o2delta/loss_o2delta
    
    return o2delta    
    
    
    