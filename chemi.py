#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 15:32:01 2019

@author: anqil
"""

import numpy as np

def ozone_sme(m, T, V, J3=8e-3, Js=6.3e-9):
    #kinetic model presented in Thomas et al (1983)
    #inputs:
    #m: background air density
    #T: temperature in K
    #V: volumn emission 
    #J3: the number of ozone dissociations per second per molecule
    #Js: excitation rate of o2(1 sigma) -- per second per molecule
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
    o3 = (L - Js * o2 * K)/(J3 * epsilon * R * K + J3 * epsilon)
    
    return o3

def ozone_textbook(m, T, V, J3=8e-3):
    #kinetic model presented in textbook (aeronomy of the middle atmosphere) p203
    epsilon = 0.9 #production efficiency of o2(1 delta) from o3
    Ad = 2.58e-4 # sec-1 Einstein coefficients for o2(1 delta)   
    kd = 2.22e-18*(T/300)**0.78 #cm3 sec-1
    o2 = 0.8*m
    
    F = epsilon*J3*(Ad + kd*o2)
    o3 = V/Ad/F
    return o3


    