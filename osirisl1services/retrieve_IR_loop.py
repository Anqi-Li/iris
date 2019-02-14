#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 17:17:03 2019

@author: anqil
"""

def osiris_save_to_nc(orbit, scan, channel, folder):
    scanno = int(('%04d' % orbit) + ('%03d' % scan))
    ir = open_level1_ir(scanno=scanno, channel=channel)
    units = 'days since 1858-11-17 00:00:00.0'
    y = num2date(ir.mjd[0], units).year
    ncfilename = folder +str(y)+'/'+str(scanno) + '.nc'

    ir.data.to_netcdf(ncfilename)
    ir.error.to_netcdf(ncfilename, mode='a')
    ir.flags.to_netcdf(ncfilename, mode='a')

    ir.stw.to_netcdf(ncfilename, mode='a')
    ir.exposureTime.to_netcdf(ncfilename, mode='a')
    ir.temperature.to_netcdf(ncfilename, mode='a')
    ir.tempavg.to_netcdf(ncfilename, mode='a')
    ir.mode.to_netcdf(ncfilename, mode='a')
    ir.scienceprog.to_netcdf(ncfilename, mode='a')
    ir.shutter.to_netcdf(ncfilename, mode='a')
    ir.lamp1.to_netcdf(ncfilename, mode='a')
    ir.lamp2.to_netcdf(ncfilename, mode='a')
    ir.targetIndex.to_netcdf(ncfilename, mode='a')
    ir.exceptions.to_netcdf(ncfilename, mode='a')
    ir.processingflags.to_netcdf(ncfilename, mode='a')

    ir.channel.to_netcdf(ncfilename, mode='a')
    
    print('alt-lat-lon')
#    print('altitude')
    ir.l1.altitude.rename('altitude').to_netcdf(ncfilename, mode='a')
#    print('longitude')
    ir.l1.longitude.rename('longitude').to_netcdf(ncfilename, mode='a')
#    print('latitude')
    ir.l1.latitude.rename('latitude').to_netcdf(ncfilename, mode='a')
#    print('sza')
#    ir.l1.sza.rename('sza').to_netcdf(ncfilename, mode='a')
    
    print('orbit:'+str(orbit)+' scan:'+str(scan))
    print('year:'+str(num2date(ir.mjd[0], units)))
    return
    
    
    
from osirisl1services.readlevel1 import open_level1_ir
from osirisl1services.services import Level1Services
from netCDF4 import num2date

channel = 2
orb_start = 18514
orb_end = orb_start + 1
orb = range(orb_start, orb_end)
sc = range(70)
#folder = '/home/anqil/Documents/OSIRIS_IR/IR_ch'+str(channel)+'/'
folder = './'
print('From orbit '+str(orb_start)+' to '+str(orb_end))

for orbit in orb:
    for scan in sc:
        try:
            osiris_save_to_nc(orbit, scan, channel, folder)
        except:
            pass














