#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 11:39:03 2019

@author: anqil
"""

import sqlite3 as sql
from osirisl1services.readlevel1 import open_level1_ir
from osirisl1services.services import Level1Services
import numpy as np

db = sql.connect('OSIRIS.db')
cur = db.cursor()
insert_iri = 'insert or replace into IRI values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'
insert_ch = 'insert or replace into channel{} values (?,?,?,?,?,?,?,?,?)'
search_str = 'SELECT orbit FROM IRI WHERE orbit={}'
#insert_non = 'insert or replace into "non orbit" values (?)'

orbit = 20903 
while orbit<=20903:#26500:
    #check if this orbit exists in database
    if len(cur.execute(search_str.format(orbit)).fetchall()) == 0:
        try:
        #%% open channel 1 data and write into top table and ch1 table in database
            ir = open_level1_ir(orbit=orbit, channel=1)
            num_of_image = ir.mjd.shape[0]
            print('inserting orbit: '+ str(orbit)+', with '+str(num_of_image)+' images')
            print('channel 1')
            sc_position_ecef = ir.l1.position_ecef.data.copy()
            sc_position_eci = ir.l1.position_eci.data.copy()
            sc_look_ecef = ir.l1.look_ecef.data.copy()
            sc_look_eci = ir.l1.look_eci.data.copy()
            alt = ir.l1.altitude.data.copy()
            lat = ir.l1.latitude.data.copy()
            lon = ir.l1.longitude.data.copy()
            #sza = ir.l1.sza.data.copy()
            sza = np.ones((num_of_image, 128)) #insert temp data as sza to speed up
        
            for i in range(num_of_image): 
                cur.execute(insert_iri, (ir.stw[i].data.item(), ir.mjd[i].data.item(), 
                            ir.exposureTime[i].data.item(), ir.temperature[i].data.item(), 
                            ir.mode[i].data.item(), ir.scienceprog[i].data.item(),
                            ir.targetIndex[i].data.item(), ir.orbit.data.item(),
                            sc_position_ecef[i].data, sc_position_eci[i].data,
                            sza[i,63].item(), alt[i,63].item(), lat[i,63].item(), 
                            lon[i,63].item(), sza[i,:].data))
            
                cur.execute(insert_ch.format(1), (ir.stw[i].data.item(), 
                            ir.data[i].data.data, ir.error[i].data.data, 
                            sc_look_ecef[i].data, sc_look_eci[i].data,
                            ir.flags[i].data.data, alt[i].data, lon[i].data, 
                            lat[i].data))
            
        #%% open channel 2 and 3 data and write into ch2 and ch3 table in database
            for ch in [2,3]:
                print('channel {}'.format(ch))
                ir = open_level1_ir(orbit=orbit, channel=ch)
                sc_position_ecef = ir.l1.position_ecef.data.copy()
                sc_position_eci = ir.l1.position_eci.data.copy()
                sc_look_ecef = ir.l1.look_ecef.data.copy()
                sc_look_eci = ir.l1.look_eci.data.copy()
                alt = ir.l1.altitude.data.copy()
                lat = ir.l1.latitude.data.copy()
                lon = ir.l1.longitude.data.copy()
                    
                for i in range(num_of_image):               
                    cur.execute(insert_ch.format(ch), (ir.stw[i].data.item(), ir.data[i].data.data,
                                ir.error[i].data.data, sc_look_ecef[i].data, sc_look_eci[i].data,
                                ir.flags[i].data.data, alt[i].data, lon[i].data, lat[i].data))
                    
            db.commit()
        except LookupError:
            print('orbit file {} not found'.format(orbit))
#            cur.execute(insert_non, orbit)
            pass
        
    #this orbit already exists so we will do nothing
    else:
        print('orbit '+str(orbit)+' already exists in database')
    
    #next orbit
    orbit += 1

db.close()    
    



































