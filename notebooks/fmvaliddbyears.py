#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 14:00:36 2017

@author: donal
"""

import requests as R
import numpy as np
import json
import sqlite3 as sql
import matplotlib.pylab as plt

def load_scans(dataset,fm,start_date,end_date):
    baseurl="http://odin.rss.chalmers.se/rest_api/v5/level2/development/"
    scansurl=baseurl+"{0}/{1}/scans/?limit=1000&offset=0&"
    scansurl+="start_time={2}&end_time={3}"
    a=R.get(scansurl.format(dataset,fm,start_date,end_date))
    return json.loads(a.text)


def load_data(dataset,fm,scanno,product,cur):
    selstr='select jsondata from odinL2 where dataset=="{0}" and fm=="{1}" and scanno=={2} and product =="{3}"'
    result=cur.execute(selstr.format(dataset,fm,scanno,product)).fetchone()
    if result==None:
        baseurl="http://odin.rss.chalmers.se/rest_api/v5/level2/development/"
        scansurl=baseurl+"{0}/{1}/{2}/L2/?product={3}"
        print ".",
        a=R.get(scansurl.format(dataset,fm,scanno,product))
        result=a.text
        instr="insert or replace into odinL2 values (?,?,?,?,?)"
        cur.execute(instr,(dataset,fm,scanno,product,result))
    else: 
        result=result[0] 
        print "-",
    return json.loads(result)

def load_L2i(dataset,fm,scanno,cur):
    selstr='select jsondata from odinL2i where dataset=="{0}" and fm=="{1}" and scanno=={2} '
    result=cur.execute(selstr.format(dataset,fm,scanno)).fetchone()
    if result==None:
        baseurl="http://odin.rss.chalmers.se/rest_api/v5/level2/development/"
        scansurl=baseurl+"{0}/{1}/{2}/L2i/"
        print ".",
        a=R.get(scansurl.format(dataset,fm,scanno))
        result=a.text
        instr="insert or replace into odinL2i values (?,?,?,?)"
        cur.execute(instr,(dataset,fm,scanno,result))
    else: 
        result=result[0] 
        print "*",
    return json.loads(result)

def load_colo(fm,scanno,cur):
    selstr='select jsondata from colocs where fm=="{0}" and scanno=={1} '
    result=cur.execute(selstr.format(fm,scanno)).fetchone()
    if result==None:
        baseurl="http://odin.rss.chalmers.se/rest_api/v5/level1/{0}/"
        scansurl=baseurl+"{1}/collocations/"
        print ".",
        a=R.get(scansurl.format(fm,scanno))
        result=a.text
        instr="insert or replace into colocs values (?,?,?)"
        cur.execute(instr,(fm,scanno,result))
    else: 
        result=result[0] 
        print "c",
    return json.loads(result)

def load_colo_data(inst, scanno, species, co_url ):
    selstr='select jsondata from colocsdata where instrument=="{0}" and scanno=={1} and species=="{2}"'
    result=cur.execute(selstr.format(inst,scanno,species)).fetchone()
    if result==None:
        print ",",
        a=R.get(co_url.replace("/v5/","/v4/"))
        result=a.text
        instr="insert or replace into colocsdata values (?,?,?,?)"
        cur.execute(instr,(inst,scanno,species,result))
    else: 
        result=result[0] 
        print "d",
    return json.loads(result)

instlab=['mls','osiris','mipas','smr v2','sageIII']



fm="2"
#species="O3"
#species="H2O"
#species="T"
species="HNO3"
#species="ClO"
#species="N2O"
#VDS="StndVDS10"
VDS="DDS"
#VDS="StndVDS9"
#VDS="Stnd1vds6"
#VDS="MESOVDS4"
if fm=="1" : 
    if species =="O3" :product="O3 / 501 GHz / 20 to 50 km"
    if species =="ClO" :product="ClO / 501 GHz / 20 to 55 km"
    if species =="N2O" :product="N2O / 502 GHz / 15 to 50 km"
if fm=="2" : 
    if species =="O3" :product="O3 / 545 GHz / 20 to 85 km"
    if species =="HNO3" :product="HNO3 / 545 GHz / 20 to 50 km"
if fm=="8" : 
    if species =="O3" :product="O3 / 488 GHz / 20 to 60 km"
    elif species=="H2O" :product="H2O / 488 GHz / 20 to 70 km"
if fm=="17" : 
    if species =="O3" :product="O3 / 488 GHz / 20 to 60 km"
    elif species=="H2O" :product="H2O / 488 GHz / 20 to 70 km"
if fm=="13": product="H2O - 557 GHz - 45 to 100 km"
#if fm=="13": product="Temperature"
if product=="Temperature" :
    Target="Temperature"
else : Target="VMR"
print (("VDS= {} product = {} Target = {}").format(VDS, product,Target))
    
plt.close('all')
perr=[]
Tsat=[]
scannos=[]
for i in range(1,6):
    plt.figure(i) 
for year in range(2003,2009):
    print "\nworking year : ",year
    data=load_scans(VDS,fm,"{}-01-01".format(str(year)),"{}-04-01".format(str(year)))
    scans=[data['Data'][i]['ScanID'] for i in range(len(data['Data']))]
    [scannos.append(a) for a in scans]
    db=sql.connect('faster.db')
    
    cur=db.cursor()
    #insertstr='insert or replace into Newsensors values (?,?,?,?,?,?,?,?)'
    
    #newdata=[]
    #olddata=[]
    #for scanno in scans:
    #    newdata.append(load_data("StndVDS8","1", scanno,product))
    #    olddata.append(load_data("Stnd1vds6","1", scanno,product))
    #avg=np.zeros(np.array(newdata[i]['Data'][0][Target]).shape)
    #for i in range(len(olddata)):
    #    avg+=np.array(newdata[i]['Data'][0][Target])-np.array(olddata[i]['Data'][0][Target])
     
    #Collocations 
    #Define standard height grid 
    heights=np.arange(15,91,1)
    mlsavg=[];osirisavg=[];mipasavg=[];smravg=[];sageavg=[]
    mlsper=[];osirisper=[];mipasper=[];smrper=[];sageper=[]
    res=[]
#    perr=[]
    for scanno in scans:
        odindata=load_data(VDS,fm,scanno,product,cur)['Data'][0]
        for i,mr in enumerate (odindata['MeasResponse']):
            if mr>=0.8: odindata[Target][i]*=1.0 
            else: odindata[Target][i]*=np.nan
        db.commit()
        coloc=load_colo(fm,scanno,cur)
        l2i=load_L2i(VDS,fm,scanno,cur)['Data']
        res.append(l2i['Residual'])
        perr.append(l2i['PointOffset'])
        Tsat.append(l2i['Tsat'])
        co_inst=[e['Instrument'] for e in coloc['Data'] if e['Species'] == species] 
        co_url=[e['URL'] for e in coloc['Data'] if e['Species'] == species] 
        for n,inst in enumerate(co_inst):
            if inst=="mls":
                mlsdata=load_colo_data(inst, scanno, species,co_url[n])
                #interpolate mls to Odin pressure grid then use Odin Z to interpolate to stnd
                compar=np.interp(np.log(odindata['Pressure'][::-1]),
                                 np.log(100*np.array(mlsdata['geolocation_fields']['Pressure'][::-1])),
                                 np.array(mlsdata['data_fields'][species])[::-1])
                compar=np.interp(heights,np.array(odindata['Altitude'])/1000.,compar[::-1])
                odin=np.interp(heights,np.array(odindata['Altitude'])/1000.,np.array(odindata[Target]),left=np.nan,right=np.nan)
                mlsavg.append(odin-compar)
                mlsper.append((odin-compar)/(odin+compar)*2)
            elif inst=="osiris" :
                osirisdata=load_colo_data(inst, scanno, species,co_url[n])
                osiriso3=osirisdata['data_fields']['O3']
                osiriso3=[nn if nn <> -9999 else np.nan for nn in osiriso3]
                #interpolate osiris and  Odin  Z to interpolate to stnd
                compar=np.interp(heights,np.array(osirisdata['geolocation_fields']['Altitude']),
                                 osiriso3)
                odin=np.interp(heights,np.array(odindata['Altitude'])/1000.,np.array(odindata[Target]),left=np.nan,right=np.nan)
                osirisavg.append(odin-compar)
                osirisper.append((odin-compar)/(odin+compar)*2)
            elif inst=="sageIII" :
                sagedata=load_colo_data(inst, scanno, species,co_url[n]) 
                N=np.array(sagedata['Pressure'])*100/1.38066e-23/np.array(sagedata['Temperature'])
                sageo3=[sagedata[species][i][0]/N[i]*1e6 for i in range(len(sagedata['Pressure']))]
                #interpolate osiris and  Odin  Z to interpolate to stnd
                compar=np.interp(np.log(odindata['Pressure'][::-1]),
                                 np.log(100*np.array(sagedata['Pressure'][::-1])),
                                 sageo3[::-1])
                compar=np.interp(heights,np.array(odindata['Altitude'])/1000.,compar[::-1])
                odin=np.interp(heights,np.array(odindata['Altitude'])/1000.,np.array(odindata[Target]),left=np.nan,right=np.nan)
                sageavg.append(odin-compar)
                sageper.append((odin-compar)/(odin+compar)*2)
            elif inst=="mipas" :
                mipasdata=load_colo_data(inst, scanno, species,co_url[n])
                mipaso3=np.array(mipasdata['target'])/1e6
                mipaso3=[nn if nn <> -9999 else np.nan for nn in mipaso3]
                compar=np.interp(heights,np.array(mipasdata['altitude']),
                                 mipaso3)
                odin=np.interp(heights,np.array(odindata['Altitude'])/1000.,np.array(odindata[Target]),left=np.nan,right=np.nan)
                mipasavg.append(odin-compar)
                mipasper.append((odin-compar)/(odin+compar)*2)
            elif inst=="smr" :
                smrdata=load_colo_data(inst, scanno, species,co_url[n])['Data']
                compar=np.interp(heights,np.array(smrdata['Altitudes']),
                                 smrdata['Profiles'])
                odin=np.interp(heights,np.array(odindata['Altitude'])/1000.,np.array(odindata[Target]),left=np.nan,right=np.nan)
                smravg.append(odin-compar)
                smrper.append((odin-compar)/(odin+compar)*2)
                
            else: print ('missing inst {} for scan {}'.format(inst, scanno))
    db.commit()
    if mlsavg: 
#        plt.figure(1)
#        plt.plot(np.array(mlsavg).mean(axis=0)*1e6,heights,label=str(year))
#    if osirisavg:
#        plt.figure(2)
#        plt.plot(np.array(osirisavg).mean(axis=0)*1e6,heights,label=str(year))    
#    if mipasavg:
#        plt.figure(3)
#        plt.plot(np.array(mipasavg).mean(axis=0)*1e6,heights,label=str(year))    
#    if smravg:
#        plt.figure(4)
#        plt.plot(np.array(smravg).mean(axis=0)*1e6,heights,label=str(year)) 
#    if sageavg:
#        plt.figure(5)
#        plt.plot(np.array(sageavg).mean(axis=0)*100,heights,label=str(year)) 
        plt.figure(1)
        plt.plot(np.array(mlsper).mean(axis=0)*100,heights,label=str(year))
    if osirisavg:
        plt.figure(2)
        plt.plot(np.array(osirisper).mean(axis=0)*100,heights,label=str(year))    
    if mipasavg:
        plt.figure(3)
        plt.plot(np.array(mipasper).mean(axis=0)*100,heights,label=str(year))    
    if smravg:
        plt.figure(4)
        plt.plot(np.array(smrper).mean(axis=0)*100,heights,label=str(year))
    if sageavg:
        plt.figure(5)
        plt.plot(np.array(sageper).mean(axis=0)*100,heights,label=str(year)) 
cur.close()
db.commit()
db.close  
for i in range(5):
    plt.figure(i+1)
    plt.xlim([-50,50])
    plt.legend()
    plt.xlabel('Difference (%)')
    plt.ylabel('Altitude (km)')
    plt.title('FM{} comparision with {}'.format(fm,instlab[i]))
    plt.savefig("{}_fm{}:{}_perdiff{}.png".format(VDS,fm,species,instlab[i]),transparent=True)