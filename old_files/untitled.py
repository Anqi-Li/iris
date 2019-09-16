import sqlite3 as sql
import sys
sys.path.append('..')
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
from netCDF4 import num2date
units = 'days since 1858-11-17 00:00:00.000'
import astropy.coordinates as coord
from astropy.time import Time
import astropy.units as u
from oem_functions import linear_oem
from geometry_functions import pathl1d_iris
plt.ioff()


def extract_iris_db(orbit, ch, figure_dir):
    try:
        db = sql.connect(db_file)
        cur = db.cursor()
        return_column = ('data, mjd, look_ecef, sc_position_ecef, latitude, longitude, altitude')              
        select_str = 'SELECT {} FROM IRI JOIN channel{} ON IRI.stw = channel{}.stw WHERE orbit={}'
        result = cur.execute(select_str.format(return_column, ch, ch, orbit))
        all_image = result.fetchall()
        if len(all_image) == 0:
            print('No data for orbit {}'.format(orbit))
        print(orbit,' : num of images: {}'.format(len(all_image)))
        db.close()
        
        
        l1_blob = np.array(all_image)[:,0]
        mjd = np.array(all_image)[:,1].astype(float)
        date = num2date(mjd, units)
        sc_look_blob = np.array(all_image)[:,2]
        sc_pos_blob = np.array(all_image)[:,3]
        tan_lat_blob = np.array(all_image)[:,4]
        tan_lon_blob = np.array(all_image)[:,5]
        tan_alt_blob = np.array(all_image)[:,6]
        
        #====unfolding blobs
        l1 = np.empty((len(all_image),128))
        sc_look = np.empty((len(all_image), 128, 3))
        sc_pos = np.empty((len(all_image), 3))
        tan_lat = np.empty((len(all_image),128))
        tan_lon = np.empty((len(all_image),128))
        tan_alt = np.empty((len(all_image),128))
        for i in range(len(all_image)):
            l1[i,:] = np.frombuffer(l1_blob[i])
            sc_look[i,:,:] = np.frombuffer(sc_look_blob[i]).reshape(128,3)
            sc_pos[i,:] = np.frombuffer(sc_pos_blob[i])
            tan_lat[i,:] = np.frombuffer(tan_lat_blob[i])
            tan_lon[i,:] = np.frombuffer(tan_lon_blob[i])
            tan_alt[i,:] = np.frombuffer(tan_alt_blob[i])
        #====construct xarray data array
        pixel = np.arange(128)
        l1 = xr.DataArray(l1, coords=(mjd, pixel), 
                          dims=('mjd', 'pixel'), 
                          attrs={'units':'Rayleigh??'})
        sc_look = xr.DataArray(sc_look, coords=(mjd, pixel, ['x', 'y', 'z']), 
                               dims=('mjd', 'pixel', 'xyz'))
        sc_pos = xr.DataArray(sc_pos, coords=(mjd, ['x', 'y', 'z']), 
                              dims=('mjd', 'xyz'))
        tan_lat = xr.DataArray(tan_lat, coords=(mjd, pixel),
                               dims=('mjd', 'pixel'), 
                               attrs={'units':'degree'})
        tan_lon = xr.DataArray(tan_lon, coords=(mjd, pixel),
                               dims=('mjd', 'pixel'), 
                               attrs={'units':'degree'})
        tan_alt = xr.DataArray(tan_alt, coords=(mjd, pixel),
                               dims=('mjd', 'pixel'), 
                               attrs={'units':'meter'})
    #    import pandas as pd
    #    from geometry_functions import lla2ecef
    #    tan_ecef = xr.concat(lla2ecef(tan_lat,tan_lon,tan_alt), 
    #                         pd.Index(['x','y','z'], name='xyz'))
    #    
        #====drop all dates which have nan in l1
        l1 = l1.dropna('mjd')
        sc_look = sc_look.sel(mjd=l1.dropna('mjd').mjd)
        sc_pos = sc_pos.sel(mjd=l1.dropna('mjd').mjd)
        tan_lat = tan_lat.sel(mjd=l1.dropna('mjd').mjd)
        tan_lon = tan_lon.sel(mjd=l1.dropna('mjd').mjd)
        tan_alt = tan_alt.sel(mjd=l1.dropna('mjd').mjd)
        mjd = l1.dropna('mjd').mjd.data
        print('num of images after removing nan: {}'.format(len(date)))
            
        
        #==== plot limb radiance
        alts_interp = np.arange(40e3, 120e3, .25e3)
        data_interp = []
        
        for (data, alt) in zip(l1, tan_alt):
            f = interp1d(alt, data, bounds_error=False)
            data_interp.append(f(alts_interp))
        data_interp = xr.DataArray(data_interp, 
                                   coords=(mjd, alts_interp), 
                                   dims=('mjd', 'alt'))
        data_interp.attrs['units'] = 'Rayleigh?'
        data_interp.attrs['long_name'] = 'interpolated data'
        
        #==== plot the full orbit
        fig = plt.figure()
        data_interp.plot(x='mjd', y='alt', 
                         norm=LogNorm(), 
                         vmin=1e9, vmax=1e13, 
                         size=5, aspect=3)
        ax = plt.gca()
        ax.set(title='From {} \n to {}, \n channel {}'.format(num2date(mjd[0],units),
               num2date(mjd[-1], units), ch),
              xlabel='image index')
    
        ax.set_xticks(mjd[np.arange(0,len(mjd),300, dtype=int)])
        ax.set_xticklabels(np.arange(0,len(mjd),300))
        plt.savefig(figure_dir+'limb_{}.png'.format(orbit), bbox_inches = "tight") #fig file name
        plt.close(fig)
        
    except:
        pass
    
        return


figure_dir = '/home/anqil/Documents/osiris_database/plots/2005/'
db_file = '/home/anqil/Documents/osiris_database/OSIRIS_2005.db'
db = sql.connect(db_file)
cur = db.cursor()
select_str = 'SELECT orbit from IRI'
result = cur.execute(select_str)
all_image = result.fetchall()
db.close()
orb_lst = np.unique(all_image)
orb_from = 26168
idx_from = (np.abs(orb_lst - orb_from)).argmin()
for orbit in orb_lst[idx_from:]:
    extract_iris_db(orbit, 3, figure_dir)        
