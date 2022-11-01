# Plot SST

import numpy as np
import xarray as xr
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import datetime
import time
import os
import sys
import shapely.geometry as sgeom
import glob

# RV Roger Revelle obs file
nc_file = f"/glade/scratch/harrold/uni_phys/obs/rv_rog_rev_met/surface.ship.revelle.cruise2.nc"
nc = Dataset(nc_file)

# Read in vars from nc file
time = nc.variables['time'][:]
obs_lat = nc.variables['lat'][:]
obs_lon = nc.variables['lon'][:]
obs_sst = nc.variables['sst'][:]

dtimes = [datetime.datetime.fromtimestamp(x) for x in time]
dtimes_format = ([x.strftime('%Y%m%d%H%M%S') for x in dtimes])
dtimes_sub = np.where( np.logical_and( np.array(dtimes) > datetime.datetime(2011, 10, 15, 00, 00, 00), np.array(dtimes) < datetime.datetime(2011, 10,20, 00, 00, 00) ))

dtimes_sub = [x for x in dtimes if x.strftime('%M') == "00"]
start_time = datetime.datetime(2011, 10, 15, 00, 00, 00)
n_fcst = 121
model_dtimes = [start_time + datetime.timedelta(seconds = 3600*x) for x in range(n_fcst)]

obs_sst_subset = [obs_sst[np.where(np.array(dtimes) == x)].data[0] + 273.15 for x in model_dtimes]
obs_lat_subset = [obs_lat[np.where(np.array(dtimes) == x)].data[0] for x in model_dtimes]
obs_lon_subset = [obs_lon[np.where(np.array(dtimes) == x)].data[0] for x in model_dtimes]

# Grib2 post files
flist = glob.glob("/glade/scratch/harrold/uni_phys/output/DYNAMO_13km_GFS_v16/2011101500/postprd/gfs.t00z.prs*")
flist.sort()

# Open the GRIBv2 file with xarray, and store it in an xarray "Dataset"
for f in flist:
  if not "ds" in locals():
    ds = xr.open_dataset(f,engine='cfgrib', backend_kwargs={'filter_by_keys':{'stepType': 'instant', 'typeOfLevel': 'surface', 'cfVarName': 'sst'},'indexpath':''})
  else:
    ds2 = xr.open_dataset(f,engine='cfgrib', backend_kwargs={'filter_by_keys':{'stepType': 'instant', 'typeOfLevel': 'surface', 'cfVarName': 'sst'},'indexpath':''})
    ds3 = xr.concat([ds,ds2],'step')
    del(ds)
    ds = ds3
    del(ds2)

model_jind = [np.argmin(abs(x-ds.latitude.values)) for x in obs_lat_subset]

model_iind = [np.argmin(abs(x-ds.longitude.values)) for x in obs_lon_subset]

time_inds = [x for x in range(0,len(model_jind),1)]

model_sst = [ds.sst.isel(step=t,longitude=y,latitude=x).values.flatten()[0] for t,y,x in tuple(zip(time_inds,model_iind,model_jind))]
print(model_sst)

## Plot time series of model/obs SSTs

# Set the figure size, projection, and extent
plt.figure(figsize=(6,3))

# Plot data
ax = plt.gca()
#ax.tick_params(axis='x', labelrotation=45)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.xlabel('Date', fontsize=6)
plt.ylabel('SST (K)', fontsize=6)
ax.set_ylim([300,310])
plt.title('R/V Roger Revelle (black) vs. GFSv16 (blue)', fontsize=6)
#ax.legend(('OBS','GFSv16'), loc="upper left")

ax.plot(model_dtimes, obs_sst_subset, color='black', linewidth=0.5)
ax.plot(model_dtimes, model_sst, color='blue', linewidth=0.5)

# Save the plot as a PNG image
plt_ts_name = f"/glade/scratch/harrold/uni_phys/plots/obs/rv_rev_sst_cruise2.png"
plt.savefig(plt_ts_name, format='png', dpi=360)


## Plot map plot of cruise track

#lat_1 = -11.0
#lon_1 = 50.0
#lat_2 = 11.0
#lon_2 = 102.0

# Set the figure size, projection, and extent
#fig = plt.figure(figsize=(8,4))
#ax = plt.axes(projection=ccrs.PlateCarree())
#ax.coastlines(resolution="50m",linewidth=1)
#ax.set_extent([lon_1, lon_2, lat_1, lat_2])
#gl = ax.gridlines(linestyle='--',color='black')
#gl.bottom_labels = True
#gl.left_labels = True
#gl.xlabel_style = {'size': 6}
#gl.ylabel_style = {'size': 6}

# Create a LineString object by providing a list of tuples of (lon,lat)
#pointTrack = sgeom.LineString(zip(pointdf['lon'],pointdf['lat']))
#pointTrack = sgeom.LineString(zip(lon,lat))

# Add the LineString to the axes object you have. See matplotlib documentation for "add_geometries"
#ax.add_geometries([pointTrack], ccrs.PlateCarree(),facecolor='none',edgecolor='b',linewidth=1.0,linestyle='-')

# Save the plot as a PNG image
#plt_name = f"/glade/scratch/harrold/uni_phys/plots/obs/rv_rev_cruise2_path.png"
#fig.savefig(plt_name, format='png', dpi=360)
