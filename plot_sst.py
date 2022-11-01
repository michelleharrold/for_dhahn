# Plot SST

import numpy as np
import xarray as xr
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import argparse
import time,os,sys

# Define required positional arguments
parser = argparse.ArgumentParser()
parser.add_argument("Cycle date/time in YYYYMMDDHH format")
parser.add_argument("Forecast hour in HHH format")
args = parser.parse_args()

ymdh = str(sys.argv[1])
ymd = ymdh[0:8]
year = int(ymdh[0:4])
month = int(ymdh[4:6])
day = int(ymdh[6:8])
hour = int(ymdh[8:10])
cyc = str(hour).zfill(2)
print(year, month, day, hour)

fhr = int(sys.argv[2])
fhr = str(fhr).zfill(3)
print('fhr '+fhr)

# Path ro grib2 file
f = f"/glade/scratch/harrold/uni_phys/output/DYNAMO_13km_GFS_v16/{ymdh}/postprd/gfs.t00z.prslevf{fhr}.tm00.grib2"

# Open the GRIBv2 file with xarray, and store it in an xarray "Dataset"
ds = xr.open_dataset(f,engine='cfgrib', backend_kwargs={'filter_by_keys':{'stepType': 'instant', 'typeOfLevel': 'surface'},'indexpath':''})

# Print the ds object to see it
#print(ds)

# Read in vars from file
lat = ds.latitude
lon = ds.longitude
sst = ds.sst

# Plot the data using matplotlib and cartopy

# Set the figure size, projection, and extent
fig = plt.figure(figsize=(8,4))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines(resolution="50m",linewidth=1)
gl = ax.gridlines(linestyle='--',color='black')
gl.bottom_labels = True
gl.left_labels = True
gl.xlabel_style = {'size': 6}
gl.ylabel_style = {'size': 6}

# Set contour levels, then draw the plot and a colorbar
clevs = np.arange(280,310,1)
plt.contourf(lon, lat, sst, clevs, transform=ccrs.PlateCarree(),cmap=plt.cm.jet,extend='both')
plt.title('Sea Surface Temperature (K)', loc='left', size=8)
plt.title(f"Init: {ymd} {cyc} UTC f{fhr}", loc='right', size=8)
cb = plt.colorbar(ax=ax, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
cb.set_label('K',size=8,rotation=0,labelpad=15)
cb.ax.tick_params(labelsize=8)

# Save the plot as a PNG image
plt_name = f"/glade/scratch/harrold/uni_phys/plots/DYNAMO_13km_GFS_v16/{ymdh}/sst_f{fhr}.png"
fig.savefig(plt_name, format='png', dpi=360)
