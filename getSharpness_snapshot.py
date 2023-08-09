import os
import netCDF4 as nc
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import geopy.distance
import sys
from scipy.stats import binned_statistic
import glob

#Calculate curvature for pixels 'crop' units away from the edges of the snapshot
crop=5 # must be > 1


#get curvature in km
def smoothnessPROJ(x,y,m,lat,lon):
	xp = x+1
	xdp = geopy.distance.geodesic( (lat[xp][y],lon[xp][y]) , (lat[x][y],lon[x][y]) ).km
	#xdp = geopy.distance.great_circle( (lat[xp][y],lon[xp][y]) , (lat[x][y],lon[x][y]) ).km
	xm = x-1
	xdm = geopy.distance.geodesic((lat[xm][y],lon[xm][y]) , (lat[x][y],lon[x][y])).km
	#xdm = geopy.distance.great_circle((lat[xm][y],lon[xm][y]) , (lat[x][y],lon[x][y])).km
	yp = y+1
	ydp = geopy.distance.geodesic((lat[x][yp],lon[x][yp]) , (lat[x][y],lon[x][y])).km
	#ydp = geopy.distance.great_circle((lat[x][yp],lon[x][yp]) , (lat[x][y],lon[x][y])).km
	ym = y-1
	ydm = geopy.distance.geodesic((lat[x][ym],lon[xp][ym]) , (lat[x][y],lon[x][y])).km
	#ydm = geopy.distance.great_circle((lat[x][ym],lon[xp][ym]) , (lat[x][y],lon[x][y])).km

	Dx = (m[xp][y] - m[x][y])/xdp**2 + (m[xm][y] - m[x][y])/xdm**2
	Dy = (m[x][yp] - m[x][y])/ydp**2 + (m[x][ym] - m[x][y])/ydm**2	
	sox = Dx+Dy

	if(m[xp][y]<0 or m[xm][y]<0 or m[x][yp]<0 or m[x][ym]<0 or 4*m[x][y]<0):
		return -1
	return abs(sox)

#bin data in log-scale
def getRelation(m,lat,lon):
	lx = []
	ly = []
	lso = []
	for x in range(crop,LX-crop):
		print("Loading.. " + str(int(float(100*x)/LX)) + "% <<",flush=True)
		sys.stdout.write("\033[F")
		for y in range(crop,LY-crop):			
			if m[x][y]>0:
				if smoothnessPROJ(x,y,m,lat,lon)>0:
					lx.append(np.log10(m[x][y]))
					ly.append(np.log10(smoothnessPROJ(x,y,m,lat,lon)))
	if len(lx)>0:
		res = binned_statistic(lx,ly, bins=np.arange(-4,2,0.25),statistic='mean')
		bin_mean = res[0]
		bin_edges = res[1]
		bin_std = binned_statistic(lx, ly, bins=np.arange(-4,2,0.25),statistic='std')[0]
		bin_count = binned_statistic(lx, ly, bins=np.arange(-4,2,0.25),statistic='count')[0]
		return bin_edges,bin_mean,bin_std,bin_count
	return [],[],[],[]

X=[]
Y=[]

fn = sys.argv[1]

ds = nc.Dataset("./"+fn)
YC = ds.groups['navigation_data'].variables['longitude']
XC = ds.groups['navigation_data'].variables['latitude']
LX = YC.shape[0]
LY = YC.shape[1]
print("|||  Image size is " + "(" + str(LX) + "," + str(LY) + ")")


chl = ds.groups['geophysical_data'].variables['chlor_a']

XC = np.array(XC)
YC = np.array(YC)
chl = np.array(chl)

rho,curv,e,n_samples = getRelation(chl,XC,YC)

output = open('curv_relation_snp_'+fn+'.dat', "w+")
for i in range(0,len(rho)-1):
	# [p] [curvature] [error] [number of samples]
	output.write(str(rho[i]) + "\t" + str(curv[i]) + "\t" + str(e[i]) + "\t" + str(n_samples[i]) + "\n")

