# -*- coding: utf-8 -*-
"""
Copyright Netherlands eScience Center
Function        : Packing netCDF for the vertical profile of energy transport (pressure level) from MERRA2
Author          : Yang Liu (y.liu@esciencecenter.nl)
First Built     : 2019.09.30
Last Update     : 2019.10.03
Contributor     :
Description     : This module aims to load fields from the standard netCDF4 files
                  downloaded directly from online data system of NCAR/UCAR Research
                  Data Archive and compute meridional energy transport at monthly scale
                  on pressure levels.
                  
                  MERRA2 is a state-of-the-art atmosphere reanalysis product produced
                  by NASA (US). It spans from 1979 to 2017. Natively it is generated on a hybrid
                  sigma grid with a horizontal resolution of 0.5 x 0.625 deg and 42 vertical
                  levels.
                  
                  The processing unit is monthly data, for the sake of memory saving.
                  !! This module can be used to deal with data downloaded from NASA GES system.
Return Values   : netCDF files
Caveat!         : This module is designed to work with a batch of files. Hence, there is
                  pre-requists for the location and arrangement of data. The folder should
                  have the following structure:
                  /MERRA2
                      /MERRA2_100.instM_3d_asm_Np.198001.nc4.nc
                      /MERRA2_100.instM_3d_asm_Np.198002.nc4.nc
                      ...
                      /MERRA2_200.instM_3d_asm_Np.199201.nc4.nc
                      ...
                      ...

                  Please use the default names after downloading from NASA. 
                  The files are in netCDF format. Originally, MERRA2 has ascending lat.
                  The pressure levels are from surface to TOA.
                  
                  ! By default, pressure levels are with an unit of hPa,
                  but surface pressure is Pa!
"""

##########################################################################
###########################   Units vacabulory   #########################
# cpT:  [J / kg K] * [K]     = [J / kg]
# Lvq:  [J / kg] * [kg / kg] = [J / kg]
# gz is [m2 / s2] = [ kg m2 / kg s2 ] = [J / kg]

# multiply by v: [J / kg] * [m / s] => [J m / kg s]
# sum over longitudes [J m / kg s] * [ m ] = [J m2 / kg s]

# integrate over pressure: dp: [Pa] = [N m-2] = [kg m2 s-2 m-2] = [kg s-2]
# [J m2 / kg s] * [Pa] = [J m2 / kg s] * [kg / s2] = [J m2 / s3]
# and factor 1/g: [J m2 / s3] * [s2 /m2] = [J / s] = [Wat]
##########################################################################

import sys
import os
import numpy as np
from netCDF4 import Dataset

# constants
constant = {'g' : 9.80616,      # gravititional acceleration [m / s2]
            'R' : 6371009,      # radius of the earth [m]
            'cp': 1004.64,      # heat capacity of air [J/(Kg*K)]
            'Lv': 2264670,      # Latent heat of vaporization [J/Kg]
            'R_dry' : 286.9,    # gas constant of dry air [J/(kg*K)]
            'R_vap' : 461.5,    # gas constant for water vapour [J/(kg*K)]
            }

################################   Input zone  ######################################
# specify starting and ending time
start_year = 1980
end_year = 2017
# specify data path
# ERAI 3D fields on pressure level
datapath_3D = '/project/Reanalysis/MERRA2/Monthly/Full'
# specify output path for figures
output_path = '/project/Reanalysis/MERRA2/output'
# example file
example_path = '/project/Reanalysis/MERRA2/Monthly/Full/MERRA2_200.instM_3d_asm_Np.199209.nc4.nc'
####################################################################################

def var_key_retrieve(datapath, year, month):
    # get the path to each datasets
    print ("Start retrieving datasets {0} (y) {1} (m)".format(year, month))
    # The shape of each variable is (361,576)
    # Sea Level Pressure
    if year < 1992:
        datapath_full = os.path.join(datapath,
                                     'MERRA2_100.instM_3d_asm_Np.{0}{1}.nc4.nc'.format(year,namelist_month[month-1]))
    elif year < 2001:
        datapath_full = os.path.join(datapath,
                                     'MERRA2_200.instM_3d_asm_Np.{0}{1}.nc4.nc'.format(year,namelist_month[month-1]))
    elif year < 2011:
        datapath_full = os.path.join(datapath,
                                     'MERRA2_300.instM_3d_asm_Np.{0}{1}.nc4.nc'.format(year,namelist_month[month-1]))
    else:
        datapath_full = os.path.join(datapath,
                                     'MERRA2_400.instM_3d_asm_Np.{0}{1}.nc4.nc'.format(year,namelist_month[month-1]))
    # get the variable keys
    var_key = Dataset(datapath_full)
    print ("Retrieving datasets successfully and return the variable key!")
    return var_key


def create_netcdf_point (pool_cpT_vert, pool_gz_vert, pool_Lvq_vert,
                         pool_E_vert, output_path, level, latitude):
    print ('*******************************************************************')
    print ('*********************** create netcdf file*************************')
    print ('*******************************************************************')
    #logging.info("Start creating netcdf file for the 2D fields of ERAI at each grid point.")
    # wrap the datasets into netcdf file
    # 'NETCDF3_CLASSIC', 'NETCDF3_64BIT', 'NETCDF4_CLASSIC', and 'NETCDF4'
    data_wrap = Dataset(os.path.join(output_path, 'pressure_merra_monthly_regress_1980_2017_vertProfile_E.nc'),'w',format = 'NETCDF4')
    # create dimensions for netcdf data
    year_wrap_dim = data_wrap.createDimension('year',Dim_year)
    month_wrap_dim = data_wrap.createDimension('month',Dim_month)
    lat_wrap_dim = data_wrap.createDimension('latitude',Dim_latitude)
    lev_wrap_dim = data_wrap.createDimension('level',Dim_level)
    # create coordinate variable
    year_wrap_var = data_wrap.createVariable('year',np.int32,('year',))
    month_wrap_var = data_wrap.createVariable('month',np.int32,('month',))
    lat_wrap_var = data_wrap.createVariable('latitude',np.float32,('latitude',))
    lev_wrap_var = data_wrap.createVariable('level',np.int32,('level',))
    # create the actual 4d variable
    cpT_vert_wrap_var = data_wrap.createVariable('cpt_vert',np.float64,('year', 'month', 'level', 'latitude'),zlib=True)
    gz_vert_wrap_var = data_wrap.createVariable('gz_vert',np.float64,('year', 'month', 'level', 'latitude'),zlib=True)
    Lvq_vert_wrap_var = data_wrap.createVariable('Lvq_vert',np.float64,('year', 'month', 'level', 'latitude'),zlib=True)
    E_vert_wrap_var = data_wrap.createVariable('E_vert',np.float64,('year', 'month', 'level', 'latitude'),zlib=True)
    # global attributes
    data_wrap.description = 'Monthly mean vertical profile of fields from MERRA2 on pressure level'
    # variable attributes
    lat_wrap_var.units = 'degree_north'
    lev_wrap_var.units = 'hPa'

    cpT_vert_wrap_var.units = 'Tera Watt'
    gz_vert_wrap_var.units = 'Tera Watt'
    Lvq_vert_wrap_var.units = 'Tera Watt'
    E_vert_wrap_var.units = 'Tera Watt'

    cpT_vert_wrap_var.long_name = 'temperature transport'
    gz_vert_wrap_var.long_name = 'geopotential transport'
    Lvq_vert_wrap_var.long_name = 'latent heat transport'
    E_vert_wrap_var.long_name = 'total heat transport'

    # writing data
    lat_wrap_var[:] = latitude
    lev_wrap_var[:] = level
    month_wrap_var[:] = index_month
    year_wrap_var[:] = period

    cpT_vert_wrap_var[:] = pool_cpT_vert
    gz_vert_wrap_var[:] = pool_gz_vert
    Lvq_vert_wrap_var[:] = pool_Lvq_vert
    E_vert_wrap_var[:] = pool_E_vert

    # close the file
    data_wrap.close()
    print ("The generation of netcdf files for fields on surface is complete!!")

def retriver(var_key, lev):
    print ('Extract monthly mean fields.')
    # first dimension is time
    v = var_key.variables['V'][0,:,:,:]
    T = var_key.variables['T'][0,:,:,:]
    ps = var_key.variables['PS'][0,:,:] # surface pressure Pa
    phis = var_key.variables['PHIS'][0,:,:] # surface geopotential height m2/s2
    q = var_key.variables['QV'][0,:,:,:]
    ######################################################################
    ######      compute geopotential with hypsometric function      ######
    ######          z2 - z1 = Rd * Tv / g0 * ln(p1 - p2)            ######
    ######   more details can be found in ECMWF IFS documentation   ######
    ######   ECMWF IFS 9220 part III numerics equation 2.20 - 2.23  ######
    ######################################################################
    # create space for geopotential height
    z = np.zeros(T.shape, dtype=float)
    # compute the moist temperature (virtual temperature)
    Tv = T * (1 + (constant['R_vap'] / constant['R_dry'] - 1) * q)
    for i in np.arange(len(lev)):
    	z_interim = phis / constant['g'] + constant['R_dry'] * Tv[i,:,:] / constant['g'] * np.log(ps / lev[i])
    	# below surface ->0
    	z_interim[lev[i]>ps] = 0
    	z[i,:,:] = z_interim
    # calculate geopotential
    Z = z * constant['g']

    return T, q, v, Z

def amet(t, q, v, z, lev, lat, lon):
    # allocation of dp array
    dp_level = np.zeros(lev.shape, dtype=float) # from surface to TOA
    for i in np.arange(len(dp_level)-1):
        dp_level[i] = lev[i] - lev[i+1]
    # the earth is taken as a perfect sphere, instead of a ellopsoid
    dx = 2 * np.pi * constant['R'] * np.cos(2 * np.pi * lat / 360) / len(lon)
    # plugin the weight of grid box width and apply the correction
    dx[0] = 0
    # data allocation
    cpT = np.zeros(t.shape, dtype=float)
    gz = np.zeros(t.shape, dtype=float)
    Lvq = np.zeros(t.shape, dtype=float)
    # weight by pressure
    for i in np.arange(len(dp_level)):
        # weight by longitudinal length
        for j in np.arange(len(lat)):
            cpT[i,j,:] =  constant['cp'] * t[i,j,:] * v[i,j,:] * dp_level[i] / constant['g'] * dx[j]
            gz[i,j,:] = z[i,j,:] * v[i,j,:] * dp_level[i] / constant['g'] * dx[j]
            Lvq[i,j,:] = constant['Lv'] * q[i,j,:] * v[i,j,:] * dp_level[i] / constant['g'] * dx[j]
    # take the vertical profile - summation
    cpT_vert = np.sum(cpT,2)
    gz_vert = np.sum(gz,2)
    Lvq_vert = np.sum(Lvq,2)
    E_vert = cpT_vert + gz_vert + Lvq_vert

    return cpT_vert, gz_vert, Lvq_vert, E_vert

if __name__=="__main__":
    ####################################################################
    ######  Create time namelist matrix for variable extraction  #######
    ####################################################################
    # date and time arrangement
    # namelist of month and days for file manipulation
    namelist_month = ['01','02','03','04','05','06','07','08','09','10','11','12']
    # index of months
    period = np.arange(start_year,end_year+1,1)
    index_month = np.arange(1,13,1)
    ####################################################################
    ######       Extract invariant and calculate constants       #######
    ####################################################################
    # get invariant from benchmark file
    Dim_year = len(period)
    Dim_month = len(index_month)
    example_key = Dataset(example_path)
    # shape [1, 42, 361, 576]
    level = example_key.variables['lev'][:] # from surface to top, hPa!
    lat = example_key.variables['lat'][:] # ascending
    lon = example_key.variables['lon'][:]
    Dim_latitude = len(lat)
    Dim_longitude = len(lon)
    Dim_level = len(level)
    #############################################
    #####   Create space for stroing data   #####
    #############################################
    # data pool
    pool_cpT = np.zeros((Dim_year, Dim_month, Dim_level, Dim_latitude),dtype = float)
    pool_gz = np.zeros((Dim_year, Dim_month, Dim_level, Dim_latitude),dtype = float)
    pool_Lvq = np.zeros((Dim_year, Dim_month, Dim_level, Dim_latitude),dtype = float)
    pool_E = np.zeros((Dim_year, Dim_month, Dim_level, Dim_latitude),dtype = float)
    # loop for calculation
    for i in period:
    	for j in index_month:
        	# get the key of each variable
        	var_key = var_key_retrieve(datapath_3D,i,j)
        	t, q, v, z = retriver(var_key, level*100)
        	cpT, gz, Lvq, E = amet(t, q, v, z, level*100, lat, lon)
        	pool_cpT[i-1980,j-1,:,:] = cpT / 1E+12 # unit is tera watt
        	pool_gz[i-1980,j-1,:,:] = gz / 1E+12
        	pool_Lvq[i-1980,j-1,:,:] = Lvq / 1E+12
       		pool_E[i-1980,j-1,:,:] = E / 1E+12
    ####################################################################
    ######                 Data Wrapping (NetCDF)                #######
    ####################################################################
    create_netcdf_point(pool_cpT, pool_gz, pool_Lvq,
                        pool_E, output_path, level, lat)
    print ('Packing 3D fields of MERRA2 on pressure level is complete!!!')
    print ('The output is in sleep, safe and sound!!!')