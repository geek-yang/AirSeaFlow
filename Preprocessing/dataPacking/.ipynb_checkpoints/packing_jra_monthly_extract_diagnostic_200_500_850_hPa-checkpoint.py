# -*- coding: utf-8 -*-
"""
Copyright Netherlands eScience Center
Function        : Packing netCDF for variables at diagnostic levels from JRA55
Author          : Yang Liu (y.liu@esciencecenter.nl)
First Built     : 2019.09.30
Last Update     : 2019.10.30
Contributor     :
Description     : This module aims to load fields from the standard GRIB files
                  downloaded directly from online data system of NCAR/UCAR Research
                  Data Archive and compute meridional energy transport at monthly scale
                  on pressure levels.

                  JRA55 is a state-of-the-art atmosphere reanalysis product produced
                  by JMA (Japan). It spans from 1979 to 2015. Natively it is generated on a hybrid
                  sigma grid with a horizontal resolution of 1.25 x 1.25 deg and 60 vertical
                  levels.

                  The processing unit is monthly data, for the sake of memory saving.

                  !! This module can be used to deal with data downloaded from UCAR
                  or JDDS. They are arranged in different ways.

Return Values   : netCDF files
Caveat!         : This module is designed to work with a batch of files. Hence, there is
                  pre-requists for the location and arrangement of data. The folder should
                  have the following structure:
                  /JRA55
                      /anl_p125.007_hgt.197901_197912
                      /anl_p125.007_hgt.198001_198012
                      ...
                      /anl_p125.011_tmp.198001_198012
                      ...
                      /anl_p125.033_ugrd.198001_198012
                      ...
                      /anl_p125.034_vgrd.198001_198012
                      ...
                      /anl_p125.051_spfh.198001_198012
                      ...
                      (since 2014)
                      /anl_p125_hgt.201401
                      /anl_p125_hgt.201402
                      /anl_p125_hgt.201403
                      ...

                  Please use the default names after downloading from NCAR/UCAR Research
                  Data Archive. The files are in GRIB format. Originally, JRA55 has descending lat.
                  The pressure levels are from TOA to surface.
                  
                  The unit o hgt is gpm, which is equal to height.
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
import pygrib
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
start_year = 1979
end_year = 2017
# specify data path
# JRA55 3D fields on pressure level
datapath_3D = '/project/Reanalysis/JRA55/Monthly/pressure'
# specify output path for figures
output_path = '/project/Reanalysis/JRA55/Monthly/output'
# example file
example = '/project/Reanalysis/JRA55/Monthly/pressure/anl_p125.011_tmp.201301_201312'
# diagnostic level list
diag_level = [200, 500, 850]
diag_index = [14, 21, 30]
diag_index_q = [4, 11, 20]
choice = 1
####################################################################################

def var_retrieve_year(datapath, year, level, level_q):
    """
    The vertical levels for specific humidity and other variables are different!
    """
    # get the path to each datasets
    print ("Start retrieving datasets {0} (y)".format(year))
    # The shape of each variable is (145,288)
    # create space for the output
    T = np.zeros((Dim_month, Dim_latitude, Dim_longitude), dtype=float)
    q = np.zeros((Dim_month, Dim_latitude, Dim_longitude), dtype=float)
    u = np.zeros((Dim_month, Dim_latitude, Dim_longitude), dtype=float)
    v = np.zeros((Dim_month, Dim_latitude, Dim_longitude), dtype=float)
    z = np.zeros((Dim_month, Dim_latitude, Dim_longitude), dtype=float)
    # get the keys of data
    key_tmp = pygrib.open(os.path.join(datapath,
                          'anl_p125.011_tmp.{0}01_{1}12'.format(year,year)))
    key_spfh = pygrib.open(os.path.join(datapath,
                           'anl_p125.051_spfh.{0}01_{1}12'.format(year,year)))
    key_ugrd = pygrib.open(os.path.join(datapath,
                           'anl_p125.033_ugrd.{0}01_{1}12'.format(year,year)))
    key_vgrd = pygrib.open(os.path.join(datapath,
                           'anl_p125.034_vgrd.{0}01_{1}12'.format(year,year)))
    key_hgt = pygrib.open(os.path.join(datapath,
                          'anl_p125.007_hgt.{0}01_{1}12'.format(year,year))) # with an unit of gpm, which is height
    # extract data
    # reset counters
    counter_time = 0
    counter_message = diag_index[choice]
    while (counter_message <= Dim_level*12):
        # take the key
        key_T = key_tmp.message(counter_message)
        key_u = key_ugrd.message(counter_message)
        key_v = key_vgrd.message(counter_message)
        key_z = key_hgt.message(counter_message)
        # take the values
        T[counter_time,:,:] = key_T.values
        u[counter_time,:,:] = key_u.values
        v[counter_time,:,:] = key_v.values
        z[counter_time,:,:] = key_z.values
        # push the counter
        counter_time = counter_time + 1
        counter_message = counter_message + Dim_level
    # for q
    # reset counters
    counter_time = 0
    counter_message = diag_index_q[choice]
    while (counter_message <= Dim_level_q*12):
        # take the key
        key_q = key_spfh.message(counter_message)
        # take the values
        q[counter_time,:,:] = key_q.values
        # push the counter
        counter_time = counter_time + 1
        counter_message = counter_message + Dim_level_q
    # close all the grib files
    key_tmp.close()
    key_spfh.close()
    key_ugrd.close()
    key_vgrd.close()
    key_hgt.close()

    print ("Retrieving datasets successfully and return the variables!")
    return T, q, u, v, z * constant['g'] # the unit of z originally is gpm

def var_retrieve_month(datapath, year, month, level, level_q):
    # get the path to each datasets
    print ("Start retrieving datasets {0} (y) {1} (m)".format(year, namelist_month[month-1]))
    # The shape of each variable is (145,288)
    # create space for the output
    T = np.zeros((Dim_latitude, Dim_longitude), dtype=float)
    q = np.zeros((Dim_latitude, Dim_longitude), dtype=float)
    u = np.zeros((Dim_latitude, Dim_longitude), dtype=float)
    v = np.zeros((Dim_latitude, Dim_longitude), dtype=float)
    z = np.zeros((Dim_latitude, Dim_longitude), dtype=float)
    # get the keys of data
    key_tmp = pygrib.open(os.path.join(datapath,
                          'anl_p125_tmp.{0}{1}'.format(year,namelist_month[month-1])))
    key_spfh = pygrib.open(os.path.join(datapath,
                          'anl_p125_spfh.{0}{1}'.format(year,namelist_month[month-1])))
    key_ugrd = pygrib.open(os.path.join(datapath,
                          'anl_p125_ugrd.{0}{1}'.format(year,namelist_month[month-1])))
    key_vgrd = pygrib.open(os.path.join(datapath,
                          'anl_p125_vgrd.{0}{1}'.format(year,namelist_month[month-1])))
    key_hgt = pygrib.open(os.path.join(datapath,
                          'anl_p125_hgt.{0}{1}'.format(year,namelist_month[month-1])))
    # extract data
    # reset counters
    counter_message = diag_index[choice]
    key_T = key_tmp.message(counter_message)
    key_u = key_vgrd.message(counter_message)
    key_v = key_vgrd.message(counter_message)
    key_z = key_hgt.message(counter_message)
    T[:,:] = key_T.values
    v[:,:] = key_v.values
    z[:,:] = key_z.values
    # reset counters
    counter_message = diag_index_q[choice]
    # take the key
    key_q = key_spfh.message(counter_message)
    # take the values
    q[:,:] = key_q.values
    # close all the grib files
    key_tmp.close()
    key_spfh.close()
    key_ugrd.close()
    key_vgrd.close()
    key_hgt.close()
    
    print ("Retrieving datasets successfully and return the variables!")
    return T, q, u, v, z*constant['g'] # the unit of z originally is gpm


def create_netcdf_point (pool_T,  pool_q, pool_u, pool_v,
                         pool_gz, output_path, latitude, longitude):
    print ('*******************************************************************')
    print ('*********************** create netcdf file*************************')
    print ('*******************************************************************')
    #logging.info("Start creating netcdf file for the 2D fields of ERAI at each grid point.")
    # wrap the datasets into netcdf file
    # 'NETCDF3_CLASSIC', 'NETCDF3_64BIT', 'NETCDF4_CLASSIC', and 'NETCDF4'
    data_wrap = Dataset(os.path.join(output_path,
                        'pressure_jra_monthly_regress_1979_2017_var_{0}hPa.nc'.format(diag_level[choice])),'w',format = 'NETCDF4')
    # create dimensions for netcdf data
    year_wrap_dim = data_wrap.createDimension('year',Dim_year)
    month_wrap_dim = data_wrap.createDimension('month',Dim_month)
    lat_wrap_dim = data_wrap.createDimension('latitude',Dim_latitude)
    lon_wrap_dim = data_wrap.createDimension('longitude',Dim_longitude)
    # create coordinate variable
    year_wrap_var = data_wrap.createVariable('year',np.int32,('year',))
    month_wrap_var = data_wrap.createVariable('month',np.int32,('month',))
    lat_wrap_var = data_wrap.createVariable('latitude',np.float32,('latitude',))
    lon_wrap_var = data_wrap.createVariable('longitude',np.float32,('longitude',))
    # create the actual 4d variable
    T_wrap_var = data_wrap.createVariable('T',np.float64,('year', 'month', 'latitude', 'longitude'),zlib=True)
    q_wrap_var = data_wrap.createVariable('q',np.float64,('year', 'month', 'latitude', 'longitude'),zlib=True)
    u_wrap_var = data_wrap.createVariable('u',np.float64,('year', 'month', 'latitude', 'longitude'),zlib=True)
    v_wrap_var = data_wrap.createVariable('v',np.float64,('year', 'month', 'latitude', 'longitude'),zlib=True)
    z_wrap_var = data_wrap.createVariable('z',np.float64,('year', 'month', 'latitude', 'longitude'),zlib=True)
    # global attributes
    data_wrap.description = 'Monthly mean fields from JRA55 on pressure level'
    # variable attributes
    lat_wrap_var.units = 'degree_north'
    lon_wrap_var.units = 'degree_south'

    T_wrap_var.units = 'Kelvin'
    q_wrap_var.units = 'kg/kg'
    u_wrap_var.units = 'm/s'
    v_wrap_var.units = 'm/s'
    z_wrap_var.units = 'm2/s2'

    T_wrap_var.long_name = 'temperature'
    q_wrap_var.long_name = 'specific humidity'
    u_wrap_var.long_name = 'zonal velocity'
    v_wrap_var.long_name = 'meridional velocity'
    z_wrap_var.long_name = 'geopotential'

    # writing data
    lat_wrap_var[:] = latitude
    lon_wrap_var[:] = longitude
    month_wrap_var[:] = index_month
    year_wrap_var[:] = period

    T_wrap_var[:] = pool_T
    q_wrap_var[:] = pool_q
    u_wrap_var[:] = pool_u
    v_wrap_var[:] = pool_v
    z_wrap_var[:] = pool_gz

    # close the file
    data_wrap.close()
    print ("The generation of netcdf files for fields on surface is complete!!")

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
    example_grbs = pygrib.open(example)
    example_key = example_grbs.message(1)
    lats, lons = example_key.latlons()
    lat = lats[:,0] # descending
    lon = lons[0,:]
    level_q = np.array(([100, 125, 150, 175, 200,
                         225, 250, 300, 350, 400,
                         450, 500, 550, 600, 650,
                         700, 750, 775, 800, 825,
                         850, 875, 900, 925, 950,
                         975, 1000]),dtype=int)
    level = np.array(([1, 2, 3, 5, 7, 
                       10, 20, 30, 50, 70, 
                       100, 125, 150, 175, 200,
                       225, 250, 300, 350, 400,
                       450, 500, 550, 600, 650,
                       700, 750, 775, 800, 825,
                       850, 875, 900, 925, 950,
                       975, 1000]),dtype=int)
    ####################################################################
    ######       Extract invariant and calculate constants       #######
    ####################################################################
    # get invariant from benchmark file
    Dim_year = len(period)
    Dim_month = len(index_month)
    Dim_latitude = len(lat)
    Dim_longitude = len(lon)
    Dim_level = len(level)
    Dim_level_q = len(level_q)
    #############################################
    #####   Create space for stroing data   #####
    #############################################
    # data pool
    pool_T = np.zeros((Dim_year, Dim_month, Dim_latitude, Dim_longitude),dtype = float)
    pool_z = np.zeros((Dim_year, Dim_month, Dim_latitude, Dim_longitude),dtype = float)
    pool_q = np.zeros((Dim_year, Dim_month, Dim_latitude, Dim_longitude),dtype = float)
    pool_u = np.zeros((Dim_year, Dim_month, Dim_latitude, Dim_longitude),dtype = float)
    pool_v = np.zeros((Dim_year, Dim_month, Dim_latitude, Dim_longitude),dtype = float)
    # loop for calculation
    for i in period:
        # to deal with different data layout
        q = np.zeros((Dim_month, Dim_level, Dim_latitude, Dim_longitude), dtype=float)
        if i < 2014:
            pool_T[i-1979,:,:,:], pool_q[i-1979,:,:,:], pool_u[i-1979,:,:,:], pool_v[i-1979,:,:,:],\
            pool_z[i-1979,:,:,:] = var_retrieve_year(datapath_3D, i, level, level_q)
        else:
            for j in index_month:
                pool_T[i-1979, j-1,:,:], pool_q[i-1979, j-1,:,:],\
                pool_u[i-1979, j-1,:,:], pool_v[i-1979, j-1,:,:],\
                pool_z[i-1979, j-1,:,:] = var_retrieve_month(datapath_3D, i, j, level, level_q)
    ####################################################################
    ######                 Data Wrapping (NetCDF)                #######
    ####################################################################
    create_netcdf_point(pool_T, pool_q, pool_u, pool_v,
                        pool_z, output_path, lat, lon)
    print ('Packing 3D fields of JRA55 on pressure level is complete!!!')
    print ('The output is in sleep, safe and sound!!!')