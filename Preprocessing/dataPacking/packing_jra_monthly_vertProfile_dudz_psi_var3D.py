# -*- coding: utf-8 -*-
"""
Copyright Netherlands eScience Center
Function        : Packing netCDF for the vertical profile of stream function and wind shear (pressure level) from JRA55
Author          : Yang Liu (y.liu@esciencecenter.nl)
First Built     : 2019.09.30
Last Update     : 2019.10.03
Contributor     :
Description     : This module aims to load fields from the standard GRIB files
                  downloaded directly from online data system of NCAR/UCAR Research
                  Data Archive and compute stream function and wind shear at monthly scale
                  on pressure levels.
                  
                  Wind shear is defined as du/dz (an indicator for the baroclinic instability)<br>
                  For the calculation of stream function <br>
                  psi = 2 * pi * R * cos(theta) / g * int (v dp) <br>

                  JRA55 is a state-of-the-art atmosphere reanalysis product produced
                  by JMA (Japan). It spans from 1979 to 2015. Natively it is generated on a hybrid
                  sigma grid with a horizontal resolution of 0.56 x 0.56 deg and 60 vertical
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
# ERAI 3D fields on pressure level
datapath_3D = '/project/Reanalysis/JRA55/Monthly/pressure'
# specify output path for figures
output_path = '/project/Reanalysis/JRA55/Monthly/output'
# example file
example = '/project/Reanalysis/JRA55/Monthly/pressure/anl_p125.011_tmp.201301_201312'
####################################################################################

def var_retrieve_year(datapath, year, level, level_q):
    # get the path to each datasets
    print ("Start retrieving datasets {0} (y)".format(year))
    # The shape of each variable is (145,288)
    # create space for the output
    T = np.zeros((Dim_month, Dim_level, Dim_latitude, Dim_longitude), dtype=float)
    q = np.zeros((Dim_month, Dim_level_q, Dim_latitude, Dim_longitude), dtype=float)
    u = np.zeros((Dim_month, Dim_level, Dim_latitude, Dim_longitude), dtype=float)
    v = np.zeros((Dim_month, Dim_level, Dim_latitude, Dim_longitude), dtype=float)
    z = np.zeros((Dim_month, Dim_level, Dim_latitude, Dim_longitude), dtype=float)
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
                          'anl_p125.007_hgt.{0}01_{1}12'.format(year,year)))
    # extract data
    # reset counters
    counter_time = 0
    counter_lev = 0
    counter_message = 1
    while (counter_message <= Dim_level*12):
        # take the key
        key_T = key_tmp.message(counter_message)
        key_u = key_ugrd.message(counter_message)
        key_v = key_vgrd.message(counter_message)
        key_z = key_hgt.message(counter_message)
        # 27 levels (0-26) # descending
        if counter_lev == Dim_level:
            counter_lev = 0
            counter_time = counter_time + 1
        # take the values
        T[counter_time,counter_lev,:,:] = key_T.values
        u[counter_time,counter_lev,:,:] = key_u.values
        v[counter_time,counter_lev,:,:] = key_v.values
        z[counter_time,counter_lev,:,:] = key_z.values
        # push the counter
        counter_lev = counter_lev + 1
        counter_message = counter_message + 1
    # for q
    # reset counters
    counter_time = 0
    counter_lev = 0
    counter_message = 1
    while (counter_message <= Dim_level_q*12):
        # take the key
        key_q = key_spfh.message(counter_message)
        # 27 levels (0-26) # descending
        if counter_lev == Dim_level_q:
            counter_lev = 0
            counter_time = counter_time + 1
        # take the values
        q[counter_time,counter_lev,:,:] = key_q.values
        # push the counter
        counter_lev = counter_lev + 1
        counter_message = counter_message + 1
    # close all the grib files
    key_tmp.close()
    key_spfh.close()
    key_ugrd.close()
    key_vgrd.close()
    key_hgt.close()

    print ("Retrieving datasets successfully and return the variables!")
    return T, q, u, v, z

def var_retrieve_month(datapath, year, month, level, level_q):
    # get the path to each datasets
    print ("Start retrieving datasets {0} (y) {1} (m)".format(year, namelist_month[month-1]))
    # The shape of each variable is (145,288)
    # create space for the output
    T = np.zeros((Dim_level, Dim_latitude, Dim_longitude), dtype=float)
    q = np.zeros((Dim_level_q, Dim_latitude, Dim_longitude), dtype=float)
    u = np.zeros((Dim_level, Dim_latitude, Dim_longitude), dtype=float)
    v = np.zeros((Dim_level, Dim_latitude, Dim_longitude), dtype=float)
    z = np.zeros((Dim_level, Dim_latitude, Dim_longitude), dtype=float)
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
    counter_lev = 0
    counter_message = 1
    while (counter_message <= Dim_level):
        # take the key
        key_T = key_tmp.message(counter_message)
        key_u = key_ugrd.message(counter_message)
        key_v = key_vgrd.message(counter_message)
        key_z = key_hgt.message(counter_message)
        # 27 levels (0-26) # descending
        # take the values
        T[counter_lev,:,:] = key_T.values
        u[counter_lev,:,:] = key_u.values
        v[counter_lev,:,:] = key_v.values
        z[counter_lev,:,:] = key_z.values
        # push the counter
        counter_lev = counter_lev + 1
        counter_message = counter_message + 1
    # reset counters
    counter_lev = 0
    counter_message = 1
    while (counter_message <= Dim_level_q):
        # take the key
        key_q = key_spfh.message(counter_message)
        # take the values
        q[counter_lev,:,:] = key_q.values
        # push the counter
        counter_lev = counter_lev + 1
        counter_message = counter_message + 1
    # close all the grib files
    key_tmp.close()
    key_spfh.close()
    key_ugrd.close()
    key_vgrd.close()
    key_hgt.close()
    
    print ("Retrieving datasets successfully and return the variables!")
    return T, q, u, v, z

def create_netcdf_point (pool_t_vert, pool_q_vert, pool_u_vert, pool_v_vert,
                         pool_z_vert, pool_dudz_vert, pool_psi_vert, output_path,
                         level, latitude):
    print ('*******************************************************************')
    print ('*********************** create netcdf file*************************')
    print ('*******************************************************************')
    #logging.info("Start creating netcdf file for the 2D fields of ERAI at each grid point.")
    # wrap the datasets into netcdf file
    # 'NETCDF3_CLASSIC', 'NETCDF3_64BIT', 'NETCDF4_CLASSIC', and 'NETCDF4'
    data_wrap = Dataset(os.path.join(output_path, 'pressure_jra_monthly_regress_1979_2017_vertProfile_dudz_psi_var3D.nc'),'w',format = 'NETCDF4')
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
    t_vert_wrap_var = data_wrap.createVariable('t_vert',np.float64,('year', 'month', 'level', 'latitude'),zlib=True)
    q_vert_wrap_var = data_wrap.createVariable('q_vert',np.float64,('year', 'month', 'level', 'latitude'),zlib=True)
    u_vert_wrap_var = data_wrap.createVariable('u_vert',np.float64,('year', 'month', 'level', 'latitude'),zlib=True)
    v_vert_wrap_var = data_wrap.createVariable('v_vert',np.float64,('year', 'month', 'level', 'latitude'),zlib=True)
    z_vert_wrap_var = data_wrap.createVariable('z_vert',np.float64,('year', 'month', 'level', 'latitude'),zlib=True)
    dudz_vert_wrap_var = data_wrap.createVariable('dudz_vert',np.float64,('year', 'month', 'level', 'latitude'),zlib=True)
    psi_vert_wrap_var = data_wrap.createVariable('psi_vert',np.float64,('year', 'month', 'level', 'latitude'),zlib=True)
    # global attributes
    data_wrap.description = 'Monthly mean vertical profile of fields from JRA55 on pressure level'
    # variable attributes
    lat_wrap_var.units = 'degree_north'
    lev_wrap_var.units = 'hPa'

    t_vert_wrap_var.units = 'K'
    q_vert_wrap_var.units = 'kg/kg'
    u_vert_wrap_var.units = 'm/s'
    v_vert_wrap_var.units = 'm/s'
    z_vert_wrap_var.units = 'm2/s2'
    dudz_vert_wrap_var.units = '/s'
    psi_vert_wrap_var.units = 'kg/s'

    t_vert_wrap_var.long_name = 'temperature'
    q_vert_wrap_var.long_name = 'specific humidity'
    u_vert_wrap_var.long_name = 'zonal wind velocity'
    v_vert_wrap_var.long_name = 'meridional wind velocity'
    z_vert_wrap_var.long_name = 'geopotential'
    dudz_vert_wrap_var.long_name = 'zonal wind vertical shear'
    psi_vert_wrap_var.long_name = 'meridional overturning stream function'

    # writing data
    lat_wrap_var[:] = latitude
    lev_wrap_var[:] = level
    month_wrap_var[:] = index_month
    year_wrap_var[:] = period

    t_vert_wrap_var[:] = pool_t_vert
    q_vert_wrap_var[:] = pool_q_vert
    u_vert_wrap_var[:] = pool_u_vert
    v_vert_wrap_var[:] = pool_v_vert
    z_vert_wrap_var[:] = pool_z_vert
    dudz_vert_wrap_var[:] = pool_dudz_vert
    psi_vert_wrap_var[:] = pool_psi_vert

    # close the file
    data_wrap.close()
    print ("The generation of netcdf files for fields on surface is complete!!")
    
def calc(t, q, u, v, gz, level, lat, lon):
    print ('Extract monthly mean fields.')
    # calculate the height
    z = gz / constant['g'] #????
    # calculate the vertical shear
    # create arrays to store the values
    dudz = np.zeros(u.shape,dtype=float)
    for i in np.arange(len(level)-2):
        dudz[:,i+1,:,:] = (u[:,i,:,:] - u[:,i+2,:,:]) / (z[:,i,:,:] - z[:,i+2,:,:])
    # calculate the stokes stream function
    mass_flux = np.zeros(u.shape,dtype=float)
    psi = np.zeros(u.shape,dtype=float)
    dx = 2 * np.pi * constant['R'] * np.cos(2 * np.pi * lat / 360) / len(lon)
    for i in np.arange(len(level)-1):
        for j in np.arange(len(lat)):
            mass_flux[:,i+1,j,:] = dx[j] * (v[:,i+1,j,:] + v[:,i,j,:]) / 2 * (level[i+1] - level[i]) * 100 / constant['g']
    for i in np.arange(len(level)-1):
        psi[:,i,:,:] = np.sum(mass_flux[:,0:i+1,:,:],1)
    # take the vertical profile
    t_vert = np.mean(t,3)
    q_vert = np.mean(q,3)
    u_vert = np.mean(u,3)
    v_vert = np.mean(v,3)
    gz_vert = np.mean(gz,3)
    dudz_vert = np.mean(dudz,3)
    psi_vert = np.mean(psi,3) * len(lon) # by definition
    
    return t_vert, q_vert, u_vert, v_vert, gz_vert, dudz_vert, psi_vert

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
    lev_diff = len(level) - len(level_q)
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
    pool_t = np.zeros((Dim_year, Dim_month, Dim_level, Dim_latitude),dtype = float)
    pool_q = np.zeros((Dim_year, Dim_month, Dim_level, Dim_latitude),dtype = float)
    pool_u = np.zeros((Dim_year, Dim_month, Dim_level, Dim_latitude),dtype = float)
    pool_v = np.zeros((Dim_year, Dim_month, Dim_level, Dim_latitude),dtype = float)
    pool_z = np.zeros((Dim_year, Dim_month, Dim_level, Dim_latitude),dtype = float)
    pool_dudz = np.zeros((Dim_year, Dim_month, Dim_level, Dim_latitude),dtype = float)
    pool_psi = np.zeros((Dim_year, Dim_month, Dim_level, Dim_latitude),dtype = float)
    # loop for calculation
    for i in period:
        # to deal with different data layout
        q = np.zeros((Dim_month, Dim_level, Dim_latitude, Dim_longitude), dtype=float)
        if i < 2014:
            T, q[:,lev_diff:,:,:], u, v,\
            z = var_retrieve_year(datapath_3D, i, level, level_q)
            t_vert, q_vert, u_vert, v_vert, z_vert, dudz,\
            psi = calc(T, q, u, v, z, level, lat, lon)
        else:
            for j in index_month:
                fields_T = np.zeros((Dim_month, Dim_level, Dim_latitude, Dim_longitude), dtype=float)
                fields_q = np.zeros((Dim_month, Dim_level, Dim_latitude, Dim_longitude), dtype=float)
                fields_u = np.zeros((Dim_month, Dim_level, Dim_latitude, Dim_longitude), dtype=float)
                fields_v = np.zeros((Dim_month, Dim_level, Dim_latitude, Dim_longitude), dtype=float)
                fields_z = np.zeros((Dim_month, Dim_level, Dim_latitude, Dim_longitude), dtype=float)
                fields_T[j-1,:,:,:], fields_q[j-1,lev_diff:,:,:], fields_u[j-1,:,:,:],\
                fields_v[j-1,:,:,:], fields_z[j-1,:,:,:] = var_retrieve_month(datapath_3D,
                                                                              i, j, level, level_q)
            t_vert, q_vert, u_vert, v_vert, z_vert, dudz,\
            psi = calc(fields_T, fields_q, fields_u, fields_v,
                       fields_z, level, lat, lon)                
        # get the key of each variable
        pool_t[i-1979,:,:,:] = t_vert
        pool_q[i-1979,:,:,:] = q_vert
        pool_u[i-1979,:,:,:] = u_vert
        pool_v[i-1979,:,:,:] = v_vert
        pool_z[i-1979,:,:,:] = z_vert
        pool_dudz[i-1979,:,:,:] = dudz
        pool_psi[i-1979,:,:,:] = psi
    ####################################################################
    ######                 Data Wrapping (NetCDF)                #######
    ####################################################################
    create_netcdf_point(pool_t, pool_q, pool_u, pool_v,
                        pool_z, pool_dudz, pool_psi, output_path,
                        level, lat)
    print ('Packing 3D fields of JRA55 on pressure level is complete!!!')
    print ('The output is in sleep, safe and sound!!!')