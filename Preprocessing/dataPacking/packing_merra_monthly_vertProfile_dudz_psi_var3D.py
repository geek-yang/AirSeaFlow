# -*- coding: utf-8 -*-
"""
Copyright Netherlands eScience Center
Function        : Packing netCDF for the vertical profile of stream function and wind shear (pressure level) from MERRA2
Author          : Yang Liu (y.liu@esciencecenter.nl)
First Built     : 2019.09.30
Last Update     : 2019.10.03
Contributor     :
Description     : This module aims to load fields from the standard netCDF4 files
                  downloaded directly from online data system of NCAR/UCAR Research
                  Data Archive and compute meridional energy transport at monthly scale
                  on pressure levels.
                  
                  Wind shear is defined as du/dz (an indicator for the baroclinic instability)<br>
                  For the calculation of stream function <br>
                  psi = 2 * pi * R * cos(theta) / g * int (v dp) <br>

                  MERRA2 is a state-of-the-art atmosphere reanalysis product produced
                  by NASA (US). It spans from 1979 to 2017. Natively it is generated on a hybrid
                  sigma grid with a horizontal resolution of 0.5 x 0.625 deg and 42 vertical
                  levels.

                  The processing unit is monthly data, for the sake of memory saving.

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
                  
                  The data is saved on a descending pressure coordinate. In order
                  to use the script, the data should have an ascending coordinate.
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

def retriver(var_key, lev):
    print ('Extract monthly mean fields.')
    # first dimension is time
    u = var_key.variables['U'][0,::-1,:,:]
    v = var_key.variables['V'][0,::-1,:,:]
    T = var_key.variables['T'][0,::-1,:,:]
    ps = var_key.variables['PS'][0,:,:] # surface pressure Pa
    phis = var_key.variables['PHIS'][0,:,:] # surface geopotential height m2/s2
    q = var_key.variables['QV'][0,::-1,:,:]
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
    ######################################################################
    ######       compute wind shear and stokes stream function      ######
    ######################################################################
    # create arrays to store the values
    dudz = np.zeros(u.shape,dtype=float)
    for i in np.arange(len(lev)-2):
        dudz_interim = (u[i,:,:] - u[i+2,:,:]) / (z[i,:,:] - z[i+2,:,:])
    	# below surface ->0
        # correction in need
    	dudz_interim[lev[i]>ps] = 0
        dudz_interim[dudz_interim>1.0] = 0
        dudz_interim[dudz_interim<-1.0] = 0
        dudz_interim[u[i,:,:]>1000] = 0
        dudz_interim[u[i+2,:,:]>1000] = 0
        duze[i+1,:,:] = dudz_interim
    # calculate the stokes stream function
    mass_flux = np.zeros(u.shape,dtype=float)
    psi = np.zeros(u.shape,dtype=float)
    dx = 2 * np.pi * constant['R'] * np.cos(2 * np.pi * lat / 360) / len(lon)
    for i in np.arange(len(lev)-1):
        for j in np.arange(len(lat)):
            mass_flux_interim = dx[j] * (v[i+1,j,:] + v[i,j,:]) / 2 * (lev[i+1] - lev[i]) * 100 / constant['g']
            # below surface ->0
            # correction in need
            mass_flux_interim[lev[i]>ps[j,:]] = 0
            mass_flux_interim[v[i+1,j,:]>1000] = 0
            mass_flux[i+1,j,:] = mass_flux_interim
    for i in np.arange(len(lev)-1):
        psi[i,:,:] = np.sum(mass_flux[0:i+1,:,:],0)
    # take the vertical profile
    t_vert = np.mean(T,2)
    q_vert = np.mean(q,2)
    u_vert = np.mean(u,2)
    v_vert = np.mean(v,2)
    gz_vert = np.mean(z*constant['g'],2)
    dudz_vert = np.mean(dudz,2)
    psi_vert = np.mean(psi,2) * len(lon) # by definition
    
    return t_vert, q_vert, u_vert, v_vert, gz_vert, dudz_vert, psi_vert

def create_netcdf_point (pool_t_vert, pool_q_vert, pool_u_vert, pool_v_vert,
                         pool_z_vert, pool_dudz_vert, pool_psi_vert, output_path,
                         level, latitude):
    print ('*******************************************************************')
    print ('*********************** create netcdf file*************************')
    print ('*******************************************************************')
    #logging.info("Start creating netcdf file for the 2D fields of ERAI at each grid point.")
    # wrap the datasets into netcdf file
    # 'NETCDF3_CLASSIC', 'NETCDF3_64BIT', 'NETCDF4_CLASSIC', and 'NETCDF4'
    data_wrap = Dataset(os.path.join(output_path, 'pressure_merra_monthly_regress_1980_2017_vertProfile_dudz_psi_var3D.nc'),'w',format = 'NETCDF4')
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
    data_wrap.description = 'Monthly mean vertical profile of fields from MERRA2 on pressure level'
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
    level = example_key.variables['lev'][::-1] # from surface to top, with an unit of hPa
    lat = example_key.variables['lat'][:] # ascending
    lon = example_key.variables['lon'][:]
    Dim_latitude = len(lat)
    Dim_longitude = len(lon)
    Dim_level = len(level)
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
    	for j in index_month:
        	# get the key of each variable
            var_key = var_key_retrieve(datapath_3D,i,j)
            t, q, u, v, z, dudz, psi = retriver(var_key, level*100)             
            # get the key of each variable
            pool_t[i-1980,j-1,:,:] = t
            pool_q[i-1980,j-1,:,:] = q
            pool_u[i-1980,j-1,:,:] = u
            pool_v[i-1980,j-1,:,:] = v
            pool_z[i-1980,j-1,:,:] = z
            pool_dudz[i-1980,j-1,:,:] = dudz
            pool_psi[i-1980,j-1,:,:] = psi
    ####################################################################
    ######                 Data Wrapping (NetCDF)                #######
    ####################################################################
    create_netcdf_point(pool_t, pool_q, pool_u, pool_v,
                        pool_z, pool_dudz, pool_psi, output_path,
                        level, lat)
    print ('Packing 3D fields of MERRA2 on pressure level is complete!!!')
    print ('The output is in sleep, safe and sound!!!')