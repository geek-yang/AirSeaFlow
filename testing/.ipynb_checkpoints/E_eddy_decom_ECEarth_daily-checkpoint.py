#!/usr/bin/env python
"""
Copyright Netherlands eScience Center
Function        : Quantify stationary and transient eddy from atmospheric meridional energy transport (ERA-Interim)(HPC-cloud customised)
Author          : Yang Liu
Date            : 2018.11.30
Last Update     : 2019.11.26
Description     : The code aims to calculate the time and space dependent components
                  of atmospheric meridional energy transport based on atmospheric
                  reanalysis dataset ERA-Interim from ECMWF. The complete procedure
                  includes the ecomposition of standing & transient eddies.
                  Much attention should be paid that we have to use daily
                  mean since the decomposition takes place at subdaily level could introduce
                  non-meaningful oscillation due to daily cycling.
                  The procedure is generic and is able to adapt any atmospheric
                  reanalysis datasets, with some changes.
                  Referring to the book "Physics of Climate", the concept of decomposition
                  of circulation is given with full details. As a consequence, the meridional
                  energy transport can be decomposed into 4 parts:
                  @@@   A = [/overbar{A}] + /ovrebar{A*} + [A]' + A'*   @@@
                  [/overbar{A}]:    energy transport by steady mean circulation
                  /ovrebar{A*}:     energy transport by stationary eddy
                  [A]':             energy transport by transient eddy
                  A'*:              energy transport by instantaneous and asymmetric part
                  An example is given at page 277, in terms of transport of moisture.
                  Here we will focus on three components of total meridional energy
                  transport:
                  @@@   [/overbar{vT}] = [/overbar{v}] x [/overbar{T}] + [/overbar{v}* x /overbar{T}*] + [/overbar{v'T'}]   @@@
                  [/overbar{v}] x [/overbar{T}]:    energy transport by steady mean circulation
                  [/overbar{v}* x /overbar{T}*]:    energy transport by stationary eddy
                  [/overbar{v'T'}]:                 energy transport by transient eddy
                  Due to a time dependent surface pressure, we will take the vertical
                  integral first and then decompose the total energy transport. Hence,
                  we actually harness the equation of single variable. Thus, we will calculate
                  all the 4 components.
Return Value    : NetCFD4 data file
Dependencies    : os, time, numpy, netCDF4, sys, matplotlib
variables       : Absolute Temperature              T
                  Specific Humidity                 q
                  Logarithmic Surface Pressure      lnsp
                  Zonal Divergent Wind              u
                  Meridional Divergent Wind         v
		          Surface geopotential  	        z
Caveat!!	    : The dataset is from 20 deg north to 90 deg north (Northern Hemisphere).
		          Attention should be paid when calculating the meridional grid length (dy)!
"""

import sys
import numpy as np
import time as tttt
from netCDF4 import Dataset, num2date
import os

##########################################################################
###########################   Units vacabulory   #########################
# cpT:  [J / kg K] * [K]     = [J / kg]
# Lvq:  [J / kg] * [kg / kg] = [J / kg]
# gz in [m2 / s2] = [ kg m2 / kg s2 ] = [J / kg]

# multiply by v: [J / kg] * [m / s] => [J m / kg s]
# sum over longitudes [J m / kg s] * [ m ] = [J m2 / kg s]

# integrate over pressure: dp: [Pa] = [N m-2] = [kg m2 s-2 m-2] = [kg s-2]
# [J m2 / kg s] * [Pa] = [J m2 / kg s] * [kg / s2] = [J m2 / s3]
# and factor 1/g: [J m2 / s3] * [s2 /m2] = [J / s] = [Wat]
##########################################################################

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
start_year = 1991
end_year = 1991
# choose the slice number for the vertical layer
#  pressure levels: (0)200, (1)300, (2)400, (3)500, (4)600, (5)750, (6)850, (7)950
lev_slice = 0
# specify data path
#datapath = '/home/ESLT0068/WorkFlow/Core_Database_AMET_OMET_reanalysis/ERAI/regression/pressure/daily'
datapath = '/home/ESLT0068/WorkFlow/Core_Database_BlueAction_WP3/ECEarth_NLeSC/exp1'
# specify output path for figures
output_path = '/home/ESLT0068/WorkFlow/Core_Database_BlueAction_WP3/ECEarth_NLeSC/exp1'
# benchmark datasets for basic dimensions
benchmark_file = 'ECE_W5S0_T_daily_1991_v2.nc'
benchmark = Dataset(os.path.join(datapath, benchmark_file))
####################################################################################

def var_key_retrieve(datapath, year):
    # get the path to each datasets
    print ("Start retrieving datasets {} (y)".format(year))
    # The shape of each variable is (241,480)
    datapath_T = os.path.join(datapath, 'ECE_W5S0_T_daily_{}_v2.nc'.format(year))
    datapath_v = os.path.join(datapath, 'ECE_W5S0_V_daily_{}_v2.nc'.format(year))
    # get the variable keys
    var_key_v = Dataset(datapath_v)
    var_key_T = Dataset(datapath_T)

    print ("Retrieving datasets successfully and return the variable key!")
    return var_key_v, var_key_T

def initialization(benchmark):
    print ("Prepare for the main work!")
    # create the month index
    period = np.arange(start_year,end_year+1,1)
    index_month = np.arange(1,13,1)
    # create dimensions for saving data
    #Dim_level = len(benchmark.variables['level'][:])
    Dim_latitude = len(benchmark.variables['lat'][:])
    Dim_longitude = len(benchmark.variables['lon'][:])
    Dim_month = len(index_month)
    Dim_period = len(period)
    #latitude = benchmark.variables['latitude'][:]
    #longitude = benchmark.variables['longitude'][:]
    #Dim_time = len(benchmark.variables['time'][:])
    # a list of the index of starting day in each month
    month_day_length = [31,28,31,30,31,30,31,31,30,31,30,31] #! we ignore the last day of February for the leap year
    month_day_index = [0,31,59,90,120,151,181,212,243,273,304,334]
    # create variables
    v_temporal_sum = np.zeros((365,Dim_latitude,Dim_longitude),dtype=float) #! we ignore the last day of February for the leap year
    T_temporal_sum = np.zeros((365,Dim_latitude,Dim_longitude),dtype=float)

    return period, index_month, Dim_latitude, Dim_longitude, Dim_month, Dim_period,\
           month_day_length, month_day_index, v_temporal_sum, T_temporal_sum

def pick_var(var_key_v, var_key_T, year):
    # extract variables
    T = var_key_T.variables['T'][:,lev_slice,:,:]
    v = var_key_v.variables['V'][:,lev_slice,:,:]
    v = v.filled(fill_value=0) # turn filled values to 0
    T = T.filled(fill_value=0)
    # Feb 29th
    if year%4 == 0:
        v_out = np.zeros((365,Dim_latitude,Dim_longitude),dtype=float)
        T_out = np.zeros((365,Dim_latitude,Dim_longitude),dtype=float)
        v_out[0:59,:,:] = v[0:59,:,:]
        v_out[59:,:,:] = v[60:,:,:]
        T_out[0:59,:,:] = T[0:59,:,:]
        T_out[59:,:,:] = T[60:,:,:]
    else:
        v_out = v
        T_out = T
    print ('Extracting variables successfully!')

    return v_out, T_out

def initialization_eddy():
    # Here we only use the temporal mean, for the spatial mean we will take it dynamically
    # during the calculation of eddies, for the sake of memory usage.
    # create space for eddies
    var_cpT_transient_pool = np.zeros((Dim_period,Dim_month,Dim_latitude,Dim_longitude),dtype=float)
    # create space for overall momentum
    var_cpT_overall_pool = np.zeros((Dim_period,Dim_month,Dim_latitude,Dim_longitude),dtype=float)

    return var_cpT_transient_pool, var_cpT_overall_pool

def compute_eddy(var_v_temporal_mean_select, var_T_temporal_mean_select,
                 var_v, var_T):
    '''
    We follow the method given by Peixoto and Oort, 1983.
    The equation is listed on page 61-63.
    equation 4.6 and 4.10
    The example is given on page 288.
    Here we take our naming convention for different eddies.
    For the details, please visit "Transient & Standing eddy"
    in notes.
    '''
    # shape of v[days,lat,lon]
    # calculate transient eddies
    ################# transient eddy ###################
    print ("Calculate transient eddies!")
    var_v_prime = var_v - var_v_temporal_mean_select
    var_T_prime = var_T - var_T_temporal_mean_select
    # eddy
    var_cpT_transient = var_v_prime * var_T_prime
    ####################################################
    # calculate the overall energy transport
    ##############   overall transport   ###############
    print ("Calculate overall energy transport!")
    var_cpT_overall = var_v * var_T
    # monthly mean
    var_cpT_transient_monthly_mean = np.zeros((12,Dim_latitude,Dim_longitude),dtype=float)
    var_cpT_overall_monthly_mean = np.zeros((12,Dim_latitude,Dim_longitude),dtype=float)
    for j in index_month:
        var_cpT_transient_monthly_mean[j-1,:,:] = np.mean(var_cpT_transient[month_day_index[j-1]:month_day_index[j-1]+month_day_length[j-1],:,:],0)
        var_cpT_overall_monthly_mean[j-1,:,:] = np.mean(var_cpT_overall[month_day_index[j-1]:month_day_index[j-1]+month_day_length[j-1],:,:],0)
    
    return var_cpT_transient_monthly_mean, var_cpT_overall_monthly_mean

def create_netcdf_point_eddy(var_cpT_overall,var_cpT_transient,
                             output_path):
    # take the zonal mean
    var_cpT_overall_zonal = np.mean(var_cpT_overall,3)
    var_cpT_transient_zonal = np.mean(var_cpT_transient,3)
    # create netCDF
    print ('*******************************************************************')
    print ('*********************** create netcdf file*************************')
    print ('*******************************************************************')
    # wrap the datasets into netcdf file
    # 'NETCDF3_CLASSIC', 'NETCDF3_64BIT', 'NETCDF4_CLASSIC', and 'NETCDF4'
    data_wrap = Dataset(os.path.join(output_path,'model_daily_E_eddies_point.nc'),'w',format = 'NETCDF4')
    # create dimensions for netcdf data
    year_wrap_dim = data_wrap.createDimension('year',Dim_period)
    month_wrap_dim = data_wrap.createDimension('month',Dim_month)
    lat_wrap_dim = data_wrap.createDimension('latitude',Dim_latitude)
    lon_wrap_dim = data_wrap.createDimension('longitude',Dim_longitude)
    # create coordinate variables for 1-dimensions
    year_wrap_var = data_wrap.createVariable('year',np.int32,('year',))
    month_wrap_var = data_wrap.createVariable('month',np.int32,('month',))
    lat_wrap_var = data_wrap.createVariable('latitude',np.float32,('latitude',))
    lon_wrap_var = data_wrap.createVariable('longitude',np.float32,('longitude',))
    # create the 4d variable
    var_cpT_overall_wrap_var = data_wrap.createVariable('cpT_overall',np.float64,('year','month','latitude','longitude'),zlib=True)
    var_cpT_transient_wrap_var = data_wrap.createVariable('cpT_transient',np.float64,('year','month','latitude','longitude'),zlib=True)
    # create the 3d variable
    var_cpT_overall_zonal_wrap_var = data_wrap.createVariable('cpT_overall_zonal',np.float64,('year','month','latitude'),zlib=True)
    var_cpT_transient_zonal_wrap_var = data_wrap.createVariable('cpT_transient_zonal',np.float64,('year','month','latitude'),zlib=True)
    # global attributes
    data_wrap.description = 'Monthly transient eddies at each grid point'
    # variable attributes
    lat_wrap_var.units = 'degree_north'
    lon_wrap_var.units = 'degree_east'
    
    var_cpT_overall_wrap_var.units = 'K m/s'
    var_cpT_transient_wrap_var.units = 'K m/s'

    lat_wrap_var.long_name = 'Latitude'
    lon_wrap_var.long_name = 'Longitude'
    
    var_cpT_overall_wrap_var.long_name = 'Northward transport of temperature by all motions'
    var_cpT_transient_wrap_var.long_name = 'Northward transport of temperature by transient eddy'
    
    # writing data
    year_wrap_var[:] = period
    month_wrap_var[:] = index_month
    lat_wrap_var[:] = benchmark.variables['lat'][:]
    lon_wrap_var[:] = benchmark.variables['lon'][:]
    
    var_cpT_overall_wrap_var[:] = var_cpT_overall
    var_cpT_transient_wrap_var[:] = var_cpT_transient

    # close the file
    data_wrap.close()
    print ("The generation of netcdf files for fields on surface is complete!!")
    
if __name__=="__main__":
    # calculate the time for the code execution
    start_time = tttt.time()
    # initialization
    period, index_month, Dim_latitude, Dim_longitude, Dim_month, Dim_period,\
    month_day_length, month_day_index, v_temporal_sum, T_temporal_sum = initialization(benchmark)
    print ('*******************************************************************')
    print ('************  calculate the temporal and spatial mean  ************')
    print ('*******************************************************************')
    for i in period:
        # get the key of each variable
        variable_key_v, variable_key_T = var_key_retrieve(datapath,i)
        # take the daily mean of target fields
        var_v, var_T = pick_var(variable_key_v, variable_key_T, i)
        # add daily field to the summation operator
        v_temporal_sum[:] = v_temporal_sum[:] + var_v
        T_temporal_sum[:] = T_temporal_sum[:] + var_T
    # calculate the temporal mean
    v_temporal_mean = v_temporal_sum / Dim_period
    T_temporal_mean = T_temporal_sum / Dim_period
    print ('*******************************************************************')
    print ('**********  calculate the stationary and transient eddy  **********')
    print ('*******************************************************************')
    # Initialization
    # Grab temporal & spatial mean
    # The mean meridional circulation is calculated here
    var_cpT_transient_pool, var_cpT_overall_pool = initialization_eddy()
    # start the loop for the computation of eddies
    for i in period:
        # get the key of each variable
        variable_key_v, variable_key_T = var_key_retrieve(datapath,i)
        # take the daily mean of target fields
        var_v, var_T = pick_var(variable_key_v, variable_key_T, i)
        # calculate the eddies
        var_cpT_transient, var_cpT_overall = compute_eddy(v_temporal_mean, 
                                                          T_temporal_mean, var_v, var_T)
        # save output to the data pool for netCDF
        var_cpT_overall_pool[i-start_year,:,:,:] = var_cpT_overall
        var_cpT_transient_pool[i-start_year,:,:,:] = var_cpT_transient
    # correction applied to the unrealistic values due to the partial cells
    var_cpT_transient_pool[var_cpT_transient_pool>500] = 0
    var_cpT_transient_pool[var_cpT_transient_pool<-500] = 0
    create_netcdf_point_eddy(var_cpT_overall_pool,var_cpT_transient_pool,
                             output_path)
    print ('The full pipeline of the decomposition of meridional energy transport in the atmosphere is accomplished!')
    print ("--- %s minutes ---" % ((tttt.time() - start_time)/60))