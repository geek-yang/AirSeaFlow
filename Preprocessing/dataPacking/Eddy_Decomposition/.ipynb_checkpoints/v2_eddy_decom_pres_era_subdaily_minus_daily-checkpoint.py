
#!/usr/bin/env python
"""
Copyright Netherlands eScience Center
Function        : Quantify stationary and transient eddy from atmospheric kinetic energy (ERA-Interim)(HPC-cloud customised)
Author          : Yang Liu
Date            : 2018.11.30
Last Update     : 2018.11.30
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
start_year = 1979
end_year = 2017
# choose the slice number for the vertical layer
#  pressure levels: (0)200, (1)300, (2)400, (3)500, (4)600, (5)750, (6)850, (7)950
lev_slice = 0
# specify data path
# ERAI 3D fields on pressure level
#datapath = '/home/ESLT0068/WorkFlow/Core_Database_AMET_OMET_reanalysis/ERAI/regression/pressure/daily'
datapath = '/project/Reanalysis/ERA_Interim/Subdaily/Pressure/T_v_z_q'
# specify output path for figures
#output_path = '/home/ESLT0068/WorkFlow/Core_Database_AMET_OMET_reanalysis/ERAI/regression'
output_path = '/project/Reanalysis/ERA_Interim/Subdaily/Pressure/output'
# benchmark datasets for basic dimensions
benchmark_file = 'pressure_daily_075_diagnostic_1998_3_all.nc'
benchmark = Dataset(os.path.join(datapath, 'era1998', benchmark_file))
####################################################################################

def var_key_retrieve(datapath, year, month):
    # get the path to each datasets
    print ("Start retrieving datasets {} (y) {} (m)".format(year,month))
    # The shape of each variable is (241,480)
    datapath = os.path.join(datapath, 'era{}'.format(year),
                            'pressure_daily_075_diagnostic_{}_{}_all.nc'.format(year,month))
    # get the variable keys
    var_key = Dataset(datapath)

    print ("Retrieving datasets successfully and return the variable key!")
    return var_key

def initialization(benchmark):
    print ("Prepare for the main work!")
    # create the month index
    period = np.arange(start_year,end_year+1,1)
    index_month = np.arange(1,13,1)
    # create dimensions for saving data
    #Dim_level = len(benchmark.variables['level'][:])
    Dim_latitude = len(benchmark.variables['latitude'][:])
    Dim_longitude = len(benchmark.variables['longitude'][:])
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
    return period, index_month, Dim_latitude, Dim_longitude, Dim_month, Dim_period,\
           month_day_length, month_day_index, v_temporal_sum

def pick_var(var_key):
    # validate time and location info
    time = var_key.variables['time'][:]
    level = var_key.variables['level'][:]
    latitude = var_key.variables['latitude'][:]
    longitude = var_key.variables['longitude'][:]
    date = num2date(time,var_key.variables['time'].units)
    days = len(time)//4
    print ('*******************************************************************')
    print ('The datasets contain information from %s to %s' % (date[0],date[-1]))
    print ('There are %d days in this month' % (len(time)//4))
    print ('The coordinates include %d vertical levels' % (len(level)))
    print ('The grid employs %d points in latitude, and %d points in longitude' % (len(latitude),len(longitude)))
    print ('*******************************************************************')
    # extract variables
    print ("Start extracting velocity for the calculation of mean over time and space.")
    # extract data at certain levels
    v = np.zeros((len(time),len(latitude),len(longitude)),dtype=float)
    v[:,:,:] = var_key.variables['v'][:,lev_slice,:,:]
    # daily mean
    # first we reshape the array
    v_expand = v.reshape(len(time)//4,4,len(latitude),len(longitude))
    # Then we take daily mean
    v_daily = np.mean(v_expand,1)
    if days == 29:
        v_out = v_daily[:-1,:,:]
    else:
        v_out = v_daily
    print ('Extracting variables successfully!')

    return v_out

def initialization_eddy(v_temporal_mean):
    # Here we only use the temporal mean, for the spatial mean we will take it dynamically
    # during the calculation of eddies, for the sake of memory usage.
    # create space for eddies
    var_v2_transient_pool = np.zeros((Dim_period,Dim_month,Dim_latitude,Dim_longitude),dtype=float)
    var_v2_standing_pool = np.zeros((Dim_period,Dim_month,Dim_latitude,Dim_longitude),dtype=float)
    var_v2_transient_mean_pool = np.zeros((Dim_period,Dim_month,Dim_latitude),dtype=float)
    var_v2_stationary_mean_pool = np.zeros((Dim_period,Dim_month,Dim_latitude,Dim_longitude),dtype=float)
    # create space for overall momentum
    var_v2_overall_pool = np.zeros((Dim_period,Dim_month,Dim_latitude,Dim_longitude),dtype=float)
    # calculate mean meridional circulation
    var_v_steady_mean_zonal_mean = np.mean(v_temporal_mean,2)
    var_v_steady_mean_monthly_zonal_mean = np.zeros((12,Dim_latitude),dtype=float)

    for i in np.arange(Dim_month):
        var_v_steady_mean_monthly_zonal_mean[i,:] = np.mean(var_v_steady_mean_zonal_mean[month_day_index[i-1]:month_day_index[i-1]+month_day_length[i-1],:],0)
    var_v2_steady_mean = var_v_steady_mean_monthly_zonal_mean * var_v_steady_mean_monthly_zonal_mean

    return var_v2_transient_pool, var_v2_standing_pool, var_v2_transient_mean_pool,\
           var_v2_stationary_mean_pool, var_v2_overall_pool, var_v2_steady_mean, \

def compute_eddy(var_v_temporal_mean_select, var_v):
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
    # eddy
    var_v2_transient = var_v_prime * var_v_prime
    # monthly mean
    # shape[lat,lon]
    var_v2_transient_monthly_mean = np.mean(var_v2_transient,0)
    ####################################################
    # calculate transient mean eddies
    ############### transient mean eddy ################
    print ("Calculate transient mean eddies!")
    var_v_prime_zonal_mean = np.mean(var_v,2) - np.mean(var_v_temporal_mean_select,2)
    # eddy
    var_v2_transient_mean = var_v_prime_zonal_mean * var_v_prime_zonal_mean
    # monthly mean
    # shape[lat]
    var_v2_transient_mean_monthly_mean = np.mean(var_v2_transient_mean,0)
    ####################################################
    # Calculate standing eddies
    ################## standing eddy ###################
    print ("Calculate standing eddies!")
    var_v_star = np.zeros(var_v.shape,dtype=float)
    var_v_zonal_mean = np.mean(var_v,2)
    var_v_zonal_mean_enlarge = np.repeat(var_v_zonal_mean[:,:,np.newaxis],Dim_longitude,2)
    var_v_star = var_v - var_v_zonal_mean_enlarge
    # eddy
    var_v2_standing = var_v_star * var_v_star
    # monthly mean
    # shape[lat,lon]
    var_v2_standing_monthly_mean = np.mean(var_v2_standing,0)
    ####################################################
    # Calculate stationary mean eddies
    ##############  stationary mean eddy ###############
    print ("Calculate stationary mean eddies!")
    var_v_monthly_mean = np.mean(var_v,0)
    var_v_monthly_zonal_mean = np.mean(var_v_monthly_mean,1)
    var_v_monthly_zonal_mean_enlarge = np.repeat(var_v_monthly_zonal_mean[:,np.newaxis],Dim_longitude,1)
    var_v_star_monthly_zonal_mean = var_v_monthly_mean - var_v_monthly_zonal_mean_enlarge
    # monthly mean
    # shape[lat,lon]
    var_v2_stationary_mean_monthly_mean = var_v_star_monthly_zonal_mean * var_v_star_monthly_zonal_mean
    ####################################################
    # calculate the overall momentum transport
    ##############   overall transport   ###############
    print ("Calculate overall momentum transport!")
    var_v2_overall = var_v * var_v
    # monthly mean
    # shape[lat,lon]
    var_v2_overall_monthly_mean = np.mean(var_v2_overall,0)
    ####################################################

    return var_v2_transient_monthly_mean, var_v2_transient_mean_monthly_mean,\
           var_v2_standing_monthly_mean, var_v2_stationary_mean_monthly_mean,\
           var_v2_overall_monthly_mean


def create_netcdf_point_eddy(var_v2_overall,var_v2_transient,var_v2_transient_mean,
                             var_v2_standing, var_v2_stationary_mean, var_v2_steady_mean,
                             output_path):
    # take the zonal mean
    var_v2_overall_zonal = np.mean(var_v2_overall,3)
    var_v2_transient_zonal = np.mean(var_v2_transient,3)
    # transient_mean is zonal mean already
    var_v2_standing_zonal = np.mean(var_v2_standing,3)
    var_v2_stationary_mean_zonal = np.mean(var_v2_stationary_mean,3)
    # create netCDF
    print ('*******************************************************************')
    print ('*********************** create netcdf file*************************')
    print ('*******************************************************************')
    # wrap the datasets into netcdf file
    # 'NETCDF3_CLASSIC', 'NETCDF3_64BIT', 'NETCDF4_CLASSIC', and 'NETCDF4'
    data_wrap = Dataset(os.path.join(output_path,'model_era_daily_075_v2_eddies_{0}hPa_point.nc'.format(name_list[lev_slice])),
                        'w',format = 'NETCDF4')
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
    # create the 4-d variable
    var_v2_overall_wrap_var = data_wrap.createVariable('v2_overall',np.float64,('year','month','latitude','longitude'),zlib=True)
    var_v2_transient_wrap_var = data_wrap.createVariable('v2_transient',np.float64,('year','month','latitude','longitude'),zlib=True)
    var_v2_standing_wrap_var = data_wrap.createVariable('v2_standing',np.float64,('year','month','latitude','longitude'),zlib=True)
    var_v2_stationary_mean_wrap_var = data_wrap.createVariable('v2_stationary_mean',np.float64,('year','month','latitude','longitude'),zlib=True)
    # create the 4d variable
    var_v2_overall_zonal_wrap_var = data_wrap.createVariable('v2_overall_zonal',np.float64,('year','month','latitude'),zlib=True)
    var_v2_transient_zonal_wrap_var = data_wrap.createVariable('v2_transient_zonal',np.float64,('year','month','latitude'),zlib=True)
    var_v2_transient_mean_wrap_var = data_wrap.createVariable('v2_transient_mean',np.float64,('year','month','latitude'),zlib=True)
    var_v2_standing_zonal_wrap_var = data_wrap.createVariable('v2_standing_zonal',np.float64,('year','month','latitude'),zlib=True)
    var_v2_stationary_mean_zonal_wrap_var = data_wrap.createVariable('v2_stationary_mean_zonal',np.float64,('year','month','latitude'),zlib=True)
    # create the 2d variable
    var_v2_steady_mean_wrap_var = data_wrap.createVariable('v2_steady_mean',np.float64,('month','latitude'),zlib=True)
    # global attributes
    data_wrap.description = 'Monthly stationary and transient eddies at each grid point'
    # variable attributes
    lat_wrap_var.units = 'degree_north'
    lon_wrap_var.units = 'degree_east'
    
    var_v2_overall_wrap_var.units = 'm2/s2'
    var_v2_transient_wrap_var.units = 'm2/s2'
    var_v2_standing_wrap_var.units = 'm2/s2'
    var_v2_stationary_mean_wrap_var.units = 'm2/s2'
    var_v2_overall_zonal_wrap_var.units = 'm2/s2'
    var_v2_transient_zonal_wrap_var.units = 'm2/s2'
    var_v2_transient_mean_wrap_var.units = 'm2/s2'
    var_v2_standing_zonal_wrap_var.units = 'm2/s2'
    var_v2_stationary_mean_zonal_wrap_var.units = 'm2/s2'
    var_v2_steady_mean_wrap_var.units = 'm2/s2'

    lat_wrap_var.long_name = 'Latitude'
    lon_wrap_var.long_name = 'Longitude'
    
    var_v2_overall_wrap_var.long_name = 'Northward transport of kinetic energy by all motions'
    var_v2_transient_wrap_var.long_name = 'Northward transport of kinetic energy by transient eddy'
    var_v2_standing_wrap_var.long_name = 'Northward transport of kinetic energy by standing eddy'
    var_v2_stationary_mean_wrap_var.long_name = 'Northward transport of kinetic energy by stationary mean eddy'
    var_v2_overall_zonal_wrap_var.long_name = 'Zonal mean of northward transport of kinetic energy by all motions'
    var_v2_transient_zonal_wrap_var.long_name = 'Zonal mean of northward transport of kinetic energy by transient eddy'
    var_v2_transient_mean_wrap_var.long_name = 'Northward transport of kinetic energy by transient mean eddy'
    var_v2_standing_zonal_wrap_var.long_name = 'Zonal mean of northward transport of kinetic energy by standing eddy'
    var_v2_stationary_mean_zonal_wrap_var.long_name = 'Zonal mean of northward transport of kinetic energy by stationary mean eddy'
    var_v2_steady_mean_wrap_var.long_name = 'Northward transport of kinetic energy by steady mean meridional circulation'
    
    # writing data
    year_wrap_var[:] = period
    month_wrap_var[:] = index_month
    lat_wrap_var[:] = benchmark.variables['latitude'][:]
    lon_wrap_var[:] = benchmark.variables['longitude'][:]
    
    var_v2_overall_wrap_var[:] = var_v2_overall
    var_v2_transient_wrap_var[:] = var_v2_transient
    var_v2_standing_wrap_var[:] = var_v2_standing
    var_v2_stationary_mean_wrap_var[:] = var_v2_stationary_mean
    var_v2_overall_zonal_wrap_var[:] = var_v2_overall_zonal
    var_v2_transient_zonal_wrap_var[:] = var_v2_transient_zonal
    var_v2_transient_mean_wrap_var[:] = var_v2_transient_mean
    var_v2_standing_zonal_wrap_var[:] = var_v2_standing_zonal
    var_v2_stationary_mean_zonal_wrap_var[:] = var_v2_stationary_mean_zonal
    var_v2_steady_mean_wrap_var[:] = var_v2_steady_mean

    # close the file
    data_wrap.close()
    print ("The generation of netcdf files for fields on surface is complete!!")
    
if __name__=="__main__":
    # calculate the time for the code execution
    start_time = tttt.time()
    # initialization
    period, index_month, Dim_latitude, Dim_longitude, Dim_month, Dim_period,\
    month_day_length, month_day_index, v_temporal_sum  = initialization(benchmark)
    print ('*******************************************************************')
    print ('************  calculate the temporal and spatial mean  ************')
    print ('*******************************************************************')
    for i in period:
        for j in index_month:
            # get the key of each variable
            variable_key = var_key_retrieve(datapath,i,j)
            # take the daily mean of target fields
            var_v = pick_var(variable_key)
            # add daily field to the summation operator
            v_temporal_sum[month_day_index[j-1]:month_day_index[j-1]+month_day_length[j-1],:,:] = \
            v_temporal_sum[month_day_index[j-1]:month_day_index[j-1]+month_day_length[j-1],:,:] + var_v
    # calculate the temporal mean
    v_temporal_mean = v_temporal_sum / Dim_period
    print ('*******************************************************************')
    print ('**********  calculate the stationary and transient eddy  **********')
    print ('*******************************************************************')
    # Initialization
    # Grab temporal & spatial mean
    # The mean meridional circulation is calculated here
    var_v2_transient_pool, var_v2_standing_pool, var_v2_transient_mean_pool,\
    var_v2_stationary_mean_pool, var_v2_overall_pool, var_v2_steady_mean,\
    = initialization_eddy(v_temporal_mean)
    # start the loop for the computation of eddies
    for i in period:
        for j in index_month:
            # get the key of each variable
            variable_key = var_key_retrieve(datapath,i,j)
            # take the daily mean of target fields at certain levels
            var_v = pick_var(variable_key)
            # take the temporal mean for the certain month
            var_v_temporal_mean_select = v_temporal_mean[month_day_index[j-1]:month_day_index[j-1]+month_day_length[j-1],:,:]            # calculate the eddies
            var_v2_transient, var_v2_transient_mean, var_v2_standing, var_v2_stationary_mean, var_v2_overall\
            = compute_eddy(var_v_temporal_mean_select, var_v)
            # save output to the data pool for netCDF
            var_v2_overall_pool[i-start_year,j-1,:,:] = var_v2_overall
            var_v2_transient_pool[i-start_year,j-1,:,:] = var_v2_transient
            var_v2_transient_mean_pool[i-start_year,j-1,:] = var_v2_transient_mean
            var_v2_standing_pool[i-start_year,j-1,:,:] = var_v2_standing
            var_v2_stationary_mean_pool[i-start_year,j-1,:,:] = var_v2_stationary_mean
    create_netcdf_point_eddy(var_v2_overall_pool,var_v2_transient_pool,var_v2_transient_mean_pool,
                             var_v2_standing_pool,var_v2_stationary_mean_pool,var_v2_steady_mean,
                             output_path)
    print ('The full pipeline of the decomposition of meridional energy transport in the atmosphere is accomplished!')
    print ("--- %s minutes ---" % ((tttt.time() - start_time)/60))