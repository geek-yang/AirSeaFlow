#!/usr/bin/env python
"""
Copyright Netherlands eScience Center
Function        : Quantify stationary and transient eddy from atmospheric meridional energy transport (JRA55)(Cartesius customised)
Author          : Yang Liu
Date            : 2018.11.30
Last Update     : 2019.10.04
Description     : The code aims to calculate the time and space dependent components
                  of atmospheric meridional energy transport based on atmospheric
                  reanalysis dataset JRA55 from JMA. The complete procedure
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
                  Surface Pressure                  sp
                  Zonal Divergent Wind              u
                  Meridional Divergent Wind         v
		          Surface geopotential  	        z
Caveat!!	    : This module is designed to work with a batch of files. Hence, there is
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
                  The pressure levels are from TOA to surface. The temporal resolution is 6 hourly.
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


##########################################################################
###########################   level information  #########################
level = np.array(([1, 2, 3, 5, 7,
                   10, 20, 30, 50, 70,
                   100, 125, 150, 175, 200,
                   225, 250, 300, 350, 400,
                   450, 500, 550, 600, 650,
                   700, 750, 775, 800, 825,
                   850, 875, 900, 925, 950,
                   975, 1000]),dtype=int)
native_level = len(level)

level_q = np.array(([100, 125, 150, 175, 200,
                     225, 250, 300, 350, 400,
                     450, 500, 550, 600, 650,
                     700, 750, 775, 800, 825,
                     850, 875, 900, 925, 950,
                     975, 1000]),dtype=int)
native_level_q = len(level_q)
##########################################################################

################################   Input zone  ######################################
# specify starting and ending time
start_year = 1979
end_year = 2017
# choose the slice number for the vertical layer
# pressure levels: (0)200, (1)300, (2)400, (3)500, (4)600, (5)750, (6)850, (7)950
lev_slice = 0
target_lev = [14, 17, 19, 21, 23, 26, 30, 34]
name_list = ['200', '300', '400', '500', '600', '750', '850', '950']
# specify data path
# ERAI 3D fields on pressure level
#datapath = '/home/ESLT0068/WorkFlow/Core_Database_AMET_OMET_reanalysis/ERAI/regression/pressure/daily'
datapath = '/project/0/blueactn/reanalysis/JRA55/subdaily/pressure'
# specify output path for figures
#output_path = '/home/ESLT0068/WorkFlow/Core_Database_AMET_OMET_reanalysis/ERAI/regression'
output_path = '/project/Reanalysis/ERA_Interim/Subdaily/Pressure/output'
# benchmark datasets for basic dimensions
benchmark_file = 'anl_p125.007_hgt.1980030100_1980033118'
benchmark = os.path.join(datapath, 'jra1980_p', benchmark_file)
####################################################################################

def var_retrieve(datapath, year, month, namelist_month, last_day):
    # get the path to each datasets
    print ("Start retrieving datasets {0} (y) {1} (m)".format(year,month))
    # get the keys of data
    key_tmp = pygrib.open(os.path.join(datapath, 'jra{0}_p'.format(year),
                          'anl_p125.011_tmp.{0}{1}0100_{2}{3}{4}18'.format(year, namelist_month[month-1], year, namelist_month[month-1], last_day)))
    key_spfh = pygrib.open(os.path.join(datapath, 'jra{0}_p'.format(year),
                           'anl_p125.051_spfh.{0}{1}0100_{2}{3}{4}18'.format(year, namelist_month[month-1], year, namelist_month[month-1], last_day)))
    key_vgrd = pygrib.open(os.path.join(datapath, 'jra{0}_p'.format(year),
                           'anl_p125.034_vgrd.{0}{1}0100_{2}{3}{4}18'.format(year, namelist_month[month-1], year, namelist_month[month-1], last_day)))
    key_hgt = pygrib.open(os.path.join(datapath, 'jra{0}_p'.format(year),
                          'anl_p125.007_hgt.{0}{1}0100_{2}{3}{4}18'.format(year, namelist_month[month-1], year, namelist_month[month-1], last_day)))
    print ("Retrieving datasets successfully and return the variable key!")
    # extract variables
    print ("Start extracting velocity for the calculation of mean over time and space.")
    # extract data at certain levels
    v = np.zeros((last_day*4, Dim_latitude, Dim_longitude),dtype=float)
    T = np.zeros((last_day*4, Dim_latitude, Dim_longitude),dtype=float)
    q = np.zeros((last_day*4, Dim_latitude, Dim_longitude),dtype=float)
    z = np.zeros((last_day*4, Dim_latitude, Dim_longitude),dtype=float)
    # extract data
    # reset counters
    counter_time = 0
    counter_message = 1 + target_lev[lev_slice]
    while (counter_message <= last_day*4*native_level):
        # take the key
        key_T = key_tmp.message(counter_message)
        key_v = key_vgrd.message(counter_message)
        key_z = key_hgt.message(counter_message)
        # take the values
        T[counter_time,:,:] = key_T.values
        v[counter_time,:,:] = key_v.values
        z[counter_time,:,:] = key_z.values
        # push the counter
        counter_message = counter_message + native_level
    # for q
    # reset counters
    counter_time = 0
    counter_message = 1 + target_lev[lev_slice] - 10 # q doesn't have first 10 layers
    while (counter_message <= last_day*4*native_level_q):
        # take the key
        key_q = key_spfh.message(counter_message)
        # take the values
        q[counter_time,counter_lev,:,:] = key_q.values
        # push the counter
        counter_message = counter_message + native_level_q
    # close all the grib files
    key_tmp.close()
    key_spfh.close()
    key_vgrd.close()
    key_hgt.close()
    # daily mean
    # first we reshape the array
    v_expand = v.reshape(len(time)//4,4,len(latitude),len(longitude))
    T_expand = T.reshape(len(time)//4,4,len(latitude),len(longitude))
    q_expand = q.reshape(len(time)//4,4,len(latitude),len(longitude))
    z_expand = z.reshape(len(time)//4,4,len(latitude),len(longitude))
    # Then we take daily mean
    v_daily = np.mean(v_expand,1)
    T_daily = np.mean(T_expand,1)
    q_daily = np.mean(q_expand,1)
    z_daily = np.mean(z_expand,1)
    if last_day == 29:
        v_out = v_daily[:-1,:,:]
        T_out = T_daily[:-1,:,:]
        q_out = q_daily[:-1,:,:]
        z_out = z_daily[:-1,:,:]
    else:
        v_out = v_daily
        T_out = T_daily
        q_out = q_daily
        z_out = z_daily
    print ('Extracting variables successfully!')

    return v_out, T_out, q_out, z_out

def initialization(benchmark):
    print ("Prepare for the main work!")
    # date and time arrangement
    # namelist of month and days for file manipulation
    namelist_month = ['01','02','03','04','05','06','07','08','09','10','11','12']
    long_month_list = np.array([1,3,5,7,8,10,12])
    leap_year_list = np.array([1976,1980,1984,1988,1992,1996,2000,2004,2008,2012,2016,2020])
    # index of months
    period = np.arange(start_year,end_year+1,1)
    index_month = np.arange(1,13,1)
    example_grbs = pygrib.open(benchmark)
    example_key = example_grbs.message(1)
    lats, lons = example_key.latlons()
    lat = lats[:,0] # descending
    lon = lons[0,:]
    # create dimensions for saving data
    Dim_year = len(period)
    Dim_month = len(index_month)
    Dim_latitude = len(lat)
    Dim_longitude = len(lon)
    #latitude = benchmark.variables['latitude'][:]
    #longitude = benchmark.variables['longitude'][:]
    #Dim_time = len(benchmark.variables['time'][:])
    # a list of the index of starting day in each month
    month_day_length = [31,28,31,30,31,30,31,31,30,31,30,31] #! we ignore the last day of February for the leap year
    month_day_index = [0,31,59,90,120,151,181,212,243,273,304,334]
    # create variables
    v_temporal_sum = np.zeros((365,Dim_latitude,Dim_longitude),dtype=float) #! we ignore the last day of February for the leap year
    T_temporal_sum = np.zeros((365,Dim_latitude,Dim_longitude),dtype=float)
    q_temporal_sum = np.zeros((365,Dim_latitude,Dim_longitude),dtype=float)
    z_temporal_sum = np.zeros((365,Dim_latitude,Dim_longitude),dtype=float)
    return period, index_month, namelist_month, long_month_list, leap_year_list, Dim_latitude,\
           Dim_longitude, Dim_month, Dim_period, month_day_length, month_day_index,\
           v_temporal_sum, T_temporal_sum, q_temporal_sum, z_temporal_sum

def initialization_eddy(v_temporal_mean, T_temporal_mean,
                        q_temporal_mean, z_temporal_mean):
    # Here we only use the temporal mean, for the spatial mean we will take it dynamically
    # during the calculation of eddies, for the sake of memory usage.
    # create space for eddies
    var_cpT_transient_pool = np.zeros((Dim_period,Dim_month,Dim_latitude,Dim_longitude),dtype=float)
    var_cpT_standing_pool = np.zeros((Dim_period,Dim_month,Dim_latitude,Dim_longitude),dtype=float)
    var_cpT_transient_mean_pool = np.zeros((Dim_period,Dim_month,Dim_latitude),dtype=float)
    var_cpT_stationary_mean_pool = np.zeros((Dim_period,Dim_month,Dim_latitude,Dim_longitude),dtype=float)
    var_Lvq_transient_pool = np.zeros((Dim_period,Dim_month,Dim_latitude,Dim_longitude),dtype=float)
    var_Lvq_standing_pool = np.zeros((Dim_period,Dim_month,Dim_latitude,Dim_longitude),dtype=float)
    var_Lvq_transient_mean_pool = np.zeros((Dim_period,Dim_month,Dim_latitude),dtype=float)
    var_Lvq_stationary_mean_pool = np.zeros((Dim_period,Dim_month,Dim_latitude,Dim_longitude),dtype=float)
    var_gz_transient_pool = np.zeros((Dim_period,Dim_month,Dim_latitude,Dim_longitude),dtype=float)
    var_gz_standing_pool = np.zeros((Dim_period,Dim_month,Dim_latitude,Dim_longitude),dtype=float)
    var_gz_transient_mean_pool = np.zeros((Dim_period,Dim_month,Dim_latitude),dtype=float)
    var_gz_stationary_mean_pool = np.zeros((Dim_period,Dim_month,Dim_latitude,Dim_longitude),dtype=float)
    # create space for overall momentum
    var_cpT_overall_pool = np.zeros((Dim_period,Dim_month,Dim_latitude,Dim_longitude),dtype=float)
    var_Lvq_overall_pool = np.zeros((Dim_period,Dim_month,Dim_latitude,Dim_longitude),dtype=float)
    var_gz_overall_pool = np.zeros((Dim_period,Dim_month,Dim_latitude,Dim_longitude),dtype=float)
    # calculate mean meridional circulation
    var_v_steady_mean_zonal_mean = np.mean(v_temporal_mean,2)
    var_v_steady_mean_monthly_zonal_mean = np.zeros((12,Dim_latitude),dtype=float)
    var_T_steady_mean_zonal_mean = np.mean(T_temporal_mean,2)
    var_T_steady_mean_monthly_zonal_mean = np.zeros((12,Dim_latitude),dtype=float)
    var_q_steady_mean_zonal_mean = np.mean(q_temporal_mean,2)
    var_q_steady_mean_monthly_zonal_mean = np.zeros((12,Dim_latitude),dtype=float)
    var_z_steady_mean_zonal_mean = np.mean(z_temporal_mean,2)
    var_z_steady_mean_monthly_zonal_mean = np.zeros((12,Dim_latitude),dtype=float)
    for i in np.arange(Dim_month):
        var_v_steady_mean_monthly_zonal_mean[i,:] = np.mean(var_v_steady_mean_zonal_mean[month_day_index[i-1]:month_day_index[i-1]+month_day_length[i-1],:],0)
        var_T_steady_mean_monthly_zonal_mean[i,:] = np.mean(var_T_steady_mean_zonal_mean[month_day_index[i-1]:month_day_index[i-1]+month_day_length[i-1],:],0)
        var_q_steady_mean_monthly_zonal_mean[i,:] = np.mean(var_q_steady_mean_zonal_mean[month_day_index[i-1]:month_day_index[i-1]+month_day_length[i-1],:],0)
        var_z_steady_mean_monthly_zonal_mean[i,:] = np.mean(var_z_steady_mean_zonal_mean[month_day_index[i-1]:month_day_index[i-1]+month_day_length[i-1],:],0)
    var_cpT_steady_mean = var_v_steady_mean_monthly_zonal_mean * var_T_steady_mean_monthly_zonal_mean
    var_Lvq_steady_mean = var_v_steady_mean_monthly_zonal_mean * var_q_steady_mean_monthly_zonal_mean
    var_gz_steady_mean = var_v_steady_mean_monthly_zonal_mean * var_z_steady_mean_monthly_zonal_mean

    return var_cpT_transient_pool, var_cpT_standing_pool, var_cpT_transient_mean_pool,\
           var_cpT_stationary_mean_pool, var_cpT_overall_pool, var_cpT_steady_mean, \
           var_Lvq_transient_pool, var_Lvq_standing_pool, var_Lvq_transient_mean_pool,\
           var_Lvq_stationary_mean_pool, var_Lvq_overall_pool, var_Lvq_steady_mean, \
           var_gz_transient_pool, var_gz_standing_pool, var_gz_transient_mean_pool,\
           var_gz_stationary_mean_pool, var_gz_overall_pool, var_gz_steady_mean

def compute_eddy(var_v_temporal_mean_select, var_T_temporal_mean_select,
                 var_q_temporal_mean_select, var_z_temporal_mean_select,
                 var_v, var_T, var_q, var_z):
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
    var_q_prime = var_q - var_q_temporal_mean_select
    var_z_prime = var_z - var_z_temporal_mean_select
    # eddy
    var_cpT_transient = var_v_prime * var_T_prime
    var_Lvq_transient = var_v_prime * var_q_prime
    var_gz_transient = var_v_prime * var_z_prime
    # monthly mean
    # shape[lat,lon]
    var_cpT_transient_monthly_mean = np.mean(var_cpT_transient,0)
    var_Lvq_transient_monthly_mean = np.mean(var_Lvq_transient,0)
    var_gz_transient_monthly_mean = np.mean(var_gz_transient,0)
    ####################################################
    # calculate transient mean eddies
    ############### transient mean eddy ################
    print ("Calculate transient mean eddies!")
    var_v_prime_zonal_mean = np.mean(var_v,2) - np.mean(var_v_temporal_mean_select,2)
    var_T_prime_zonal_mean = np.mean(var_T,2) - np.mean(var_T_temporal_mean_select,2)
    var_q_prime_zonal_mean = np.mean(var_q,2) - np.mean(var_q_temporal_mean_select,2)
    var_z_prime_zonal_mean = np.mean(var_z,2) - np.mean(var_z_temporal_mean_select,2)
    # eddy
    var_cpT_transient_mean = var_v_prime_zonal_mean * var_T_prime_zonal_mean
    var_Lvq_transient_mean = var_v_prime_zonal_mean * var_q_prime_zonal_mean
    var_gz_transient_mean = var_v_prime_zonal_mean * var_z_prime_zonal_mean
    # monthly mean
    # shape[lat]
    var_cpT_transient_mean_monthly_mean = np.mean(var_cpT_transient_mean,0)
    var_Lvq_transient_mean_monthly_mean = np.mean(var_Lvq_transient_mean,0)
    var_gz_transient_mean_monthly_mean = np.mean(var_gz_transient_mean,0)
    ####################################################
    # Calculate standing eddies
    ################## standing eddy ###################
    print ("Calculate standing eddies!")
    var_v_star = np.zeros(var_v.shape,dtype=float)
    var_v_zonal_mean = np.mean(var_v,2)
    var_v_zonal_mean_enlarge = np.repeat(var_v_zonal_mean[:,:,np.newaxis],Dim_longitude,2)
    var_v_star = var_v - var_v_zonal_mean_enlarge
    var_T_star = np.zeros(var_T.shape,dtype=float)
    var_T_zonal_mean = np.mean(var_T,2)
    var_T_zonal_mean_enlarge = np.repeat(var_T_zonal_mean[:,:,np.newaxis],Dim_longitude,2)
    var_T_star = var_T - var_T_zonal_mean_enlarge
    var_q_star = np.zeros(var_q.shape,dtype=float)
    var_q_zonal_mean = np.mean(var_q,2)
    var_q_zonal_mean_enlarge = np.repeat(var_q_zonal_mean[:,:,np.newaxis],Dim_longitude,2)
    var_q_star = var_q - var_q_zonal_mean_enlarge
    var_z_star = np.zeros(var_z.shape,dtype=float)
    var_z_zonal_mean = np.mean(var_z,2)
    var_z_zonal_mean_enlarge = np.repeat(var_z_zonal_mean[:,:,np.newaxis],Dim_longitude,2)
    var_z_star = var_z - var_z_zonal_mean_enlarge
    # eddy
    var_cpT_standing = var_v_star * var_T_star
    var_Lvq_standing = var_v_star * var_q_star
    var_gz_standing = var_v_star * var_z_star
    # monthly mean
    # shape[lat,lon]
    var_cpT_standing_monthly_mean = np.mean(var_cpT_standing,0)
    var_Lvq_standing_monthly_mean = np.mean(var_Lvq_standing,0)
    var_gz_standing_monthly_mean = np.mean(var_gz_standing,0)
    ####################################################
    # Calculate stationary mean eddies
    ##############  stationary mean eddy ###############
    print ("Calculate stationary mean eddies!")
    var_v_monthly_mean = np.mean(var_v,0)
    var_v_monthly_zonal_mean = np.mean(var_v_monthly_mean,1)
    var_v_monthly_zonal_mean_enlarge = np.repeat(var_v_monthly_zonal_mean[:,np.newaxis],Dim_longitude,1)
    var_v_star_monthly_zonal_mean = var_v_monthly_mean - var_v_monthly_zonal_mean_enlarge
    var_T_monthly_mean = np.mean(var_T,0)
    var_T_monthly_zonal_mean = np.mean(var_T_monthly_mean,1)
    var_T_monthly_zonal_mean_enlarge = np.repeat(var_T_monthly_zonal_mean[:,np.newaxis],Dim_longitude,1)
    var_T_star_monthly_zonal_mean = var_T_monthly_mean - var_T_monthly_zonal_mean_enlarge
    var_q_monthly_mean = np.mean(var_q,0)
    var_q_monthly_zonal_mean = np.mean(var_q_monthly_mean,1)
    var_q_monthly_zonal_mean_enlarge = np.repeat(var_q_monthly_zonal_mean[:,np.newaxis],Dim_longitude,1)
    var_q_star_monthly_zonal_mean = var_q_monthly_mean - var_q_monthly_zonal_mean_enlarge
    var_z_monthly_mean = np.mean(var_z,0)
    var_z_monthly_zonal_mean = np.mean(var_z_monthly_mean,1)
    var_z_monthly_zonal_mean_enlarge = np.repeat(var_z_monthly_zonal_mean[:,np.newaxis],Dim_longitude,1)
    var_z_star_monthly_zonal_mean = var_z_monthly_mean - var_z_monthly_zonal_mean_enlarge
    # monthly mean
    # shape[lat,lon]
    var_cpT_stationary_mean_monthly_mean = var_v_star_monthly_zonal_mean * var_T_star_monthly_zonal_mean
    var_Lvq_stationary_mean_monthly_mean = var_v_star_monthly_zonal_mean * var_q_star_monthly_zonal_mean
    var_gz_stationary_mean_monthly_mean = var_v_star_monthly_zonal_mean * var_z_star_monthly_zonal_mean
    ####################################################
    # calculate the overall momentum transport
    ##############   overall transport   ###############
    print ("Calculate overall momentum transport!")
    var_cpT_overall = var_v * var_T
    var_Lvq_overall = var_v * var_q
    var_gz_overall = var_v * var_z
    # monthly mean
    # shape[lat,lon]
    var_cpT_overall_monthly_mean = np.mean(var_cpT_overall,0)
    var_Lvq_overall_monthly_mean = np.mean(var_Lvq_overall,0)
    var_gz_overall_monthly_mean = np.mean(var_gz_overall,0)
    ####################################################

    return var_cpT_transient_monthly_mean, var_cpT_transient_mean_monthly_mean,\
           var_cpT_standing_monthly_mean, var_cpT_stationary_mean_monthly_mean,\
           var_cpT_overall_monthly_mean, var_Lvq_transient_monthly_mean,\
           var_Lvq_transient_mean_monthly_mean, var_Lvq_standing_monthly_mean,\
           var_Lvq_stationary_mean_monthly_mean, var_Lvq_overall_monthly_mean,\
           var_gz_transient_monthly_mean, var_gz_transient_mean_monthly_mean,\
           var_gz_standing_monthly_mean, var_gz_stationary_mean_monthly_mean,\
           var_gz_overall_monthly_mean

def create_netcdf_point_eddy(var_cpT_overall,var_cpT_transient,var_cpT_transient_mean,
                             var_cpT_standing, var_cpT_stationary_mean, var_cpT_steady_mean,
                             var_Lvq_overall,var_Lvq_transient,var_Lvq_transient_mean,
                             var_Lvq_standing, var_Lvq_stationary_mean, var_Lvq_steady_mean,
                             var_gz_overall,var_gz_transient,var_gz_transient_mean,
                             var_gz_standing, var_gz_stationary_mean, var_gz_steady_mean,
                             output_path):
    # take the zonal mean
    var_cpT_overall_zonal = np.mean(var_cpT_overall,3)
    var_cpT_transient_zonal = np.mean(var_cpT_transient,3)
    var_Lvq_overall_zonal = np.mean(var_Lvq_overall,3)
    var_Lvq_transient_zonal = np.mean(var_Lvq_transient,3)
    var_gz_overall_zonal = np.mean(var_gz_overall,3)
    var_gz_transient_zonal = np.mean(var_gz_transient,3)
    # transient_mean is zonal mean already
    var_cpT_standing_zonal = np.mean(var_cpT_standing,3)
    var_cpT_stationary_mean_zonal = np.mean(var_cpT_stationary_mean,3)
    var_Lvq_standing_zonal = np.mean(var_Lvq_standing,3)
    var_Lvq_stationary_mean_zonal = np.mean(var_Lvq_stationary_mean,3)
    var_gz_standing_zonal = np.mean(var_gz_standing,3)
    var_gz_stationary_mean_zonal = np.mean(var_gz_stationary_mean,3)
    # create netCDF
    print ('*******************************************************************')
    print ('*********************** create netcdf file*************************')
    print ('*******************************************************************')
    # wrap the datasets into netcdf file
    # 'NETCDF3_CLASSIC', 'NETCDF3_64BIT', 'NETCDF4_CLASSIC', and 'NETCDF4'
    data_wrap = Dataset(os.path.join(output_path,'model_jra_daily_075_E_eddies_{0}hPa_point.nc'.format(name_list[lev_slice])),
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
    var_cpT_overall_wrap_var = data_wrap.createVariable('cpT_overall',np.float64,('year','month','latitude','longitude'),zlib=True)
    var_cpT_transient_wrap_var = data_wrap.createVariable('cpT_transient',np.float64,('year','month','latitude','longitude'),zlib=True)
    var_cpT_standing_wrap_var = data_wrap.createVariable('cpT_standing',np.float64,('year','month','latitude','longitude'),zlib=True)
    var_cpT_stationary_mean_wrap_var = data_wrap.createVariable('cpT_stationary_mean',np.float64,('year','month','latitude','longitude'),zlib=True)
    var_Lvq_overall_wrap_var = data_wrap.createVariable('Lvq_overall',np.float64,('year','month','latitude','longitude'),zlib=True)
    var_Lvq_transient_wrap_var = data_wrap.createVariable('Lvq_transient',np.float64,('year','month','latitude','longitude'),zlib=True)
    var_Lvq_standing_wrap_var = data_wrap.createVariable('Lvq_standing',np.float64,('year','month','latitude','longitude'),zlib=True)
    var_Lvq_stationary_mean_wrap_var = data_wrap.createVariable('Lvq_stationary_mean',np.float64,('year','month','latitude','longitude'),zlib=True)
    var_gz_overall_wrap_var = data_wrap.createVariable('gz_overall',np.float64,('year','month','latitude','longitude'),zlib=True)
    var_gz_transient_wrap_var = data_wrap.createVariable('gz_transient',np.float64,('year','month','latitude','longitude'),zlib=True)
    var_gz_standing_wrap_var = data_wrap.createVariable('gz_standing',np.float64,('year','month','latitude','longitude'),zlib=True)
    var_gz_stationary_mean_wrap_var = data_wrap.createVariable('gz_stationary_mean',np.float64,('year','month','latitude','longitude'),zlib=True)
    # create the 4d variable
    var_cpT_overall_zonal_wrap_var = data_wrap.createVariable('cpT_overall_zonal',np.float64,('year','month','latitude'),zlib=True)
    var_cpT_transient_zonal_wrap_var = data_wrap.createVariable('cpT_transient_zonal',np.float64,('year','month','latitude'),zlib=True)
    var_cpT_transient_mean_wrap_var = data_wrap.createVariable('cpT_transient_mean',np.float64,('year','month','latitude'),zlib=True)
    var_cpT_standing_zonal_wrap_var = data_wrap.createVariable('cpT_standing_zonal',np.float64,('year','month','latitude'),zlib=True)
    var_cpT_stationary_mean_zonal_wrap_var = data_wrap.createVariable('cpT_stationary_mean_zonal',np.float64,('year','month','latitude'),zlib=True)
    var_Lvq_overall_zonal_wrap_var = data_wrap.createVariable('Lvq_overall_zonal',np.float64,('year','month','latitude'),zlib=True)
    var_Lvq_transient_zonal_wrap_var = data_wrap.createVariable('Lvq_transient_zonal',np.float64,('year','month','latitude'),zlib=True)
    var_Lvq_transient_mean_wrap_var = data_wrap.createVariable('Lvq_transient_mean',np.float64,('year','month','latitude'),zlib=True)
    var_Lvq_standing_zonal_wrap_var = data_wrap.createVariable('Lvq_standing_zonal',np.float64,('year','month','latitude'),zlib=True)
    var_Lvq_stationary_mean_zonal_wrap_var = data_wrap.createVariable('Lvq_stationary_mean_zonal',np.float64,('year','month','latitude'),zlib=True)
    var_gz_overall_zonal_wrap_var = data_wrap.createVariable('gz_overall_zonal',np.float64,('year','month','latitude'),zlib=True)
    var_gz_transient_zonal_wrap_var = data_wrap.createVariable('gz_transient_zonal',np.float64,('year','month','latitude'),zlib=True)
    var_gz_transient_mean_wrap_var = data_wrap.createVariable('gz_transient_mean',np.float64,('year','month','latitude'),zlib=True)
    var_gz_standing_zonal_wrap_var = data_wrap.createVariable('gz_standing_zonal',np.float64,('year','month','latitude'),zlib=True)
    var_gz_stationary_mean_zonal_wrap_var = data_wrap.createVariable('gz_stationary_mean_zonal',np.float64,('year','month','latitude'),zlib=True)
    # create the 2d variable
    var_cpT_steady_mean_wrap_var = data_wrap.createVariable('cpT_steady_mean',np.float64,('month','latitude'),zlib=True)
    var_Lvq_steady_mean_wrap_var = data_wrap.createVariable('Lvq_steady_mean',np.float64,('month','latitude'),zlib=True)
    var_gz_steady_mean_wrap_var = data_wrap.createVariable('gz_steady_mean',np.float64,('month','latitude'),zlib=True)
    # global attributes
    data_wrap.description = 'Monthly stationary and transient eddies at each grid point'
    # variable attributes
    lat_wrap_var.units = 'degree_north'
    lon_wrap_var.units = 'degree_east'

    var_cpT_overall_wrap_var.units = 'K m/s'
    var_cpT_transient_wrap_var.units = 'K m/s'
    var_cpT_standing_wrap_var.units = 'K m/s'
    var_cpT_stationary_mean_wrap_var.units = 'K m/s'
    var_cpT_overall_zonal_wrap_var.units = 'K m/s'
    var_cpT_transient_zonal_wrap_var.units = 'K m/s'
    var_cpT_transient_mean_wrap_var.units = 'K m/s'
    var_cpT_standing_zonal_wrap_var.units = 'K m/s'
    var_cpT_stationary_mean_zonal_wrap_var.units = 'K m/s'
    var_cpT_steady_mean_wrap_var.units = 'K m/s'

    var_Lvq_overall_wrap_var.units = 'Kg m/kg s'
    var_Lvq_transient_wrap_var.units = 'Kg m/kg s'
    var_Lvq_standing_wrap_var.units = 'Kg m/kg s'
    var_Lvq_stationary_mean_wrap_var.units = 'Kg m/kg s'
    var_Lvq_overall_zonal_wrap_var.units = 'Kg m/kg s'
    var_Lvq_transient_zonal_wrap_var.units = 'Kg m/kg s'
    var_Lvq_transient_mean_wrap_var.units = 'Kg m/kg s'
    var_Lvq_standing_zonal_wrap_var.units = 'Kg m/kg s'
    var_Lvq_stationary_mean_zonal_wrap_var.units = 'Kg m/kg s'
    var_Lvq_steady_mean_wrap_var.units = 'Kg m/kg s'

    var_gz_overall_wrap_var.units = 'm3/s3'
    var_gz_transient_wrap_var.units = 'm3/s3'
    var_gz_standing_wrap_var.units = 'm3/s3'
    var_gz_stationary_mean_wrap_var.units = 'm3/s3'
    var_gz_overall_zonal_wrap_var.units = 'm3/s3'
    var_gz_transient_zonal_wrap_var.units = 'm3/s3'
    var_gz_transient_mean_wrap_var.units = 'm3/s3'
    var_gz_standing_zonal_wrap_var.units = 'm3/s3'
    var_gz_stationary_mean_zonal_wrap_var.units = 'm3/s3'
    var_gz_steady_mean_wrap_var.units = 'm3/s3'

    lat_wrap_var.long_name = 'Latitude'
    lon_wrap_var.long_name = 'Longitude'

    var_cpT_overall_wrap_var.long_name = 'Northward transport of temperature by all motions'
    var_cpT_transient_wrap_var.long_name = 'Northward transport of temperature by transient eddy'
    var_cpT_standing_wrap_var.long_name = 'Northward transport of temperature by standing eddy'
    var_cpT_stationary_mean_wrap_var.long_name = 'Northward transport of temperature by stationary mean eddy'
    var_cpT_overall_zonal_wrap_var.long_name = 'Zonal mean of northward transport of temperature by all motions'
    var_cpT_transient_zonal_wrap_var.long_name = 'Zonal mean of northward transport of temperature by transient eddy'
    var_cpT_transient_mean_wrap_var.long_name = 'Northward transport of temperature by transient mean eddy'
    var_cpT_standing_zonal_wrap_var.long_name = 'Zonal mean of northward transport of temperature by standing eddy'
    var_cpT_stationary_mean_zonal_wrap_var.long_name = 'Zonal mean of northward transport of temperature by stationary mean eddy'
    var_cpT_steady_mean_wrap_var.long_name = 'Northward transport of temperature by steady mean meridional circulation'

    var_Lvq_overall_wrap_var.long_name = 'Northward transport of latent heat by all motions'
    var_Lvq_transient_wrap_var.long_name = 'Northward transport of latent heat by transient eddy'
    var_Lvq_standing_wrap_var.long_name = 'Northward transport of latent heat by standing eddy'
    var_Lvq_stationary_mean_wrap_var.long_name = 'Northward transport of latent heat by stationary mean eddy'
    var_Lvq_overall_zonal_wrap_var.long_name = 'Zonal mean of northward transport of latent heat by all motions'
    var_Lvq_transient_zonal_wrap_var.long_name = 'Zonal mean of northward transport of latent heat by transient eddy'
    var_Lvq_transient_mean_wrap_var.long_name = 'Northward transport of latent heat by transient mean eddy'
    var_Lvq_standing_zonal_wrap_var.long_name = 'Zonal mean of northward transport of latent heat by standing eddy'
    var_Lvq_stationary_mean_zonal_wrap_var.long_name = 'Zonal mean of northward transport of latent heat by stationary mean eddy'
    var_Lvq_steady_mean_wrap_var.long_name = 'Northward transport of latent heat by steady mean meridional circulation'

    var_gz_overall_wrap_var.long_name = 'Northward transport of geopotential by all motions'
    var_gz_transient_wrap_var.long_name = 'Northward transport of geopotential by transient eddy'
    var_gz_standing_wrap_var.long_name = 'Northward transport of geopotential by standing eddy'
    var_gz_stationary_mean_wrap_var.long_name = 'Northward transport of geopotential by stationary mean eddy'
    var_gz_overall_zonal_wrap_var.long_name = 'Zonal mean of northward transport of geopotential by all motions'
    var_gz_transient_zonal_wrap_var.long_name = 'Zonal mean of northward transport of geopotential by transient eddy'
    var_gz_transient_mean_wrap_var.long_name = 'Northward transport of geopotential by transient mean eddy'
    var_gz_standing_zonal_wrap_var.long_name = 'Zonal mean of northward transport of geopotential by standing eddy'
    var_gz_stationary_mean_zonal_wrap_var.long_name = 'Zonal mean of northward transport of geopotential by stationary mean eddy'
    var_gz_steady_mean_wrap_var.long_name = 'Northward transport of geopotential by steady mean meridional circulation'


    # writing data
    year_wrap_var[:] = period
    month_wrap_var[:] = index_month
    lat_wrap_var[:] = benchmark.variables['latitude'][:]
    lon_wrap_var[:] = benchmark.variables['longitude'][:]

    var_cpT_overall_wrap_var[:] = var_cpT_overall
    var_cpT_transient_wrap_var[:] = var_cpT_transient
    var_cpT_standing_wrap_var[:] = var_cpT_standing
    var_cpT_stationary_mean_wrap_var[:] = var_cpT_stationary_mean
    var_cpT_overall_zonal_wrap_var[:] = var_cpT_overall_zonal
    var_cpT_transient_zonal_wrap_var[:] = var_cpT_transient_zonal
    var_cpT_transient_mean_wrap_var[:] = var_cpT_transient_mean
    var_cpT_standing_zonal_wrap_var[:] = var_cpT_standing_zonal
    var_cpT_stationary_mean_zonal_wrap_var[:] = var_cpT_stationary_mean_zonal
    var_cpT_steady_mean_wrap_var[:] = var_cpT_steady_mean

    var_Lvq_overall_wrap_var[:] = var_Lvq_overall
    var_Lvq_transient_wrap_var[:] = var_Lvq_transient
    var_Lvq_standing_wrap_var[:] = var_Lvq_standing
    var_Lvq_stationary_mean_wrap_var[:] = var_Lvq_stationary_mean
    var_Lvq_overall_zonal_wrap_var[:] = var_Lvq_overall_zonal
    var_Lvq_transient_zonal_wrap_var[:] = var_Lvq_transient_zonal
    var_Lvq_transient_mean_wrap_var[:] = var_Lvq_transient_mean
    var_Lvq_standing_zonal_wrap_var[:] = var_Lvq_standing_zonal
    var_Lvq_stationary_mean_zonal_wrap_var[:] = var_Lvq_stationary_mean_zonal
    var_Lvq_steady_mean_wrap_var[:] = var_Lvq_steady_mean

    var_gz_overall_wrap_var[:] = var_gz_overall
    var_gz_transient_wrap_var[:] = var_gz_transient
    var_gz_standing_wrap_var[:] = var_gz_standing
    var_gz_stationary_mean_wrap_var[:] = var_gz_stationary_mean
    var_gz_overall_zonal_wrap_var[:] = var_gz_overall_zonal
    var_gz_transient_zonal_wrap_var[:] = var_gz_transient_zonal
    var_gz_transient_mean_wrap_var[:] = var_gz_transient_mean
    var_gz_standing_zonal_wrap_var[:] = var_gz_standing_zonal
    var_gz_stationary_mean_zonal_wrap_var[:] = var_gz_stationary_mean_zonal
    var_gz_steady_mean_wrap_var[:] = var_gz_steady_mean

    # close the file
    data_wrap.close()
    print ("The generation of netcdf files for fields on surface is complete!!")

if __name__=="__main__":
    # calculate the time for the code execution
    start_time = tttt.time()
    # initialization
    period, index_month, namelist_month, long_month_list, leap_year_list, Dim_latitude,
    Dim_longitude, Dim_month, Dim_period, month_day_length, month_day_index, v_temporal_sum,
    T_temporal_sum, q_temporal_sum, z_temporal_sum  = initialization(benchmark)
    print ('*******************************************************************')
    print ('************  calculate the temporal and spatial mean  ************')
    print ('*******************************************************************')
    for i in period:
        for j in index_month:
            # determine how many days are there in a month
            if j in long_month_list:
                last_day = 31
            elif j == 2:
                if i in leap_year_list:
                    last_day = 29
                else:
                    last_day = 28
            else:
                last_day = 30
            # get the key of each variable and extract fields
            var_v, var_T, var_q, var_z = var_retrieve(datapath, i, j, last_day)
            # add daily field to the summation operator
            v_temporal_sum[month_day_index[j-1]:month_day_index[j-1]+month_day_length[j-1],:,:] = \
            v_temporal_sum[month_day_index[j-1]:month_day_index[j-1]+month_day_length[j-1],:,:] + var_v
            T_temporal_sum[month_day_index[j-1]:month_day_index[j-1]+month_day_length[j-1],:,:] = \
            T_temporal_sum[month_day_index[j-1]:month_day_index[j-1]+month_day_length[j-1],:,:] + var_T
            q_temporal_sum[month_day_index[j-1]:month_day_index[j-1]+month_day_length[j-1],:,:] = \
            q_temporal_sum[month_day_index[j-1]:month_day_index[j-1]+month_day_length[j-1],:,:] + var_q
            z_temporal_sum[month_day_index[j-1]:month_day_index[j-1]+month_day_length[j-1],:,:] = \
            z_temporal_sum[month_day_index[j-1]:month_day_index[j-1]+month_day_length[j-1],:,:] + var_z
    # calculate the temporal mean
    v_temporal_mean = v_temporal_sum / Dim_period
    T_temporal_mean = T_temporal_sum / Dim_period
    q_temporal_mean = q_temporal_sum / Dim_period
    z_temporal_mean = z_temporal_sum / Dim_period
    print ('*******************************************************************')
    print ('**********  calculate the stationary and transient eddy  **********')
    print ('*******************************************************************')
    # Initialization
    # Grab temporal & spatial mean
    # The mean meridional circulation is calculated here
    var_cpT_transient_pool, var_cpT_standing_pool, var_cpT_transient_mean_pool,\
    var_cpT_stationary_mean_pool, var_cpT_overall_pool, var_cpT_steady_mean,\
    var_Lvq_transient_pool, var_Lvq_standing_pool, var_Lvq_transient_mean_pool,\
    var_Lvq_stationary_mean_pool, var_Lvq_overall_pool, var_Lvq_steady_mean,\
    var_gz_transient_pool, var_gz_standing_pool, var_gz_transient_mean_pool,\
    var_gz_stationary_mean_pool, var_gz_overall_pool, var_gz_steady_mean,\
    = initialization_eddy(v_temporal_mean, T_temporal_mean, q_temporal_mean,
                          z_temporal_mean)
    # start the loop for the computation of eddies
    for i in period:
        for j in index_month:
            # determine how many days are there in a month
            if j in long_month_list:
                last_day = 31
            elif j == 2:
                if i in leap_year_list:
                    last_day = 29
                else:
                    last_day = 28
            else:
                last_day = 30
            # get the key of each variable and extract variables
            var_v, var_T, var_q, var_z = var_retrieve(datapath, i, j, last_day)
            # take the temporal mean for the certain month
            var_v_temporal_mean_select = v_temporal_mean[month_day_index[j-1]:month_day_index[j-1]+month_day_length[j-1],:,:]
            var_T_temporal_mean_select = T_temporal_mean[month_day_index[j-1]:month_day_index[j-1]+month_day_length[j-1],:,:]
            var_q_temporal_mean_select = q_temporal_mean[month_day_index[j-1]:month_day_index[j-1]+month_day_length[j-1],:,:]
            var_z_temporal_mean_select = z_temporal_mean[month_day_index[j-1]:month_day_index[j-1]+month_day_length[j-1],:,:]
            # calculate the eddies
            var_cpT_transient, var_cpT_transient_mean, var_cpT_standing, var_cpT_stationary_mean,\
            var_cpT_overall, var_Lvq_transient, var_Lvq_transient_mean, var_Lvq_standing, \
            var_Lvq_stationary_mean, var_Lvq_overall, var_gz_transient, var_gz_transient_mean,\
            var_gz_standing, var_gz_stationary_mean, var_gz_overall \
            = compute_eddy(var_v_temporal_mean_select, var_T_temporal_mean_select,
                           var_q_temporal_mean_select, var_z_temporal_mean_select,
                           var_v, var_T, var_q, var_z)
            # save output to the data pool for netCDF
            var_cpT_overall_pool[i-start_year,j-1,:,:] = var_cpT_overall
            var_cpT_transient_pool[i-start_year,j-1,:,:] = var_cpT_transient
            var_cpT_transient_mean_pool[i-start_year,j-1,:] = var_cpT_transient_mean
            var_cpT_standing_pool[i-start_year,j-1,:,:] = var_cpT_standing
            var_cpT_stationary_mean_pool[i-start_year,j-1,:,:] = var_cpT_stationary_mean
            var_Lvq_overall_pool[i-start_year,j-1,:,:] = var_Lvq_overall
            var_Lvq_transient_pool[i-start_year,j-1,:,:] = var_Lvq_transient
            var_Lvq_transient_mean_pool[i-start_year,j-1,:] = var_Lvq_transient_mean
            var_Lvq_standing_pool[i-start_year,j-1,:,:] = var_Lvq_standing
            var_Lvq_stationary_mean_pool[i-start_year,j-1,:,:] = var_Lvq_stationary_mean
            var_gz_overall_pool[i-start_year,j-1,:,:] = var_gz_overall
            var_gz_transient_pool[i-start_year,j-1,:,:] = var_gz_transient
            var_gz_transient_mean_pool[i-start_year,j-1,:] = var_gz_transient_mean
            var_gz_standing_pool[i-start_year,j-1,:,:] = var_gz_standing
            var_gz_stationary_mean_pool[i-start_year,j-1,:,:] = var_gz_stationary_mean
    create_netcdf_point_eddy(var_cpT_overall_pool,var_cpT_transient_pool,var_cpT_transient_mean_pool,
                             var_cpT_standing_pool,var_cpT_stationary_mean_pool,var_cpT_steady_mean,
                             var_Lvq_overall_pool,var_Lvq_transient_pool,var_Lvq_transient_mean_pool,
                             var_Lvq_standing_pool,var_Lvq_stationary_mean_pool,var_Lvq_steady_mean,
                             var_gz_overall_pool,var_gz_transient_pool,var_gz_transient_mean_pool,
                             var_gz_standing_pool,var_gz_stationary_mean_pool,var_gz_steady_mean,
                             output_path)
    print ('The full pipeline of the decomposition of meridional energy transport in the atmosphere is accomplished!')
    print ("--- %s minutes ---" % ((tttt.time() - start_time)/60))