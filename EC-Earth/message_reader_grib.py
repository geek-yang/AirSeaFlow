#!/usr/bin/env python
"""
Copyright Netherlands eScience Center
Function        : Message reader of grib file (output of ecearth)
Author          : Yang Liu
Date            : 2017.11.27
Last Update     : 2018.09.28
Description     : The code aims to read the message list of ec-earth. The message
                  list can be used for further post-processing then.
Return Value    : GRIB1 data file
Dependencies    : os, time, numpy, netCDF4, sys, matplotlib
Target Var      : Absolute Temperature              T         [K]
                  Specific Humidity                 q         [kg/kg]
                  Surface pressure                  ps        [Pa]
                  Zonal Divergent Wind              u         [m/s]
                  Meridional Divergent Wind         v         [m/s]
		          Geopotential Height 	            z         [gpm]
"""
import numpy as np
import time as tttt
from netCDF4 import Dataset,num2date
import os
import platform
import sys
import logging
import pygrib

# print the system structure and the path of the kernal
print platform.architecture()
print os.path

# calculate the time for the code execution
start_time = tttt.time()

####################################################################################
################################   Input zone  #####################################
datapath = '/projects/0/blueactn/ECEARTH/5DEH/output/ifs/001'
# specify output path for the txt file
output_path = '/home/lwc16308/ecearth_postproc/message_list/5DEH'
####################################################################################
grbs = pygrib.open(datapath + os.sep + 'ICMSH5DEH+197901')
namelist = open(output_path + os.sep + 'message_list_ICMSH5DEH+197901.txt','w+')
namelist.write('%s\n' % 'Message List of GRIB file')
# enumerate the message
for messenger in grbs:
    print messenger
    namelist.write('%s\n' % (messenger))
namelist.close()
grbs.close()
print ("--- %s minutes ---" % ((tttt.time() - start_time)/60))
