#!~/.pyenv/shims/python
"""
Copyright Netherlands eScience Center
Function        : Quantify atmospheric meridional energy transport from EC-earth (Cartesius)
Author          : Yang Liu
Date            : 2017.12.07
Last Update     : 2018.09.28
Description     : The code aims to calculate the atmospheric meridional energy
                  transport based on the output from EC-Earth simulation.
                  The complete procedure includes the calculation of the mass budget
                  correction and the computation of vertical integral of zonally
                  integrated meridional energy transport.

Return Value    : GRIB1 data file
Dependencies    : os, time, numpy, netCDF4, sys, matplotlib, pygrib
variables       : Absolute Temperature              T         [K]
                  Specific Humidity                 q         [kg/kg]
                  Surface pressure                  sp        [Pa]
                  Zonal Divergent Wind              u         [m/s]
                  Meridional Divergent Wind         v         [m/s]
		          Geopotential 	                    gz        [m2/s2]
Caveat!!	:     The dataset is for the entire globe from -90N - 90N.
                  The model uses TL511 spectral resolution with N256 Gaussian Grid.
                  For postprocessing, the spectral fields will be converted to geographical grid.
                  The spatial resolution of Gaussian grid is 512 (lat) x 1024 (lon).
                  The simulation starts from 00:00:00 01-01-1979.
                  The time step in the dataset is 3 hours.
                  00:00 03:00 06:00 09:00 12:00 15:00 18:00 21:00
                  The dataset has 91 hybrid model levels. Starting from level 1 (TOA) to 91 (Surface).
                  Data is saved on reduced gaussian grid with the size of 512 (lat) x 1024(lon)

                  Attention should be paid when calculating the meridional grid length (dy)!
                  Direction of Axis:
                  Model Level: TOA to surface (1 to 91)
                  Latitude: South to Nouth (90 to -90)
                  Lontitude: West to East (0 to 360)

                  Mass correction is accmpolished through the correction of barotropic wind:
                  mass residual = surface pressure tendency + divergence of mass flux (u,v) - (E-P)
                  E-P = evaporation - precipitation = moisture tendency - divergence of moisture flux(u,v)
                  Due to the structure of the dataset, the mass budget correction are split into
                  two parts: 1. Quantify tendency terms in month loop
                             2. Quantify divergence terms in day loop
                  For more information about this method, please refer to Trenberth, 1991.
"""
import numpy as np
import time as tttt
from netCDF4 import Dataset,num2date
import os
import platform
import sys
import logging
import matplotlib
# generate images without having a window appear
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import iris
import pygrib
import errno
import subprocess
import shutil
import glob
import tempfile

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

class postprocess:
    def __init__(self, rundir, postprocess, archive, expname, leg):
        '''
        Quantify atmospheric meridional energy transport from EC-earth

        param rundir: base run directory
        param postprocess: output of postprocess netCDF data
        param archive: archive location of full grib files
        param expname: name of experiment
        param leg: leg number in experiment
        type rundir: str
        type postprocess: str, NoneType
        type archive: str, NoneType
        type expname: str
        type leg: int
        '''
        # check if output data is available
        ece_info = os.path.join(rundir, expname, 'ece.info')
        if "leg_number={}".format(leg) in open(ece_info).read():
            # format leg as str with leading zeros
            leg = str(leg).zfill(3) 
            # define the path where the output data of the experiment is saved
            datapath = os.path.join(rundir, expname, 'output', 'ifs', leg)
            # check if we need to do postprocessing
            if postprocess:
                self.runPostprocess(datapath, expname, postprocess)
            # check if we need to archive the original files
            if archive:
                self.runArchive(archive, expname, datapath, remove=True)
        else:
            raise IOError("Output for leg {} not found".format(leg))

    def runPostprocess(self, datapath, expname, outputdir):
        '''
        Run the postprocessing

        param datapath: path where the output data of the experiment is saved
        param expname: name of experiment
        param outputdir: output directory of post processed files
        type datapath: str
        type expname: str
        type outputdir: str
        '''
        # define full path output directory
        outputdir = os.path.join(outputdir, expname)
        try:
            os.makedirs(outputdir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        # logging level 'DEBUG' 'INFO' 'WARNING' 'ERROR' 'CRITICAL'
        logging.basicConfig(filename = outputdir + os.sep + 'history.log',
                            filemode = 'w+', level = logging.DEBUG,
                            format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # convert sp2gpl and copy to tmpdir
        tmpdir = tempfile.mkdtemp(dir=outputdir)
        self.sp2gpl(datapath, expname, outputdir, tmpdir)
        # actual postprocessing
        self.postprocess(tmpdir, expname, outputdir)
        # remove tmpdir
        shutil.rmtree(tmpdir)

    @staticmethod
    def runArchive(archive, expname, datapath, remove=False):
        '''
        Archiving original files using rsync

        param archive: path  of archive location
        param expname: name of experiment
        param datapath: path where the output data of the experiment is saved
        param remove: boolean indicating of original files need to be removed
                      after successful archiving
        type archive: str
        type expname: str
        type datapath: str
        type remove: bool
        '''
        # create subdir for experiment in archive location
        archive_dir = os.path.join(archive, expname)
        try:
            os.makedirs(archive_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        # archive using rsync
        subprocess.check_call(['rsync', '-az', '--recursive', datapath, archive_dir])
        # check if original files need to be removed
        if remove:
            # remove original files
            shutil.rmtree(datapath)  # don't automatically remove for now

    @staticmethod
    def setConstants():
        '''
        Define constants used in the calculations
        
        returns: dictionary with constants for g, R. cp, Lv, R_dry and R_vap
        rtype: dict
        '''
        # define the constant:
        constant = {'g' : 9.80616,      # gravititional acceleration [m / s2]
                    'R' : 6371009,      # radius of the earth [m]
                    'cp': 1004.64,      # heat capacity of air [J/(Kg*K)]
                    'Lv': 2264670,      # Latent heat of vaporization [J/Kg]
                    'R_dry' : 286.9,    # gas constant of dry air [J/(kg*K)]
                    'R_vap' : 461.5,    # gas constant for water vapour [J/(kg*K)]
                   }
        return constant

    @staticmethod
    def defineSigmaLevels():
        '''
        Definine sigma levels

        returns: tuple containing arrays with A and B values for the definition of
                 sigma levellist
        rtype: tuple
        '''
        # Since there are 60 model levels, there are 61 half levels, so it is for A and B values
        # the unit of A is Pa!!!!!!!!!!!!
        # from surface to TOA
        A = np.array([0.0, 2.00004, 3.980832, 7.387186, 12.908319, 21.413612, 33.952858,
                      51.746601, 76.167656, 108.715561, 150.986023, 204.637451, 271.356506,
                      352.824493, 450.685791, 566.519226, 701.813354, 857.945801, 1036.166504,
                      1237.585449, 1463.16394, 1713.709595, 1989.87439, 2292.155518, 2620.898438,
                      2976.302246, 3358.425781, 3767.196045, 4202.416504, 4663.776367, 5150.859863,
                      5663.15625, 6199.839355, 6759.727051, 7341.469727, 7942.92627, 8564.624023,
                      9208.305664, 9873.560547, 10558.881836, 11262.484375, 11982.662109, 12713.897461,
                      13453.225586,14192.009766, 14922.685547, 15638.053711, 16329.560547,16990.623047,
                      17613.28125, 18191.029297, 18716.96875, 19184.544922, 19587.513672, 19919.796875,
                      20175.394531, 20348.916016, 20434.158203, 20426.21875, 20319.011719, 20107.03125,
                      19785.357422, 19348.775391, 18798.822266, 18141.296875, 17385.595703, 16544.585938,
                      15633.566406, 14665.645508, 13653.219727, 12608.383789, 11543.166992, 10471.310547,
                      9405.222656, 8356.25293, 7335.164551, 6353.920898, 5422.802734, 4550.21582,
                      3743.464355, 3010.146973, 2356.202637, 1784.854614, 1297.656128, 895.193542,
                      576.314148, 336.772369, 162.043427, 54.208336, 6.575628, 0.00316, 0.0],dtype=float)
        B = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 1.4e-005,
                      5.5e-005, 0.000131, 0.000279, 0.000548, 0.001, 0.001701,
                      0.002765, 0.004267, 0.006322, 0.009035, 0.012508, 0.01686,
                      0.022189, 0.02861, 0.036227, 0.045146, 0.055474, 0.067316,
                      0.080777, 0.095964, 0.112979, 0.131935, 0.152934, 0.176091,
                      0.20152, 0.229315, 0.259554, 0.291993, 0.326329, 0.362203,
                      0.399205, 0.436906, 0.475016, 0.51328, 0.551458, 0.589317,
                      0.626559, 0.662934, 0.698224, 0.732224, 0.764679, 0.795385,
                      0.824185, 0.85095, 0.875518, 0.897767, 0.917651, 0.935157,
                      0.950274, 0.963007, 0.973466, 0.982238, 0.989153, 0.994204, 0.99763, 1.0],dtype=float)
        return (A,B)


    @staticmethod
    def var_key_retrieve(datapath, expname, index_dict):
        """
        Obtain basic parameters from GRIB file for further computation.
        param datapath: path of input grib file
        param expname: name of the experiment
        param index_dict: information about messages of the grib file
        
        return: keys to the grib files on Gaussian grid and spectral grid,
                as well as some basic parameters
        rtype: int & numpy.arrays
        """
        # use pygrib to read the grib files
        ##########################################################################
        # Due to the characteristic of GRIB file, it is highly efficient to read #
        # messages monotonically! Thus, for the sake of processing time,         #
        #                  PLEASE DON'T RETRIVE BACKWARD!                        #
        ##########################################################################
        # find starting time of leg from filename in leg directory
        filenames = [os.path.basename(f) for f in glob.glob(os.path.join(datapath, 'ICMGG*'))]
        file_time = [a for a in [a.lstrip('ICMGG' + expname + '+') for a in filenames] if int(a)][0]
        print("Start retrieving datasets ICMSHECE and ICMGGECE for the time {}".format(file_time))
        logging.info("Start retrieving variables T,q,u,v,sp,gz for from ICMSHECE and ICMGGECE for the time {}".format(file_time))
        ICMGGECE = pygrib.open(os.path.join(datapath, "ICMGG{}+{}".format(expname, file_time)))
        ICMSHECE = pygrib.open(os.path.join(datapath, "ICMSH{}+{}".format(expname, file_time)))
        print("Retrieving datasets successfully and return the key!")
        # extract the basic information about the dataset
        num_message_SH = ICMSHECE.messages
        num_message_GG = ICMGGECE.messages
        # number of days in this month
        days = (num_message_GG/index_dict['num_GG_per']+1)/8 # no 00:00:00 at each year
        # number of records
        num_record = num_message_GG/index_dict['num_GG_per']
        # get the first message
        first_message = ICMGGECE.message(1)
        # extract the latitudes and longitudes
        lats, lons = first_message.latlons()
        latitude = lats[:,0]
        longitude = lons[0,:]
        print("====================================================")
        print("==============  Output Data Profile  ===============")
        print("There are {} messages included in the spectral field".format(num_message_SH))
        print("There are {} messages included in the Gaussian grid".format(num_message_GG))
        print("There are {} days in this month ({})".format(days,file_time))
        print("====================================================")
        logging.info("Retrieving variables for {} successfully!".format(file_time))
        
        return ICMSHECE, ICMGGECE, num_message_SH, num_message_GG, num_record, days, latitude, longitude, file_time

    @staticmethod
    def message_reader(datapath, expname):
        """
        This function serves to read the message list of each GRIB file.
        The message number of each variable that will be used in the computation
        of meridional energy transport as well as the mass budget correction is
        obtained here.
        param datapath: path of input grib file
        param expname: name of the experiment
        
        return: message number of each variable in need for further computation of AMET
        rtype: dictionary    
        """
        print('Please make sure that the message list satisfy the rule of ECMWF IFS.')
        print('The variable name in GRIB file should follow the order as given by ECMWF IFS.')
        # http://apps.ecmwf.int/codes/grib/param-db
        print('For more information, please check the ECMWF parameter base!')
        # use pygrib to read the grib files
        filenames = [os.path.basename(f) for f in glob.glob(os.path.join(datapath, 'ICMGG*'))]
        file_time = [a for a in [a.lstrip('ICMGG' + expname + '+') for a in filenames] if int(a)][0]
        ICMGGECE = pygrib.open(os.path.join(datapath, "ICMGG{}+{}".format(expname, file_time)))
        ICMSHECE = pygrib.open(os.path.join(datapath, "ICMSH{}+{}".format(expname, file_time)))
        # enumerate the message and create a list for it
        namelist_GG_pygrib = [i for i in ICMGGECE] # ignore the first 2
        ICMGGECE.close()
        namelist_SH_pygrib = [i for i in ICMSHECE]
        ICMSHECE.close()
        namelist_GG = [str(i) for i in namelist_GG_pygrib[:5000]]
        namelist_SH = [str(i) for i in namelist_SH_pygrib[:5000]]
        # create index dict for further computation
        index_dict = {}
        # calculate the number of messages per record
        message_1st = namelist_GG[0]
        for i, s in enumerate(namelist_GG[1:]):
            if message_1st[2:15] in s: # skip "1:"
                index_dict['num_GG_per'] = i + 1
                break
        message_1st = namelist_SH[0]
        for i, s in enumerate(namelist_SH[1:]):
            if message_1st[2:15] in s: # skip "1:"
                index_dict['num_SH_per'] = i + 1
                break
        # list of standard variable name as message in GRIB from ECMWF
        name_u = 'U component of wind'
        name_v = 'V component of wind'
        name_T = 'Temperature'
        name_gz = 'Geopotential'
        name_sp = 'Surface pressure'
        name_q = 'Specific humidity'
        # check the message index of 1st specific humidity in the 1st record
        # attention! the index here are message index instead of python index!
        for i, s in enumerate(namelist_GG[:]):
            if name_q in s:
                index_dict['q'] = i + 1
                break
        # check the message index of 1st surface pressure in the 1st record
        for i, s in enumerate(namelist_GG[:]):
            if name_sp in s:
                index_dict['sp'] = i + 1
                break            
        # check the message index of 1st zonal velocity in the 1st record
        for i, s in enumerate(namelist_SH[:]):
            if name_u in s:
                index_dict['u'] = i + 1
                break
        # check the message index of 1st zonal velocity in the 1st record
        for i, s in enumerate(namelist_SH[:]):
            if name_v in s:
                index_dict['v'] = i + 1
                break
        # check the message index of 1st zonal velocity in the 1st record
        for i, s in enumerate(namelist_SH[:]):
            if name_T in s:
                index_dict['T'] = i + 1
                break
        # check the message index of 1st zonal velocity in the 1st record
        for i, s in enumerate(namelist_SH[:]):
            if name_gz in s:
                index_dict['gz'] = i + 1
                break   
            
        return index_dict

    def visualization(self, E_total, E_internal, E_latent,
                      E_geopotential, E_kinetic, output_path, expname, filename):
        """
        Plot the monthly mean of zonal integral of AMET, for a first check.
        param E_total: vertical integral of total meridional energy transport in the atmosphere
        
        return: figure of monthly mean of zonal integral of AMET
        rtype: png
        """
        # make plots
        print("Start making plots for the total meridional energy transport and each component.")
        logging.info("Start making plots for the total meridional energy transport and each component.")
        # calculate monthly mean of total energy transport
        # unit change from tera to peta (from 1E+12 to 1E+15)
        E_total_monthly_mean = E_total / 1000
        # Plot the total meridional energy transport against the latitude
        fig1 = plt.figure()
        plt.plot(self.latitude,E_total_monthly_mean,'b-',label='EC-earth')
        plt.axhline(y=0, color='r',ls='--')
        #plt.hold()
        plt.title('Total Atmospheric Meridional Energy Transport {} {}'.format(expname, filename))
        #plt.legend()
        plt.xlabel("Laitude")
        plt.xticks(np.linspace(-90,90,13))
        #plt.yticks(np.linspace(0,6,7))
        plt.ylabel("Meridional Energy Transport (PW)")
        #plt.show()
        fig1.savefig(os.path.join(output_path, 'AMET_EC-earth_total_{}_{}.png'.format(expname, filename)), dpi = 300)
        plt.close(fig1)

    def create_netcdf_point (self, meridional_E_point_pool,meridional_E_internal_point_pool,
                             meridional_E_latent_point_pool,meridional_E_geopotential_point_pool,
                             meridional_E_kinetic_point_pool,uc_point_pool,vc_point_pool,
                             output_path,expname,filename):
        """
        Save all the components of AMET on geographical grid as a netcdf file.
        No compression is applied.
        param meridional_E_point_pool: monthly mean total AMET
        param meridional_E_internal_point_pool: monthly mean internal energy cpT
        param meridional_E_latent_point_pool: monthly mean latent heat Lvq
        param meridional_E_geopotential_point_pool: monthly mean geopotential energy gz
        param meridional_E_kinetic_point_pool: monthly mean kinetic energy v2
        param uc_point_pool: baratropic zonal correction wind component
        param vc_point_pool: baratropic meridional correction wind component
        param output_path: path of output netcdf file
        param expname: name of the experiment
        param filename: name of the output netcdf file based on time
        
        return: netcdf file of AMET
        rtype: netcdf
        """
        # save output datasets
        print('*******************************************************************')
        print('*********************** create netcdf file*************************')
        print('*******************************************************************')
        logging.info("Start creating netcdf file for total meridional energy transport and each component at each grid point.")
        # wrap the datasets into netcdf file
        # 'NETCDF3_CLASSIC', 'NETCDF3_64BIT', 'NETCDF4_CLASSIC', and 'NETCDF4'
        data_wrap = Dataset(os.path.join(output_path,'AMET_EC-earth_model_daily_{}_{}_E_point.nc'.format(expname, filename)),
                            'w',format = 'NETCDF4')
        # create dimensions for netcdf data
        lat_wrap_dim = data_wrap.createDimension('latitude', self.Dim_latitude)
        lon_wrap_dim = data_wrap.createDimension('longitude', self.Dim_longitude)
        # create coordinate variables for 3-dimensions
        lat_wrap_var = data_wrap.createVariable('latitude',np.float32,('latitude',))
        lon_wrap_var = data_wrap.createVariable('longitude',np.float32,('longitude',))
        # create the actual 3-d variable
        uc_wrap_var = data_wrap.createVariable('uc',np.float32,('latitude','longitude'))
        vc_wrap_var = data_wrap.createVariable('vc',np.float32,('latitude','longitude'))

        E_total_wrap_var = data_wrap.createVariable('E',np.float64,('latitude','longitude'), zlib=True)
        E_internal_wrap_var = data_wrap.createVariable('E_cpT',np.float64,('latitude','longitude'), zlib=True)
        E_latent_wrap_var = data_wrap.createVariable('E_Lvq',np.float64,('latitude','longitude'), zlib=True)
        E_geopotential_wrap_var = data_wrap.createVariable('E_gz',np.float64,('latitude','longitude'), zlib=True)
        E_kinetic_wrap_var = data_wrap.createVariable('E_uv2',np.float64,('latitude','longitude'), zlib=True)
        # global attributes
        data_wrap.description = 'Monthly mean meridional energy transport and each component at each grid point'
        # variable attributes
        lat_wrap_var.units = 'degree_north'
        lon_wrap_var.units = 'degree_east'
        uc_wrap_var.units = 'm/s'
        vc_wrap_var.units = 'm/s'
        E_total_wrap_var.units = 'tera watt'
        E_internal_wrap_var.units = 'tera watt'
        E_latent_wrap_var.units = 'tera watt'
        E_geopotential_wrap_var.units = 'tera watt'
        E_kinetic_wrap_var.units = 'tera watt'

        uc_wrap_var.long_name = 'zonal barotropic correction wind'
        vc_wrap_var.long_name = 'meridional barotropic correction wind'
        E_total_wrap_var.long_name = 'atmospheric meridional energy transport'
        E_internal_wrap_var.long_name = 'atmospheric meridional internal energy transport'
        E_latent_wrap_var.long_name = 'atmospheric meridional latent heat transport'
        E_geopotential_wrap_var.long_name = 'atmospheric meridional geopotential transport'
        E_kinetic_wrap_var.long_name = 'atmospheric meridional kinetic energy transport'
        # writing data
        lat_wrap_var[:] = self.latitude
        lon_wrap_var[:] = self.longitude
        uc_wrap_var[:] = uc_point_pool
        vc_wrap_var[:] = vc_point_pool
        E_total_wrap_var[:] = meridional_E_point_pool
        E_internal_wrap_var[:] = meridional_E_internal_point_pool
        E_latent_wrap_var[:] = meridional_E_latent_point_pool
        E_geopotential_wrap_var[:] = meridional_E_geopotential_point_pool
        E_kinetic_wrap_var[:] = meridional_E_kinetic_point_pool
         # close the file
        data_wrap.close()
        print("Create netcdf file successfully")
        logging.info("The generation of netcdf files for the total meridional energy transport and each component on each grid point is complete!!")

    # save output datasets
    def create_netcdf_zonal_int (self, meridional_E_pool, meridional_E_internal_pool,
                                 meridional_E_latent_pool, meridional_E_geopotential_pool,
                                 meridional_E_kinetic_pool, output_path,expname, filename):
        """
        Save all the zonal integral of components of AMET as a netcdf file.
        
        return: netcdf file of zonal integral of AMET
        rtype: netcdf
        """
        print('*******************************************************************')
        print('*********************** create netcdf file*************************')
        print('*******************************************************************')
        logging.info("Start creating netcdf files for the zonal integral of total meridional energy transport and each component.")
        # wrap the datasets into netcdf file
        # 'NETCDF3_CLASSIC', 'NETCDF3_64BIT', 'NETCDF4_CLASSIC', and 'NETCDF4'
        data_wrap = Dataset(os.path.join(output_path, 'AMET_EC-earth_model_daily_{}_{}_E_zonal_int.nc'.format(expname, filename)),
                            'w',format = 'NETCDF4')
        # create dimensions for netcdf data
        lat_wrap_dim = data_wrap.createDimension('latitude', self.Dim_latitude)
        # create coordinate variables for 3-dimensions
        lat_wrap_var = data_wrap.createVariable('latitude',np.float32,('latitude',))
        # create the actual 3-d variable
        E_total_wrap_var = data_wrap.createVariable('E',np.float64,('latitude',), zlib=True)
        E_internal_wrap_var = data_wrap.createVariable('E_cpT',np.float64,('latitude',), zlib=True)
        E_latent_wrap_var = data_wrap.createVariable('E_Lvq',np.float64,('latitude',), zlib=True)
        E_geopotential_wrap_var = data_wrap.createVariable('E_gz',np.float64,('latitude',), zlib=True)
        E_kinetic_wrap_var = data_wrap.createVariable('E_uv2',np.float64,('latitude',), zlib=True)
        # global attributes
        data_wrap.description = 'Monthly mean zonal integral of meridional energy transport and each component'
        # variable attributes
        lat_wrap_var.units = 'degree_north'
        E_total_wrap_var.units = 'tera watt'
        E_internal_wrap_var.units = 'tera watt'
        E_latent_wrap_var.units = 'tera watt'
        E_geopotential_wrap_var.units = 'tera watt'
        E_kinetic_wrap_var.units = 'tera watt'
        E_total_wrap_var.long_name = 'atmospheric meridional energy transport'
        E_internal_wrap_var.long_name = 'atmospheric meridional internal energy transport'
        E_latent_wrap_var.long_name = 'atmospheric meridional latent heat transport'
        E_geopotential_wrap_var.long_name = 'atmospheric meridional geopotential transport'
        E_kinetic_wrap_var.long_name = 'atmospheric meridional kinetic energy transport'
        # writing data
        lat_wrap_var[:] = self.latitude
        E_total_wrap_var[:] = meridional_E_pool
        E_internal_wrap_var[:] = meridional_E_internal_pool
        E_latent_wrap_var[:] = meridional_E_latent_pool
        E_geopotential_wrap_var[:] = meridional_E_geopotential_pool
        E_kinetic_wrap_var[:] = meridional_E_kinetic_pool
        # close the file
        data_wrap.close()
        print("Create netcdf file successfully")
        logging.info("The generation of netcdf files for the zonal integral of total meridional energy transport and each component is complete!!")

    @staticmethod
    def sp2gpl(datapath, expname, outputdir, tmpdir):
        '''
        Convert spectral field to grid field

        param datapath:
        param expname:
        param outputdir:
        param tmpdir:
        type datapath:
        type expname:
        type outputdir:
        type tmpdir:
        '''
        # create/cleanup tmpdir       
        try:
            shutil.rmtree(tmpdir)  # cleanup tmpdir
        except OSError:
            pass
        try:
            os.makedirs(tmpdir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        # get filenames to convert
        filenames = [os.path.basename(f) for f in glob.glob(os.path.join(datapath, 'ICMGG*'))]
        file_time = [a for a in [a.lstrip('ICMGG{}+'.format(expname)) for a in filenames] if int(a)][0]
        print("Start retrieving datasets ICMSHECE and ICMGGECE for the time {}".format(file_time))
        logging.info("Start retrieving variables T,q,u,v,sp,gz for from ICMSHECE and ICMGGECE for the time {}".format(file_time))
        # define filenames
        ICMGG = "ICMGG{}+{}".format(expname, file_time)
        ICMSH = "ICMSH{}+{}".format(expname, file_time)
        # rsync the guassian grid output
        subprocess.check_call(['rsync', '-az', os.path.join(datapath, ICMGG), os.path.join(tmpdir, ICMGG)])
        # sp2gpl -> tmpdir
        subprocess.check_call(['cdo', 'sp2gpl', os.path.join(datapath, ICMSH), os.path.join(tmpdir, ICMSH)])
        # extract relevant output fields and save to output directory
        subprocess.check_call(['cdo', '-t', 'ecmwf', '-f', 'nc4', '-R', 'ml2pl,85000,50000,20000',
                               os.path.join(tmpdir, ICMGG), os.path.join(tmpdir, 'gaus.nc')])
        subprocess.check_call(['cdo', '-t', 'ecmwf', '-f', 'nc4', '-R', 'ml2pl,85000,50000,20000',
                               os.path.join(tmpdir, ICMSH), os.path.join(tmpdir, 'spectral.nc')])
        subprocess.check_call(['cdo', 'selvar,U,V,T,Z', '-selhour,0,6,12,18', os.path.join(tmpdir, 'spectral.nc'),
                               os.path.join(tmpdir, 'sp-out.nc')])
        subprocess.check_call(['cdo', 'selhour,0,6,12,18', os.path.join(tmpdir, 'gaus.nc'),
                               os.path.join(tmpdir, 'gaus-out.nc')])
        subprocess.check_call(['cdo', 'selvar,Q,PT,T2M,U10M,V10M,SLHF,SP,MSL,LSP,CP,TCC,SSHF,SSR,STR,TSR,TTR,var8',
                               os.path.join(tmpdir, 'gaus-out.nc'), os.path.join(tmpdir, 'gaus-out2.nc')])
        # rename var8 to SRO (surface runoff)
        subprocess.check_call(['cdo', 'chname,var8,SRO', os.path.join(tmpdir, 'gaus-out2.nc'),
                               os.path.join(tmpdir, 'gaus-out3.nc')])
        # merge output
        outputfile = os.path.join(outputdir, "ECE_{}_{}.nc".format(expname, file_time))
        subprocess.check_call(['cdo', 'merge', os.path.join(tmpdir, 'sp-out.nc'),
                               os.path.join(tmpdir, 'gaus-out3.nc'), outputfile])

    def postprocess(self, tmpdir, expname, output_path):
        """
        Postprocess module to perform the mass budget correction and
        calculate the meridional energy transport in the atmosphere.
        param tmpdir: path to the temporary directory to keep the grib files after converting
                      variables on the spectral grid to the geographical grid
        param expname: name of experiment
        param output_path: location for the output file
        
        return: N/A
        rtype: N/A
        """
        # set constants and sigma levels
        constant = self.setConstants()
        (A,B) = self.defineSigmaLevels()
        ####################################################################
        ####  use pygrib to get the message list from ec-earth outputs  ####
        ####  the message list is used to calc. the index of variables  ####
        ####################################################################
        index_dict = self.message_reader(tmpdir, expname)
        ####################################################################
        ######  use pygrib.open to get the key from ec-earth outputs  ######
        ####################################################################
        ICMSHECE, ICMGGECE, num_message_SH, num_message_GG, num_record, days,\
        latitude, longitude, file_time = self.var_key_retrieve(tmpdir, expname, index_dict)
        ####################################################################
        ######       Extract invariant and calculate constants       #######
        ####################################################################
        # create dimensions for saving data
        Dim_level = 91
        Dim_latitude = 512
        Dim_longitude = 1024
        # number of messages for one record
        num_SH_per = index_dict['num_SH_per']
        num_GG_per = index_dict['num_GG_per']
        # calculate zonal & meridional grid size on earth
        # the earth is taken as a perfect sphere, instead of a ellopsoid
        dx = 2 * np.pi * constant['R'] * np.cos(2 * np.pi * latitude / 360) / len(longitude)
        dy = np.pi * constant['R'] / (len(latitude)-1)
        ####################################################################
        ###  Create space for stroing intermediate variables and outputs ###
        ####################################################################
        # for climatological variables
        u = np.zeros((Dim_level, len(latitude), len(longitude)), dtype=float)
        v = np.zeros((Dim_level, len(latitude), len(longitude)), dtype=float)
        T = np.zeros((Dim_level, len(latitude), len(longitude)), dtype=float)
        gz = np.zeros((Dim_level, len(latitude), len(longitude)), dtype=float)
        sp = np.zeros((len(latitude), len(longitude)), dtype=float)
        q = np.zeros((Dim_level, len(latitude), len(longitude)), dtype=float)
        # for vertical integral
        dp = np.zeros((Dim_level,len(latitude),len(longitude)),dtype=float)
        # for tendency terms
        q_start = np.zeros((Dim_level, len(latitude), len(longitude)), dtype=float)
        q_end = np.zeros((Dim_level, len(latitude), len(longitude)), dtype=float)
        sp_start = np.zeros((len(latitude), len(longitude)), dtype=float)
        sp_end = np.zeros((len(latitude), len(longitude)), dtype=float)
        dp_start = np.zeros((Dim_level, len(latitude), len(longitude)), dtype=float)
        dp_end = np.zeros((Dim_level, len(latitude), len(longitude)), dtype=float)
        # data pool for grid point values
        uc_point_pool = np.zeros((Dim_latitude, Dim_longitude), dtype = float)
        vc_point_pool = np.zeros((Dim_latitude, Dim_longitude), dtype = float)
        meridional_E_point_pool = np.zeros((Dim_latitude, Dim_longitude), dtype = float)
        meridional_E_internal_point_pool = np.zeros((Dim_latitude, Dim_longitude), dtype = float)
        meridional_E_latent_point_pool = np.zeros((Dim_latitude, Dim_longitude), dtype = float)
        meridional_E_geopotential_point_pool = np.zeros((Dim_latitude, Dim_longitude), dtype = float)
        meridional_E_kinetic_point_pool = np.zeros((Dim_latitude, Dim_longitude), dtype = float)
        # data pool for zonal integral
        meridional_E_pool = np.zeros((Dim_latitude), dtype = float)
        meridional_E_internal_pool = np.zeros((Dim_latitude), dtype = float)
        meridional_E_latent_pool = np.zeros((Dim_latitude), dtype = float)
        meridional_E_geopotential_pool = np.zeros((Dim_latitude), dtype = float)
        meridional_E_kinetic_pool = np.zeros((Dim_latitude), dtype = float)
        ####################################################################
        ####  Create space for stroing temporary variables and outputs  ####
        ####################################################################
        # data pool for mass budget correction module
        pool_div_moisture_flux_u = np.zeros((num_record, Dim_latitude, Dim_longitude), dtype=float)
        pool_div_moisture_flux_v = np.zeros((num_record, Dim_latitude, Dim_longitude), dtype=float)
        pool_div_mass_flux_u = np.zeros((num_record, Dim_latitude, Dim_longitude), dtype=float)
        pool_div_mass_flux_v = np.zeros((num_record, Dim_latitude, Dim_longitude), dtype=float)
        pool_precipitable_water = np.zeros((num_record, Dim_latitude, Dim_longitude), dtype=float)
        pool_sp = np.zeros((num_record, Dim_latitude, Dim_longitude), dtype=float)
        # data pool for meridional energy tansport module
        pool_internal_flux_int = np.zeros((num_record, Dim_latitude, Dim_longitude), dtype=float)
        pool_latent_flux_int = np.zeros((num_record, Dim_latitude, Dim_longitude), dtype=float)
        pool_geopotential_flux_int = np.zeros((num_record, Dim_latitude, Dim_longitude), dtype=float)
        pool_kinetic_flux_int = np.zeros((num_record, Dim_latitude, Dim_longitude), dtype=float)
        # data pool for the correction of meridional energy tansport
        pool_heat_flux_int = np.zeros((num_record, Dim_latitude, Dim_longitude), dtype=float)
        pool_vapor_flux_int = np.zeros((num_record, Dim_latitude, Dim_longitude), dtype=float)
        pool_geo_flux_int = np.zeros((num_record, Dim_latitude, Dim_longitude), dtype=float)
        pool_velocity_flux_int = np.zeros((num_record, Dim_latitude, Dim_longitude), dtype=float)
        ###############################################################################
        ###  extract variables and calculate the vertical integrated zonal integral ###
        ###############################################################################
        for i in np.arange(num_record):
            ################################################################
            ######       Get all the variables - spectral field      #######
            #### The iterators are calculated from the list of messages	####
            ## Please check the example message list for more information ##
            ################################################################
            # for the variables on the spectral fields
            index_SH = index_dict['u'] + i * num_SH_per
            while (index_SH < (index_dict['u'] + Dim_level + i * num_SH_per)):
                key_u = ICMSHECE.message(index_SH)
                u[index_SH-index_dict['u']-i*num_SH_per,:,:] = key_u.values
                index_SH = index_SH + 1
            index_SH = index_dict['v'] + i * num_SH_per
            while (index_SH < (index_dict['v'] + Dim_level + i * num_SH_per)):
                key_v = ICMSHECE.message(index_SH)
                v[index_SH-index_dict['v']-i*num_SH_per,:,:] = key_v.values
                index_SH = index_SH + 1
            index_SH = index_dict['T'] + i * num_SH_per
            while (index_SH < (index_dict['T'] + Dim_level + i * num_SH_per)):
                key_T = ICMSHECE.message(index_SH)
                T[index_SH-index_dict['T']-i*num_SH_per,:,:] = key_T.values
                index_SH = index_SH + 1
            index_SH = index_dict['gz'] + i * num_SH_per
            while (index_SH < (index_dict['gz'] + Dim_level + i * num_SH_per)):
                key_gz = ICMSHECE.message(index_SH)
                gz[index_SH-index_dict['gz']-i*num_SH_per,:,:] = key_gz.values
                index_SH = index_SH + 1
            print("Retrieving datasets on the spectral fields successfully for the {} record!".format(i+1))
            logging.info("Retrieving variables on the spectral fields for the {} record successfully!".format(i+1))
            ############################################################
            ######       Get all the fields - Gaussian grid      #######
            ############################################################
            # for the computation of AMET
            index_GG = index_dict['q'] + i * num_GG_per
            while (index_GG < (index_dict['q'] + Dim_level + i*num_GG_per)):
                key_q = ICMGGECE.message(index_GG)
                q[index_GG-index_dict['q']-i*num_GG_per,:,:] = key_q.values
                index_GG = index_GG + 1
	        index_GG = index_dict['sp'] + i * num_GG_per
            key_sp = ICMGGECE.message(index_GG)
            sp = key_sp.values
            print("Retrieving datasets on the Gaussian grid successfully for the {} record!".format(i+1))
            logging.info("Retrieving variables on the Gaussian grid for the {} record successfully!".format(i+1))
            ############################################################
            ######    for the computation of tendency terms      #######
            ############################################################
            if i == 0:
                q_start = q
                sp_start = sp
            elif i == (num_record - 1):
                q_end = q
                sp_end = sp
            ############################################################
            ######       calculate flux (vertical integral)      #######
            ######           meridional energy transport         #######
            ############################################################
            # use matrix A and B to calculate dp based on half pressure level
            index_level = np.arange(Dim_level)
            for j in index_level:
                dp[j,:,:] = (A[j+1] + B[j+1] * sp) - (A[j] + B[j] * sp)
            # calculate each component of total energy
            # take the vertical integral
            # variables for correction
            # Internal Energy cpT
            internal_flux = constant['cp'] * v * T * dp / constant['g']
            internal_flux_int = np.sum(internal_flux, 0)
            # Latent heat Lq
            latent_flux = constant['Lv'] * v * q * dp / constant['g']
            latent_flux_int = np.sum(latent_flux, 0)
            # geopotential gz
            geopotential_flux = v * gz * dp / constant['g']
            geopotential_flux_int = np.sum(geopotential_flux, 0)
            # kinetic energy
            kinetic_flux = v * 1/2 *(u**2 + v**2) * dp / constant['g']
            kinetic_flux_int = np.sum(kinetic_flux, 0)
            # variables for correction
            # for the correction of Internal Energy cpT
            heat_flux = constant['cp'] * T * dp / constant['g']
            heat_flux_int = np.sum(heat_flux, 0)
            # for the correction of Latent Heat flux Lq
            vapor_flux = constant['Lv'] * q * dp / constant['g']
            vapor_flux_int = np.sum(vapor_flux, 0)
            # for the correction of Geopotential flux gz
            geo_flux = gz * dp / constant['g']
            geo_flux_int = np.sum(geo_flux, 0)
            # for the correction of Kinetic Energy flux u2
            velocity_flux = 1/2 *(u**2 + v**2) * dp / constant['g']
            velocity_flux_int = np.sum(velocity_flux, 0)
            print('Complete calculating meridional energy transport on model level')
            # save the divergence terms to the warehouse
            pool_internal_flux_int[i,:,:] = internal_flux_int
            pool_latent_flux_int[i,:,:] = latent_flux_int
            pool_geopotential_flux_int[i,:,:] = geopotential_flux_int
            pool_kinetic_flux_int[i,:,:] = kinetic_flux_int
            # variables for the correction of each energy component
            pool_heat_flux_int[i,:,:] = heat_flux_int
            pool_vapor_flux_int[i,:,:] = vapor_flux_int
            pool_geo_flux_int[i,:,:] = geo_flux_int
            pool_velocity_flux_int[i,:,:] = velocity_flux_int
            ############################################################
            ######       calculate flux (vertical integral)      #######
            ######             mass budget correction            #######
            ############################################################
            print('Begin the calculation of divergent verically integrated moisture flux.')
            # calculte the mean moisture flux for a certain month
            moisture_flux_u = u * q * dp / constant['g']
            moisture_flux_v = v * q * dp / constant['g']
            # take the vertical integral
            moisture_flux_u_int = np.sum(moisture_flux_u, 0)
            moisture_flux_v_int = np.sum(moisture_flux_v, 0)
            # save memory
            #del moisture_flux_u, moisture_flux_v
            # calculate the divergence of moisture flux
            div_moisture_flux_u = np.zeros((Dim_latitude, Dim_longitude), dtype = float)
            div_moisture_flux_v = np.zeros((Dim_latitude, Dim_longitude), dtype = float)
            ######################## Attnention to the coordinate and symbol #######################
            # zonal moisture flux divergence
            for j in np.arange(Dim_latitude):
                for k in np.arange(Dim_longitude):
                    # the longitude could be from 0 to 360 or -180 to 180, but the index remains the same
                    if k == 0:
                        div_moisture_flux_u[j,k] = (moisture_flux_u_int[j,k+1] - moisture_flux_u_int[j,-1]) / (2 * dx[j])
                    elif k == (Dim_longitude-1) :
                        div_moisture_flux_u[j,k] = (moisture_flux_u_int[j,0] - moisture_flux_u_int[j,k-1]) / (2 * dx[j])
                    else:
                        div_moisture_flux_u[j,k] = (moisture_flux_u_int[j,k+1] - moisture_flux_u_int[j,k-1]) / (2 * dx[j])
            # meridional moisture flux divergence
            # the latitude is from 90N to -90S
            for j in np.arange(Dim_latitude):
                if j == 0:
                    div_moisture_flux_v[j,:] = -(moisture_flux_v_int[j+1,:] - moisture_flux_v_int[j,:]) / (2 * dy)
                elif j == (Dim_latitude-1):
                    div_moisture_flux_v[j,:] = -(moisture_flux_v_int[j,:] - moisture_flux_v_int[j-1,:]) / (2 * dy)
                else:
                    div_moisture_flux_v[j,:] = -(moisture_flux_v_int[j+1,:] - moisture_flux_v_int[j-1,:]) / (2 * dy)
            print('The calculation of divergent verically integrated moisture flux is finished !!')
            # save the divergence terms to the warehouse
            pool_div_moisture_flux_u[i,:,:] = div_moisture_flux_u
            pool_div_moisture_flux_v[i,:,:] = div_moisture_flux_v
            ########################################################################
            print('Begin the calculation of divergent verically integrated mass flux.')
            # calculate the mass flux
            mass_flux_u = u * dp / constant['g']
            mass_flux_v = v * dp / constant['g']
            # take the vertical integral
            mass_flux_u_int = np.sum(mass_flux_u, 0)
            mass_flux_v_int = np.sum(mass_flux_v, 0)
            # calculate the divergence of moisture flux
            div_mass_flux_u = np.zeros((Dim_latitude, Dim_longitude), dtype = float)
            div_mass_flux_v = np.zeros((Dim_latitude, Dim_longitude), dtype = float)
            # zonal mass flux divergence
            for j in np.arange(Dim_latitude):
                for k in np.arange(Dim_longitude):
                    # the longitude could be from 0 to 360 or -180 to 180, but the index remains the same
                    if k == 0:
                        div_mass_flux_u[j,k] = (mass_flux_u_int[j,k+1] - mass_flux_u_int[j,-1]) / (2 * dx[j])
                    elif k == (Dim_longitude-1) :
                        div_mass_flux_u[j,k] = (mass_flux_u_int[j,0] - mass_flux_u_int[j,k-1]) / (2 * dx[j])
                    else:
                        div_mass_flux_u[j,k] = (mass_flux_u_int[j,k+1] - mass_flux_u_int[j,k-1]) / (2 * dx[j])
            # meridional mass flux divergence
            for j in np.arange(Dim_latitude):
                if j == 0:
                    div_mass_flux_v[j,:] = -(mass_flux_v_int[j+1,:] - mass_flux_v_int[j,:]) / (2 * dy)
                elif j == (Dim_latitude-1):
                    div_mass_flux_v[j,:] = -(mass_flux_v_int[j,:] - mass_flux_v_int[j-1,:]) / (2 * dy)
                else:
                    div_mass_flux_v[j,:] = -(mass_flux_v_int[j+1,:] - mass_flux_v_int[j-1,:]) / (2 * dy)
            print('The calculation of divergent verically integrated mass flux is finished !!')
            # save the divergence terms to the warehouse
            pool_div_mass_flux_u[i,:,:] = div_mass_flux_u
            pool_div_mass_flux_v[i,:,:] = div_mass_flux_v
            ########################################################################
            # calculate precipitable water
            precipitable_water = q * dp / constant['g']
            precipitable_water_int = np.sum(precipitable_water, 0)
            pool_precipitable_water[i,:,:] = precipitable_water_int
            # surface pressure for further mean comp
            pool_sp[i,:,:] = sp
        print("=====================================================================")
        print(" The extraction of variables and the computation of terms are done!! ")
        print("=====================================================================")
        # now we can close the grib files
        ICMSHECE.close()
        ICMGGECE.close()
        ############################################################
        ###########            mass correction           ###########
        ############################################################
        # calculate tendency terms
        # use matrix A and B to calculate dp based on half pressure level
        for j in index_level:
            dp_start[j,:,:] = (A[j+1] + B[j+1] * sp_start) - (A[j] + B[j] * sp_start)
            dp_end[j,:,:] = (A[j+1] + B[j+1] * sp_end) - (A[j] + B[j] * sp_end)
        print('Begin the calculation of precipitable water tendency')
        moisture_start = np.sum((q_start * dp_start), 0) # start of the current month
        moisture_end = np.sum((q_end * dp_end), 0) # end of the current month
        # compute the moisture tendency (one day has 86400s)
        moisture_tendency = (moisture_end - moisture_start) / (days * 86400) / constant['g']
        print('The calculation of precipitable water tendency is finished !!')
        print('Begin the calculation of surface pressure tendency')
        sp_tendency = (sp_end - sp_start) / (days * 86400) / constant['g']
        print('The calculation of surface pressure tendency is finished !!')
        logging.info("Finish calculating the moisture tendency and surface pressure tendency")
        print("Finish calculating the moisture tendency and surface pressure tendency")
        # calculate the bartropic correction wind velocity
        E_P = moisture_tendency + np.mean(pool_div_moisture_flux_u, 0) + np.mean(pool_div_moisture_flux_v, 0)
        print('*******************************************************************')
        print("******  Computation of E-P on each grid point is finished   *******")
        print('*******************************************************************')
        logging.info("Computation of E-P on each grid point is finished!")
        # calculate the mass residual
        mass_residual = sp_tendency + constant['g'] * (np.mean(pool_div_mass_flux_u, 0) +\
                        np.mean(pool_div_mass_flux_v, 0)) - constant['g'] * E_P
        print('*******************************************************************')
        print("*** Computation of mass residual on each grid point is finished ***")
        print('*******************************************************************')
        logging.info("Computation of mass residual on each grid point is finished!")
        # calculate barotropic correction wind
        print('Begin the calculation of barotropic correction wind.')
        uc = np.zeros((Dim_latitude,Dim_longitude), dtype = float)
        vc = np.zeros((Dim_latitude,Dim_longitude), dtype = float)
        vc = mass_residual * dy / (np.mean(pool_sp, 0) - constant['g'] * np.mean(pool_precipitable_water, 0))
        # extra modification for points at polor mesh
        vc[0,:] = 0
        vc[-1,:] = 0
        for c in np.arange(Dim_latitude):
            uc[c,:] = mass_residual[c,:] * dx[c] / (np.mean(pool_sp[:,c,:], 0) - constant['g'] * np.mean(pool_precipitable_water[:,c,:],0))
        print('********************************************************************************')
        print("*** Computation of barotropic correction wind on each grid point is finished ***")
        print('********************************************************************************')
        logging.info("Computation of barotropic correction wind on each grid point is finished!")
        ####################################################################
        ##########           Meridional Energy Transport          ##########
        ####################################################################
        # calculate the correction terms
        correction_internal_flux_int = vc * np.mean(pool_heat_flux_int, 0)
        correction_latent_flux_int = vc * np.mean(pool_vapor_flux_int, 0)
        correction_geopotential_flux_int = vc * np.mean(pool_geo_flux_int, 0)
        correction_kinetic_flux_int = vc * np.mean(pool_velocity_flux_int, 0)
        # calculate the total meridional energy transport and each component respectively
        # energy on grid point
        meridional_E_internal_point = np.zeros((len(latitude), len(longitude)), dtype=float)
        meridional_E_latent_point = np.zeros((len(latitude), len(longitude)), dtype=float)
        meridional_E_geopotential_point = np.zeros((len(latitude), len(longitude)), dtype=float)
        meridional_E_kinetic_point = np.zeros((len(latitude), len(longitude)), dtype=float)
        meridional_E_point = np.zeros((len(latitude), len(longitude)), dtype=float)
        for c in np.arange(Dim_latitude):
            meridional_E_internal_point[c,:] = (np.mean(pool_internal_flux_int[:,c,:], 0) - correction_internal_flux_int[c,:]) * dx[c] / 1e+12
            meridional_E_latent_point[c,:] = (np.mean(pool_latent_flux_int[:,c,:], 0) - correction_latent_flux_int[c,:]) * dx[c] / 1e+12
            meridional_E_geopotential_point[c,:] = (np.mean(pool_geopotential_flux_int[:,c,:], 0) - correction_geopotential_flux_int[c,:]) * dx[c] / 1e+12
            meridional_E_kinetic_point[c,:] = (np.mean(pool_kinetic_flux_int[:,c,:], 0) - correction_kinetic_flux_int[c,:]) * dx[c] / 1e+12
        # total energy transport
        meridional_E_point = meridional_E_internal_point + meridional_E_latent_point + meridional_E_geopotential_point + meridional_E_kinetic_point
        # zonal integral of energy
        meridional_E_internal = np.sum(meridional_E_internal_point, 1)
        meridional_E_latent = np.sum(meridional_E_latent_point, 1)
        meridional_E_geopotential = np.sum(meridional_E_geopotential_point, 1)
        meridional_E_kinetic = np.sum(meridional_E_kinetic_point, 1)
        # total energy transport
        meridional_E = meridional_E_internal + meridional_E_latent + meridional_E_geopotential + meridional_E_kinetic
        print('*****************************************************************************')
        print("***Computation of meridional energy transport in the atmosphere is finished**")
        print("************         The result is in tera-watt (1E+12)          ************")
        print('*****************************************************************************')
        logging.info("Computation of meridional energy transport on model level is finished!")
        ####################################################################
        #######              Final Data Wrapping (NetCDF)            #######
        ####################################################################
        # save the total meridional energy and each component to the data pool
        meridional_E_pool = meridional_E
        meridional_E_internal_pool = meridional_E_internal
        meridional_E_latent_pool = meridional_E_latent
        meridional_E_geopotential_pool = meridional_E_geopotential
        meridional_E_kinetic_pool = meridional_E_kinetic
        # save uc and vc to the data pool
        uc_point_pool = uc
        vc_point_pool = vc
        # save the meridional energy on each grid point to the data pool
        meridional_E_point_pool = meridional_E_point
        meridional_E_internal_point_pool = meridional_E_internal_point
        meridional_E_latent_point_pool = meridional_E_latent_point
        meridional_E_geopotential_point_pool = meridional_E_geopotential_point
        meridional_E_kinetic_point_pool = meridional_E_kinetic_point
        ####################################################################
        # make plots for monthly means
        self.latitude = latitude
        self.Dim_latitude = Dim_latitude
        self.longitude = longitude
        self.Dim_longitude = Dim_longitude
        self.num_record = num_record
        self.visualization(meridional_E_pool, meridional_E_internal_pool, meridional_E_latent_pool,
                           meridional_E_geopotential_pool, meridional_E_kinetic_pool, output_path,
                           expname,file_time)
        # save data as netcdf file
        self.create_netcdf_zonal_int(meridional_E_pool,meridional_E_internal_pool,
                                     meridional_E_latent_pool,meridional_E_geopotential_pool,
                                     meridional_E_kinetic_pool,output_path,expname,file_time)
        self.create_netcdf_point(meridional_E_point_pool,meridional_E_internal_point_pool,
                                 meridional_E_latent_point_pool,meridional_E_geopotential_point_pool,
                                 meridional_E_kinetic_point_pool,uc_point_pool,vc_point_pool,output_path,
                                 expname,file_time)
        print('Computation of meridional energy transport on model level for ERA-Interim is complete!!!')
        print('The output is in sleep, safe and sound!!!')
        logging.info("The full pipeline of the quantification of meridional energy transport in the atmosphere is accomplished!")
