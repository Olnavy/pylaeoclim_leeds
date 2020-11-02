import paleoclim_leeds.processing as proc
import paleoclim_leeds.zones as zones
import numpy as np
import xarray as xr
import paleoclim_leeds.util_hadcm3 as util
import abc
import cftime
import os
import time
import pathlib

input_file = util.generate_input(str(pathlib.Path(__file__).parent.absolute()) + "/resources/noresm_input")


class NorESMDS(proc.ModelDS):
    MONTHS = ['ja', 'fb', 'mr', 'ar', 'my', 'jn', 'jl', 'ag', 'sp', 'ot', 'nv', 'dc']
    
    def __init__(self, experiment, start_year, end_year, month_list, verbose, logger):
        super(NorESMDS, self).__init__(verbose, logger)
        self.lon = None
        self.lat = None
        self.z = None
        self.lon_b = None
        self.lat_b = None
        self.z_b = None
        self.t = None
        self.lsm = None
        self.start_year = start_year
        self.end_year = end_year
        self.experiment = experiment
        if month_list is None:
            self.months = None
        elif month_list == "full":
            self.months = util.months_to_number(self.MONTHS)
        else:
            self.months = util.months_to_number(month_list)
        self.import_data()
        self.import_coordinates()
    
    @abc.abstractmethod
    def import_data(self):
        pass
    
    @abc.abstractmethod
    def import_coordinates(self):
        print("____ Coordinates imported in the NorESMDS instance.")
        self.guess_bounds()
    
    def get(self, data, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
            mode_z=None, value_z=None, mode_t=None, value_t=None, new_start_year=None, new_end_year=None,
            new_month_list=None):
        
        # if type(data_array) is not proc.GeoDataArray:
        #   convert_to_GeoDataArray(data_array)
        
        geo_da = proc.GeoDataArray(data, ds=self)  # add the GeoDataArray wrapper
        geo_da = zone.import_coordinates_from_data_array(geo_da.data).compact(geo_da)
        
        if any([new_start_year is not None, new_end_year is not None, new_month_list is not None]):
            print("____ Truncation to new time coordinates.")
        
        try:
            if new_start_year is not None and new_start_year < self.start_year:
                raise ValueError("**** The new start year is smaller than the imported one.")
            elif new_end_year is not None and new_end_year > self.end_year:
                raise ValueError("**** The new end year is larger than the imported one.")
            elif new_start_year is None and new_end_year is not None:
                geo_da.crop_years(self.start_year, new_end_year)
            elif new_start_year is not None and new_end_year is None:
                geo_da.crop_years(new_start_year, self.end_year)
            elif new_start_year is not None and new_end_year is not None:
                geo_da.crop_years(new_start_year, new_end_year)
            else:
                pass
            
            if new_month_list is not None and self.months is None:
                raise ValueError(f"**** The month cropping is not available with {type(self)}.")
            elif new_month_list is not None and \
                    not all(
                        month in util.months_to_number(self.months) for month in util.months_to_number(new_month_list)):
                raise ValueError("**** The new month list includes months not yet imported.")
            elif new_month_list is not None:
                geo_da.crop_months(new_month_list)
            else:
                pass
        
        except ValueError as error:
            print(error)
            print("____ The crop was not performed.")
        
        geo_da.get_lon(mode_lon, value_lon)
        geo_da.get_lat(mode_lat, value_lat)
        geo_da.get_z(mode_z, value_z)
        geo_da.get_t(mode_t, value_t)
        # geo_da.fit_coordinates_to_data() Is it still useful?
        
        return geo_da


# **************
# MONTH DATASETS
# **************

class NorESMRDS(NorESMDS):
    
    def __init__(self, experiment, start_year, end_year, file_name, month_list, verbose, logger):
        self.buffer_name = "None"
        self.buffer_array = None
        self.file_name = file_name
        self.paths = []
        super(NorESMRDS, self).__init__(experiment, start_year, end_year, month_list, verbose, logger)
        
        try:
            self.sample_data = xr.open_dataset(self.paths[0])
        except IndexError as error:
            print("No dataset to import. Please check again the import options.")
            raise error
        except FileNotFoundError as error:
            print("The file was not found. Data importation aborted.")
            raise error
    
    def import_data(self):
        print(f"__ Importing {type(self)}")
        print(f"____ Paths generated for {self.experiment} between years {self.start_year} and {self.end_year}.")
        try:
            path = input_file[self.experiment][1]
            self.paths = [f"{path}{self.file_name}{year:04d}-{month:02d}.nc"
                          for year in np.arange(int(self.start_year), int(self.end_year) + 1)
                          for month in self.months]
            for path in self.paths:
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"** {path} was not found. Data import aborted.")
            print("____ Import succeeded.")
        except KeyError as error:
            print("**** This experiment was not found in \"Experiment_to_filename\". Data import aborted.")
            raise error
    
    def import_coordinates(self):
        super(NorESMRDS, self).import_coordinates()
        self.t = None


class OCNMDS(NorESMRDS):
    """
    micom.hm
    """
    
    def __init__(self, experiment, start_year, end_year, month_list="full", verbose=False, logger="print"):
        expt_id = input_file[experiment][0]
        file_name = f"ocn/hist/{expt_id}.micom.hm."
        self.grid = xr.open_dataset(input_file[experiment][2])
        super(OCNMDS, self).__init__(experiment, start_year, end_year, file_name=file_name, month_list=month_list,
                                     verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lon, self.lat = sort_coordinates(self.grid.plon.values, self.grid.plat.values)
        self.lon_p, self.lat_p = cycle_coordinates(self.lon, self.lat)
        self.transform_matrix = get_transform_matrix(self.grid.plon.values)
        super(OCNMDS, self).import_coordinates()
    
    def sst(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None, mode_t=None,
            value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing SST.")
        return self.get(xr.open_mfdataset(self.paths, combine='by_coords').sst.rename({'time': 't'}), zone,
                        mode_lon, value_lon, mode_lat, value_lat, None, None, mode_t, value_t,
                        new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)


# *************
# LAND-SEA MASK
# *************

# class NorESMLSM(proc.LSM):
#
#     def __init__(self):
#         super(NorESMLSM, self).__init__()
#
#     def get_lsm(self, lsm_name):
#         ds_lsm = xr.open_dataset(util.path2lsm[lsm_name])
#         self.lon = ds_lsm.longitude.values
#         self.lat = ds_lsm.latitude.values
#         self.depth = ds_lsm.depthdepth.values
#         self.level = ds_lsm.depthlevel.values
#         self.lsm2d = ds_lsm.lsm.values
#         self.mask2d = (self.lsm2d - 1) * -1
#
#     def fit_lsm_ds(self, ds):
#         # Should check if longitudes are equal
#         if self.depth is None:
#             print("The lsm haven't been imported yet. Calling ls_from_ds instead")
#             self.lsm_from_ds(ds)
#         else:
#             self.lon, self.lat, self.z = ds.longitude.values, ds.latitude.values, ds.depth.values
#             lsm3d = np.zeros((len(self.lon), len(self.lat), len(self.z)))
#             for i in range(len(self.z)):
#                 lsm3d[:, :, i] = np.ma.masked_less(self.depth, self.z[i])
#             self.lsm3d = lsm3d
#             self.mask3d = (lsm3d - 1) * -1
#
#     def lsm_from_ds(self, ds):
#         pass


# class NorESMGrid(proc.Grid):
#
#     def __init__(self):
#         super(NorESMGrid, self).__init__()


# UTIL METHODS

def get_transform_matrix(lon):
    return np.argsort(lon)


def sort_coordinates(lon, lat):
    new_lon, new_lat = np.zeros(lon.shape), np.zeros(lat.shape)
    transform_matrix = get_transform_matrix(lon)
    
    for i in range(lon.shape[0]):
        for j in range(lon.shape[1]):
            new_lon[i, j] = lon[i, transform_matrix[i, j]]
            new_lat[i, j] = lat[i, transform_matrix[i, j]]
    return new_lon, new_lat


def cycle_coordinates(lon, lat):
    n_x, n_y = lon.shape
    new_lon, new_lat = sort_coordinates(lon, lat)
    
    lon_p, lat_p = np.zeros((n_x, n_y + 1)), np.zeros((n_x, n_y + 1))
    
    lon_p[:, :n_y], lat_p[:, :n_y] = new_lon, new_lat
    lon_p[:, n_y], lat_p[:, n_y] = lon_p[:, 0] + 360, lat_p[:, 0]
    
    return lon_p, lat_p
