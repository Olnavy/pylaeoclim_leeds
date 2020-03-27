import processing as proc
import zones
import numpy as np
import xarray as xr
import util_hadcm3 as util
import abc


class HadCM3DS(proc.ModelDS):
    MONTHS = ['ja', 'fb', 'mr', 'ar', 'my', 'jn', 'jl', 'ag', 'sp', 'ot', 'nv', 'dc']

    def __init__(self, experiment, start_year, end_year, verbose=False):
        super(HadCM3DS, self).__init__(verbose)
        self.paths = []
        self.lon = None
        self.lat = None
        self.z = None
        self.lon_b = None
        self.lat_b = None
        self.z_b = None
        self.lsm = None
        self.start_year = start_year
        self.end_year = end_year

        try:
            path = util.path2exp[experiment]
            self.import_data(path, experiment)
        except KeyError as error:
            print("This experiment was not found in \"Experiment_to_filename\". Data importation aborted.")
            print(error)

    @abc.abstractmethod
    def import_data(self, path, experiment):
        pass


class OCNMDS(HadCM3DS):

    def __init__(self, experiment, start_year, end_year, months_list=None):
        if months_list is None:
            self.months = self.MONTHS
        else:
            self.months = months_list

        super(OCNMDS, self).__init__(experiment, start_year, end_year, verbose=False)
        self.buffer_name = "None"
        self.buffer_array = None

    def import_data(self, path, experiment):

        try:
            self.paths = [f"{path}pf/{experiment}o#pf{year:09d}{month}+.nc"
                      for year in np.arange(int(self.start_year), int(self.end_year) + 1)
                      for month in self.months]

        except FileNotFoundError as error:
            print("One of the file was not found. Data importation aborted.")
            print(error)

    def import_coordinates(self):
        self.lon = self.buffer_array.longitude.values
        self.lon_b = util.coordinate_bounds(self.lon)
        self.lat = self.buffer_array.latitude.values
        self.lat_b = util.coordinate_bounds(self.lat)
        self.z = None

    def sst(self, zone=zones.NoZone()):
    
        if self.buffer_name=="sst":
            return self.buffer_array
        else :
            data = xr.open_mfdataset(self.paths).temp_mm_uo.isel(unspecified=0)
            # Store the value in the class buffer
            self.buffer_name = "sst"
            self.buffer_array = data
            # import coordinates
            self.import_coordinates()
            return data

    
    def sst_mean(self, zone=zones.NoZone()):

        if self.buffer_name == "sst":
            return proc.ModelDS.mean(self.buffer_array)
        else:
            return proc.ModelDS.mean(self.sst(zone))
          

    def sst_mean_lon(self, zone=zones.NoZone()):

        if self.value_stored_name == "SST":
            return proc.ModelDS.mean_lon(np.mean(self.value_stored, axis=1))
        else:
            return proc.ModelDS.mean_lon(np.mean(self.sst(zone), axis=1))

    def sst_mean_lat(self, zone=zones.NoZone()):

        if self.value_stored_name == "SST":
            return proc.ModelDS.mean_lat(np.mean(self.value_stored, axis=1))
        else:
            return proc.ModelDS.mean_lat(np.mean(self.sst(zone), axis=1))

    def sst_serie(self, zone=zones.NoZone()):

        if self.value_stored_name == "SST":
            return proc.ModelDS.serie(np.mean(self.value_stored, axis=1))
        else:
            return proc.ModelDS.serie(np.mean(self.sst(zone), axis=1))

    def sst_serie_lon(self, zone=zones.NoZone()):

        if self.value_stored_name == "SST":
            return proc.ModelDS.serie_lon(np.mean(self.value_stored, axis=1))
        else:
            return proc.ModelDS.serie_lat(np.mean(self.sst(zone), axis=1))

    def sst_global_mean(self, zone=zones.NoZone()):

        if self.value_stored_name == "SST":
            return proc.ModelDS.global_mean(np.mean(self.value_stored, axis=1))
        else:
            return proc.ModelDS.global_mean(np.mean(self.sst(zone), axis=1))


class HadCM3LSM(proc.LSM):

    def __init__(self):
        super(HadCM3LSM, self).__init__()

    def get_lsm(self, lsm_name):
        ds_lsm = xr.open_dataset(util.path2lsm[lsm_name])
        self.lon = ds_lsm.longitude.values
        self.lat = ds_lsm.latitude.values
        self.depth = ds_lsm.depthdepth.values
        self.level = ds_lsm.depthlevel.values
        self.lsm2d = ds_lsm.lsm.values
        self.mask2d = (self.lsm2d - 1) * -1

    def fit_lsm_ds(self, ds):
        # Should check if longitudes are equal
        if self.depth is None:
            print("The lsm haven't been imported yet. Calling ls_from_ds instead")
            self.lsm_from_ds(ds)
        else:
            self.lon, self.lat, self.z = ds.longitude.values, ds.latitude.values, ds.depth.values
            lsm3d = np.zeros((len(self.lon), len(self.lat), len(self.z)))
            for i in range(len(self.z)):
                lsm3d[:, :, i] = np.ma.masked_less(self.depth, self.z[i])
            self.lsm3d = lsm3d
            self.mask3d = (lsm3d - 1) * -1

    def lsm_from_ds(self, ds):
        pass
