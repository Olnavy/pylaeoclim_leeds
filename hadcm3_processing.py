import processing as proc
import zones
import numpy as np
import xarray as xr
import util_hadcm3 as util
import abc


class HadCM3DS(proc.ModelDS):
    MONTHS = ['ja', 'fb', 'mr', 'ar', 'my', 'jn', 'jl', 'ag', 'sp', 'ot', 'nv', 'dc']

    def __init__(self, experiment, start_year, end_year, import_type, verbose=False):
        super(HadCM3DS, self).__init__(verbose)
        self.ds_path = []
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
            self.import_data(path, import_type)
        except KeyError as error:
            print("This experiment was not found in \"Experiment_to_filename\". Data importation aborted.")
            print(error)

    @abc.abstractmethod
    def import_data(self, path, import_type):
        pass


class OCNMDS(HadCM3DS):

    def __init__(self, experiment, start_year, end_year, months_list=None):
        super(OCNMDS, self).__init__(experiment, start_year, end_year, import_type="monthly", verbose=False)
        if months_list is None:
            self.months = self.MONTHS
        else:
            self.months = months_list
        self.value_stored_name = "None"
        self.value_stored = None  # Allow to store the last value computed

    def import_data(self, path, import_type):

        file_path = ""

        try:
            for year in np.arange(int(self.start_year), int(self.end_year) + 1):

                # Formatting the year to be a 10 digit
                fyear = f"{year:10d}"
                year_path = f"{path}xosfbo#pf{fyear}"
                year_data = []

                for month in self.months:
                    file_path = f"{year_path}{month}+.nc"
                    year_data.append(file_path)

                self.ds_path.append(year_data)

        except FileNotFoundError as error:
            print(f"The file {file_path} was not found. Data importation aborted.")
            print(error)

    def sst(self, zone=zones.NoZone()):
        data = np.zeros((len(self.ds_path), len(self.ds_path[0]), len(self.lat), len(self.lon)))

        for year in range(len(self.ds_path)):
            for month in range(len(self.ds_path[0])):
                ds = xr.open_dataset(self.ds_path[year][month])
                data[year, month] = zone.compact(ds.temp_mm_uo.isel(t=0).isel(unspecified=0).values.shape)

        self.value_stored_name = "SST"
        self.value_stored = data

        return data

    def sst_mean(self, zone=zones.NoZone()):

        if self.value_stored_name == "SST":
            return proc.ModelDS.mean(np.mean(self.value_stored, axis=1))
        else:
            return proc.ModelDS.mean(np.mean(self.sst(zone), axis=1))

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
