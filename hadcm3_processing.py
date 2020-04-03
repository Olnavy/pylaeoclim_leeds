import processing as proc
import zones
import numpy as np
import xarray as xr
import util_hadcm3 as util
import abc
import cftime


class HadCM3DS(proc.ModelDS):
    MONTHS = ['ja', 'fb', 'mr', 'ar', 'my', 'jn', 'jl', 'ag', 'sp', 'ot', 'nv', 'dc']
    
    def __init__(self, experiment, start_year, end_year, month_list, verbose, logger):
        super(HadCM3DS, self).__init__(verbose, logger)
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
        if month_list is None:
            self.months = None
        elif month_list == "full":
            self.months = self.MONTHS
        else:
            self.months = month_list
        
        try:
            if isinstance(self, HadCM3TS):
                path = util.path2expts[experiment]
                self.import_data(path, experiment)
            else:
                path = util.path2expds[experiment]
                self.import_data(path, experiment)
        except KeyError as error:
            print("This experiment was not found in \"Experiment_to_filename\". Data importation aborted.")
            print(error)
    
    @abc.abstractmethod
    def import_data(self, path, experiment):
        pass
    
    def get(self, data, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
            mode_z=None, value_z=None, mode_t=None, value_t=None, new_start_year=None, new_end_year=None,
            new_month_list=None):
        
        # if type(data_array) is not proc.GeoDataArray:
        #   convert_to_GeoDataArray(data_array)
        
        data_array = proc.GeoDataArray(data)  # add the GeoDataArray wrapper
        try:
            if new_start_year is not None and new_start_year <= self.start_year:
                raise ValueError("The new start year is smaller than the imported one.")
            elif new_end_year is not None and new_end_year >= self.end_year:
                raise ValueError("The new end year is larger than the imported one.")
            elif new_start_year is None and new_end_year is not None:
                data_array = data_array.truncate_years(self.start_year, new_end_year)
            elif new_start_year is not None and new_end_year is None:
                data_array = data_array.truncate_years(new_start_year, self.end_year)
            elif new_start_year is not None and new_end_year is not None:
                data_array = data_array.truncate_years(new_start_year, new_end_year)
            else:
                pass

            if new_month_list is not None and self.months is None:
                raise ValueError(f"The month truncation is not available with {type(self)}.")
            elif new_month_list is not None \
                and not all(month in util.months_to_number(self.months) for month in util.months_to_number(new_month_list)):
                raise ValueError("The new month list include months not yet imported.")
            elif new_month_list is not None:
                data_array = data_array.truncate_months(new_month_list)
            else:
                pass

        except ValueError as error:
            print(error)
            print("The truncation was aborted.")
            
        data_array = self.get_lon(data_array, mode_lon, value_lon)
        data_array = self.get_lat(data_array, mode_lat, value_lat)
        data_array = self.get_z(data_array, mode_z, value_z)
        data_array = self.get_t(data_array, mode_t, value_t)
        
        return zone.compact(data_array)
    
    @staticmethod
    def get_lon(data_array, mode_lon, value_lon):
        try:
            if mode_lon is None:
                return data_array
            elif mode_lon == "index":
                if value_lon is None:
                    raise ValueError("To use the index mode, please indicate a value_lon.")
                print(f"New longitude value : {data_array.longitude.values[int(value_lon)]}")
                return data_array.isel(longitude=value_lon)
            elif mode_lon == "value":
                if value_lon is None:
                    raise ValueError("To use the value mode, please indicate a value_lon.")
                new_lon = data_array.longitude.values[util.lon_to_index(data_array.longitude.values, value_lon)]
                print(
                    f"New longitude value : {new_lon}")
                # A reprendre sur les autres + tester depassement index.
                return data_array.isel(longitude=util.lon_to_index(data_array.longitude.values, value_lon))
            elif mode_lon == "mean":
                return data_array.mean(dim="longitude")
            else:
                print("Mode wasn't recognized. The data_array was not changed.")
                return data_array
        except ValueError as error:
            print(error)
            print("The DataArray was not changed.")
            return data_array
        except IndexError as error:
            print(error)
            print("The longitude index was out of bound, the DataArray was not changed")
            return data_array
    
    @staticmethod
    def get_lat(data_array, mode_lat, value_lat):
        
        try:
            if mode_lat is None:
                return data_array
            elif mode_lat == "index":
                if value_lat is None:
                    raise ValueError("To use the index mode, please indicate a value_lat.")
                print(f"New latitude value : {data_array.latitude.values[int(value_lat)]}")
                return data_array.isel(latitude=value_lat)
            elif mode_lat == "value":
                if value_lat is None:
                    raise ValueError("To use the value mode, please indicate a value_lat.")
                new_lat = data_array.latitude.values[util.lat_to_index(data_array.latitude.values, value_lat)]
                print(
                    f"New latitude value : {new_lat}")
                return data_array.isel(latitude=util.lat_to_index(data_array.latitude.values, value_lat))
            elif mode_lat == "mean":
                return data_array.mean(dim="latitude")
            else:
                print("Mode wasn't recognized. The data_array was not changed.")
                return data_array
        except ValueError as error:
            print(error)
            print("The DataArray was not changed.")
            return data_array
        except IndexError as error:
            print(error)
            print("The latitude index was out of bound, the DataArray was not changed")
            return data_array
    
    @staticmethod
    def get_z(data_array, mode_z, value_z):
        
        try:
            if mode_z is None:
                return data_array
            elif mode_z == "index":
                if value_z is None:
                    raise ValueError("To use the index mode, please indicate a value_z.")
                print(f"New z value : {data_array.z.values[int(value_z)]}")
                return data_array.isel(z=value_z)
            elif mode_z == "value":
                if value_z is None:
                    raise ValueError("To use the value mode, please indicate a value_z.")
                print(f"New z value : {data_array.z.values[util.z_to_index(data_array.z.values, value_z)]}")
                return data_array.isel(z=util.z_to_index(data_array.z.values, value_z))
            elif mode_z == "mean":
                return data_array.mean(dim="z")
            else:
                print("Mode wasn't recognized. The data_array was not changed.")
                return data_array
        except ValueError as error:
            print(error)
            print("The DataArray was not changed.")
            return data_array
        except IndexError as error:
            print(error)
            print("The z index was out of bound, the DataArray was not changed")
            return data_array
    
    @staticmethod
    def get_t(data_array, mode_t, value_t):
        
        try:
            if mode_t is None:
                return data_array
            elif mode_t == "index":
                if value_t is None:
                    raise ValueError("To use the index mode, please indicate a value_t.")
                print(f"New t value : {data_array.t.values[int(value_t)]}")
                return data_array.isel(t=value_t)
            elif mode_t == "value":
                if value_t is None:
                    raise ValueError("To use the value mode, please indicate a value_t.")
                print(f"New t value : {data_array.t.values[util.t_to_index(data_array.t.values, value_t)]}")
                # A reprendre sur les autres + tester depassement index.
                return data_array.isel(t=util.t_to_index(data_array.t.values, value_t))
            elif mode_t == "mean":
                return data_array.mean(dim="t")
            else:
                print("Mode wasn't recognized. The data_array was not changed.")
                return data_array
        except ValueError as error:
            print(error)
            print("The DataArray was not changed.")
            return data_array
        except IndexError as error:
            print(error)
            print("The t index was out of bound, the DataArray was not changed")
            return data_array
    
    def convert_to_geodataarray(self):
        pass


# **************
# MONTH DATASETS
# **************

class OCNMDS(HadCM3DS):
    """
    PF
    """
    
    def __init__(self, experiment, start_year, end_year, month_list="full", verbose=False, logger="print"):
        
        super(OCNMDS, self).__init__(experiment, start_year, end_year, month_list, verbose, logger)
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
    
    def sst(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None, mode_t=None,
            value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        
        data_array = xr.open_mfdataset(self.paths).temp_mm_uo.isel(unspecified=0).drop("unspecified")
        return self.get(data_array, zone, mode_lon, value_lon, mode_lat, value_lat, None, None, mode_t, value_t,
                        new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)
    
    def temperature(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                    mode_z=None, value_z=None, mode_t=None, value_t=None, new_start_year=None, new_end_year=None,
                    new_month_list=None):
        
        data_array = xr.open_mfdataset(self.paths).temp_mm_dpth.rename({'depth_1': 'z'})
        return self.get(data_array, zone, mode_lon, value_lon, mode_lat, value_lat, mode_z, value_z, mode_t, value_t,
                        new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)


# ***********
# TIME SERIES
# ***********


class HadCM3TS(HadCM3DS):
    
    def __init__(self, experiment, start_year, end_year, file_name, month_list, verbose, logger):
        
        self.data = None
        self.buffer_name = "None"
        self.buffer_array = None
        self.file_name = file_name
        super(HadCM3TS, self).__init__(experiment, start_year, end_year, month_list, verbose, logger)
    
    def import_data(self, path, experiment):
        
        try:
            self.data = xr.open_dataset(f"{path}{experiment}.{self.file_name}.nc")
            # The where+lamda structure is not working (GitHub?) so each steps are done individually
            # .where(lambda x: x.t >= cftime.Datetime360Day(self.start_year, 1, 1), drop=True) \
            # .where(lambda x: x.t >= cftime.Datetime360Day(self.end_year, 12, 30), drop=True)
            # .where(lambda x: x.t.month in util.months_to_number(self.months), drop=True)
            self.data = self.data.where(self.data.t >= cftime.Datetime360Day(self.start_year, 1, 1), drop=True)
            self.data = self.data.where(self.data.t <= cftime.Datetime360Day(self.end_year, 12, 30), drop=True)
            self.data = proc.filter_months(self.data, self.months)
            # data is a xarray.Dataset -> not possible to use xarray.GeoDataArray methods. How to change that?
        
        except FileNotFoundError as error:
            print("The file was not found. Data importation aborted.")
            print(error)


class SATMTS(HadCM3TS):
    
    def __init__(self, experiment, start_year, end_year, month_list="full", verbose=False, logger="print"):
        self.data = None
        super(SATMTS, self).__init__(experiment, start_year, end_year, file_name="tempsurf.monthly",
                                     month_list=month_list, verbose=verbose, logger=logger)
    
    def sat(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None, mode_t=None,
            value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        data_array = self.data.temp_mm_srf.isel(surface=0).drop("surface")
        return self.get(data_array, zone, mode_lon, value_lon, mode_lat, value_lat, None, None, mode_t, value_t,
                        new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)


class SSTATS(HadCM3TS):
    
    def __init__(self, experiment, start_year, end_year, month_list=None, verbose=False, logger="print"):
        self.data = None
        super(SSTATS, self).__init__(experiment, start_year, end_year, file_name="oceantemppg01.annual",
                                     month_list=month_list, verbose=verbose, logger=logger)
    
    def sst(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
            mode_z=None, value_z=None, mode_t=None, value_t=None, new_start_year=None, new_end_year=None,
            new_month_list=None):
        data_array = self.data.temp_ym_dpth.isel(depth_1=0).drop("depth_1")
        return self.get(data_array, zone, mode_lon, value_lon, mode_lat, value_lat, mode_z, value_z, mode_t, value_t,
                        new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)


class MERIDATS(HadCM3TS):
    
    def __init__(self, experiment, start_year, end_year, month_list=None, verbose=False, logger="print"):
        self.data = None
        super(MERIDATS, self).__init__(experiment, start_year, end_year, file_name="merid.annual",
                                       month_list=month_list, verbose=verbose, logger=logger)
    
    def atlantic(self, zone=zones.NoZone(), mode_lat=None, value_lat=None, mode_z=None, value_z=None, mode_t=None,
                 value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        data_array = self.data.Merid_Atlantic.rename({'depth': 'z'})
        return self.get(data_array, zone, None, None, mode_lat, value_lat, mode_z, value_z, mode_t, value_t,
                        new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)
    
    def globalx(self, zone=zones.NoZone(), mode_lat=None, value_lat=None, mode_z=None, value_z=None, mode_t=None,
                value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        data_array = self.data.Merid_Global.rename({'depth': 'z'})
        return self.get(data_array, zone, None, None, mode_lat, value_lat, mode_z, value_z, mode_t, value_t,
                        new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)
    
    def indian(self, zone=zones.NoZone(), mode_lat=None, value_lat=None, mode_z=None, value_z=None, mode_t=None,
               value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        data_array = self.data.Merid_Indian.rename({'depth': 'z'})
        return self.get(data_array, zone, None, None, mode_lat, value_lat, mode_z, value_z, mode_t, value_t,
                        new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)
    
    def pacific(self, zone=zones.NoZone(), mode_lat=None, value_lat=None, mode_z=None, value_z=None, mode_t=None,
                value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        data_array = self.data.Merid_Pacific.rename({'depth': 'z'})
        return self.get(data_array, zone, None, None, mode_lat, value_lat, mode_z, value_z, mode_t, value_t,
                        new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)


# *************
# LAND-SEA MASK
# *************

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
