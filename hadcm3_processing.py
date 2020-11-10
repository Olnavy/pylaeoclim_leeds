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

input_file = util.generate_input(str(pathlib.Path(__file__).parent.absolute()) + "/resources/hadcm3_input")


class HadCM3DS(proc.ModelDS):
    MONTHS = ['ja', 'fb', 'mr', 'ar', 'my', 'jn', 'jl', 'ag', 'sp', 'ot', 'nv', 'dc']
    
    def __init__(self, experiment, start_year, end_year, month_list, verbose, logger):
        super(HadCM3DS, self).__init__(verbose, logger)
        self.t = None
        self.lsm = None
        self.start_year = start_year
        self.end_year = end_year
        self.experiment = experiment
        if month_list is None:
            self.months = None
        elif month_list == "full":
            self.months = self.MONTHS
        else:
            self.months = month_list
        
        self.import_data()
        self.import_coordinates()
    
    @staticmethod
    def transform(array_r):
        pass
    
    @abc.abstractmethod
    def import_data(self):
        pass
    
    @abc.abstractmethod
    def import_coordinates(self):
        print("____ Coordinates imported in the HadCM3DS dataset.")
    
    def get(self, data, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
            mode_z=None, value_z=None, mode_t=None, value_t=None, new_start_year=None, new_end_year=None,
            new_month_list=None):
        
        geo_da = proc.GeoDataArray(data, ds=self, transform=self.transform)  # add the GeoDataArray wrapper
        # geo_da = zone.import_coordinates(geo_da).compact(geo_da)
        geo_da = zone.compact(geo_da)
        
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
    
    def extend(self, geo_data_array):
        # ????
        pass


# ************
# RAW DATASETS
# ************

class HadCM3RDS(HadCM3DS):
    
    def __init__(self, experiment, start_year, end_year, file_name, month_list, verbose, logger):
        self.sample_data = None
        self.file_name = file_name
        self.paths = []
        super(HadCM3RDS, self).__init__(experiment, start_year, end_year, month_list, verbose, logger)
    
    def import_data(self):
        print(f"__ Importing {type(self)}")
        print(f"____ Paths generated for {self.experiment} between years {self.start_year} and {self.end_year}.")
        
        # ADD A METHOD TO CHECK THE VALID RANGE
        #
        # if min(self.data.t.values).year > self.start_year or max(self.data.t.values).year < self.end_year:
        #     raise ValueError(f"Inavlid start_year or end_year. Please check that they fit the valid range\n"
        #                      f"Valid range : start_year = {min(self.data.t.values).year}, "
        #                      f"end_year = {max(self.data.t.values).year}")
        
        try:
            path = input_file[self.experiment][1]
            if self.months is not None:
                self.paths = [f"{path}{self.file_name}{year:09d}{month}+.nc"
                              for year in np.arange(int(self.start_year), int(self.end_year) + 1)
                              for month in self.months]
            else:
                self.paths = [f"{path}{self.file_name}{year:09d}c1+.nc"
                              for year in np.arange(int(self.start_year), int(self.end_year) + 1)]
            for path in self.paths:
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"** {path} was not found. Data import aborted.")
            print("____ Import succeeded.")
        except KeyError as error:
            print("**** This experiment was not found in \"Experiment_to_filename\". Data import aborted.")
            raise error
        
        try:
            self.sample_data = xr.open_dataset(self.paths[0])
        except IndexError as error:
            print("No dataset to import. Please check again the import options.")
            raise error
        except FileNotFoundError as error:
            print("The file was not found. Data importation aborted.")
            raise error
    
    def import_coordinates(self):
        super(HadCM3RDS, self).import_coordinates()
    
    def __repr__(self):
        return f"{util.print_coordinates('lon', self.lon)}; {util.print_coordinates('lon_p', self.lon_p)}\n" \
               f"{util.print_coordinates('lonb', self.lonb)}; {util.print_coordinates('lonb_p', self.lonb_p)}\n" \
               f"{util.print_coordinates('lons', self.lons)}; {util.print_coordinates('lons_p', self.lons_p)}\n" \
               f"{util.print_coordinates('lat', self.lat)}; {util.print_coordinates('lat_p', self.lat_p)}\n" \
               f"{util.print_coordinates('latb', self.latb)}; {util.print_coordinates('latb_p', self.latb_p)}\n" \
               f"{util.print_coordinates('lats', self.lats)}; {util.print_coordinates('lats_p', self.lats_p)}\n" \
               f"{util.print_coordinates('z', self.z)}; {util.print_coordinates('z_p', self.z_p)}\n" \
               f"{util.print_coordinates('zb', self.zb)}; {util.print_coordinates('zb_p', self.zb_p)}\n" \
               f"{util.print_coordinates('zs', self.zs)}; {util.print_coordinates('zs_p', self.zs_p)}\n" \
               f"{util.print_coordinates('t', self.t)}\n" \
               f"DATA: {self.sample_data}"

class ATMUPMDS(HadCM3RDS):
    """
    PC
    """
    
    def __init__(self, experiment, start_year, end_year, month_list="full", verbose=False, logger="print"):
        expt_id = input_file[experiment][0]
        file_name = f"pcpd/{expt_id}a#pc"
        super(ATMUPMDS, self).__init__(experiment, start_year, end_year, file_name=file_name, month_list=month_list,
                                       verbose=verbose, logger=logger)
    
    @staticmethod
    def transform(array_r):
        array = array_r
        if "longitude" in array.dims:
            array = xr.concat([array, array.isel(longitude=0)], dim="longitude")
        if "longitude_1" in array.dims:
            array = xr.concat([array, array.isel(longitude_1=0)], dim="longitude_1")
        if "latitude" in array.dims:
            array = xr.concat([array.isel(latitude=0), array, array.isel(latitude=-1)], dim="latitude")
        return array.transpose(*array_r.dims)
    
    def import_coordinates(self):
        self.lon, self.lonb = np.sort(self.sample_data.longitude.values), np.sort(self.sample_data.longitude_1.values)
        self.lons = self.lonb[1:] - self.lonb[0:-1]
        self.lon_p = np.append(self.lon, self.lon[-1] + self.lons[-1])
        self.lonb_p = np.append(self.lonb, 2 * self.lonb[-1] - self.lonb[-2])
        self.lons_p = self.lonb_p[1:] - self.lonb_p[0:-1]
        
        self.lat, self.latb = np.sort(self.sample_data.latitude.values), np.sort(self.sample_data.latitude_1.values)
        self.lats = self.latb[1:] - self.latb[0:-1]
        self.lat_p = np.concatenate(([-90], self.lat, [90]))
        self.latb_p = self.latb
        self.lats_p = self.latb_p[1:] - self.latb_p[0:-1]
        
        self.z = np.sort(self.sample_data.p.values)
        self.zs = self.z[1:] - self.z[0:-1]
        self.zb = np.concatenate(
            (self.z[:-1] - self.zs / 2, [self.z[-1] - self.zs[-1] / 2], [self.z[-1] + self.zs[-1] / 2]))
        self.z_p = self.z
        self.zb_p = self.zb
        self.zs_p = self.zb_p[1:] - self.zb_p[0:-1]
        
        self.t = [cftime.Datetime360Day(year, month, 1)
                  for year in np.arange(int(self.start_year), int(self.end_year) + 1)
                  for month in util.months_to_number(self.months)]
        
        super(ATMUPMDS, self).import_coordinates()


class ATMSURFMDS(HadCM3RDS):
    """
    PD
    """
    
    def __init__(self, experiment, start_year, end_year, month_list="full", verbose=False, logger="print"):
        expt_id = input_file[experiment][0]
        file_name = f"pcpd/{expt_id}a#pd"
        super(ATMSURFMDS, self).__init__(experiment, start_year, end_year, file_name=file_name, month_list=month_list,
                                         verbose=verbose, logger=logger)
    
    @staticmethod
    def transform(array_r):
        array = array_r
        if "longitude" in array.dims:
            array = xr.concat([array, array.isel(longitude=0)], dim="longitude")
        if "longitude_1" in array.dims:
            array = xr.concat([array, array.isel(longitude_1=0)], dim="longitude_1")
        if "latitude_1" in array.dims:
            array.isel(latitude_1=-1).values = array.isel(latitude_1=-2).values
            array = xr.concat([array.isel(latitude_1=0), array, array.isel(latitude_1=-2)], dim="latitude_1")
        return array.transpose(*array_r.dims)
    
    def import_coordinates(self):
        self.lon, self.lonb = np.sort(self.sample_data.longitude.values), np.sort(self.sample_data.longitude_1.values)
        self.lons = self.lonb[1:] - self.lonb[0:-1]
        self.lon_p = np.append(self.lon, self.lon[-1] + self.lons[-1])
        self.lonb_p = np.append(self.lonb, [2 * self.lonb[-1] - self.lonb[-2]])
        self.lons_p = self.lonb_p[1:] - self.lonb_p[0:-1]
        
        self.lat, self.latb = np.sort(self.sample_data.latitude.values), np.sort(self.sample_data.latitude_1.values)
        self.lats = self.latb[1:] - self.latb[0:-1]
        self.lat_p = self.lat
        self.latb_p = np.concatenate(([-90], self.latb, [2 * self.latb[-1] - self.latb[-2]]))
        self.lats_p = self.latb_p[1:] - self.latb_p[0:-1]
        
        self.t = [cftime.Datetime360Day(year, month, 1)
                  for year in np.arange(int(self.start_year), int(self.end_year) + 1)
                  for month in util.months_to_number(self.months)]
        
        super(ATMSURFMDS, self).import_coordinates()
    
    def sat(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
            mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing SAT.")
        return self.get(xr.open_mfdataset(self.paths, combine='by_coords').temp_mm_srf.isel(surface=0), zone,
                        mode_lon, value_lon, mode_lat, value_lat, mode_t, value_t,
                        new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)


class OCNMDS(HadCM3RDS):
    """
    PF
    """
    
    def __init__(self, experiment, start_year, end_year, month_list="full", verbose=False, logger="print"):
        expt_id = input_file[experiment][0]
        file_name = f"pf/{expt_id}o#pf"
        super(OCNMDS, self).__init__(experiment, start_year, end_year, file_name=file_name, month_list=month_list,
                                     verbose=verbose, logger=logger)
    
    @staticmethod
    def transform(array_r):
        array = array_r
        if "longitude" in array.dims:
            array = xr.concat([array, array.isel(longitude=0)], dim="longitude")
        if "longitude_1" in array.dims:
            array = xr.concat([array, array.isel(longitude_1=0)], dim="longitude_1")
        if "latitude" in array.dims:
            array.isel(latitude=-1).values = array.isel(latitude=-2).values
            array = xr.concat([array.isel(latitude=np.arange(0, len(array.latitude) - 1, 1)), array.isel(latitude=-2),
                               array.isel(latitude=-2)], dim="latitude")
        if "latitude_1" in array.dims:
            array.isel(latitude_1=-1).values = array.isel(latitude_1=-2).values
            array = xr.concat(
                [array.isel(latitude_1=0), array.isel(latitude_1=np.arange(0, len(array.latitude_1) - 1, 1)),
                 array.isel(latitude_1=-2), array.isel(latitude_1=-2)], dim="latitude_1")
        if "depth_1" in array.dims:
            array = xr.concat([array, array.isel(depth_1=-1)], dim="depth_1")
        return array.transpose(*array_r.dims)
    
    def import_coordinates(self):
        self.lon, self.lonb = np.sort(self.sample_data.longitude.values), np.sort(self.sample_data.longitude_1.values)
        self.lons = self.lonb[1:] - self.lonb[0:-1]
        self.lon_p = np.append(self.lon, self.lon[-1] + self.lons[-1])
        self.lonb_p = np.append(self.lonb, 2 * self.lonb[-1] - self.lonb[-2])
        self.lons_p = self.lonb_p[1:] - self.lonb_p[0:-1]
        
        self.lat, self.latb = np.sort(self.sample_data.latitude.values), np.sort(self.sample_data.latitude_1.values)
        self.lats = self.latb[1:] - self.latb[0:-1]
        self.lat_p = np.append(self.lat, self.lat[-1] + self.lats[-1])
        self.latb_p = np.concatenate(
            ([2 * self.latb[0] - self.latb[1]], self.latb, [2 * self.latb[-1] - self.latb[-2]]))
        self.lats_p = self.latb_p[1:] - self.latb_p[0:-1]
        
        self.z, self.zb = -np.sort(self.sample_data.depth.values), -np.sort(self.sample_data.depth_1.values)
        self.zs = self.zb[1:] - self.zb[0:-1]
        self.z_p = self.z
        self.zb_p = np.append(self.zb, self.zb[-1] + (self.z[-1] - self.zb[-1]) * 2)
        self.zs_p = self.zb_p[1:] - self.zb_p[0:-1]
        
        self.t = [cftime.Datetime360Day(year, month, 1)
                  for year in np.arange(int(self.start_year), int(self.end_year) + 1)
                  for month in util.months_to_number(self.months)]
        super(OCNMDS, self).import_coordinates()
    
    def sst(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None, mode_t=None,
            value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing SST.")
        return self.get(
            xr.open_mfdataset(self.paths, combine='by_coords').temp_mm_uo.isel(unspecified=0).drop("unspecified"), zone,
            mode_lon, value_lon, mode_lat, value_lat, None, None, mode_t, value_t,
            new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)
    
    def temperature(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                    mode_z=None, value_z=None, mode_t=None, value_t=None, new_start_year=None, new_end_year=None,
                    new_month_list=None):
        print("__ Importing temperature.")
        return self.get(xr.open_mfdataset(self.paths, combine='by_coords').temp_mm_dpth.rename({'depth_1': 'z'}), zone,
                        mode_lon, value_lon, mode_lat, value_lat, mode_z, value_z, mode_t, value_t,
                        new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)
    
    def salinity(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                 mode_z=None, value_z=None, mode_t=None, value_t=None, new_start_year=None, new_end_year=None,
                 new_month_list=None):
        print("__ Importing salinity.")
        return self.get(xr.open_mfdataset(self.paths, combine='by_coords').salinity_mm_dpth.rename({'depth_1': 'z'}),
                        zone, mode_lon, value_lon, mode_lat, value_lat, mode_z, value_z, mode_t, value_t,
                        new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)
    
    def htn(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None, mode_t=None,
            value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing net surface heat flux.")
        return self.get(
            xr.open_mfdataset(self.paths, combine='by_coords').HTN_mm_uo.isel(unspecified=0).drop("unspecified"), zone,
            mode_lon, value_lon, mode_lat, value_lat, None, None, mode_t, value_t,
            new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)
    
    def u_velocity(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                   mode_z=None, value_z=None, mode_t=None, value_t=None, new_start_year=None, new_end_year=None,
                   new_month_list=None):
        print("__ Importing meridional (eastward) velocity.")
        return self.get(xr.open_mfdataset(self.paths, combine='by_coords').ucurrTot_mm_dpth.rename({'depth_1': 'z'})
                        .rename({'longitude_1': 'longitude'}).rename({'latitude_1': 'latitude'}),
                        zone, mode_lon, value_lon, mode_lat, value_lat, mode_z, value_z, mode_t, value_t,
                        new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)
    
    def v_velocity(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                   mode_z=None, value_z=None, mode_t=None, value_t=None, new_start_year=None, new_end_year=None,
                   new_month_list=None):
        print("__ Importing zonal (northward) velocity.")
        return self.get(xr.open_mfdataset(self.paths, combine='by_coords').vcurrTot_mm_dpth.rename({'depth_1': 'z'})
                        .rename({'longitude_1': 'longitude'}).rename({'latitude_1': 'latitude'}),
                        zone, mode_lon, value_lon, mode_lat, value_lat, mode_z, value_z, mode_t, value_t,
                        new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)
    
    def velocity(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                 mode_z=None, value_z=None, mode_t=None, value_t=None, new_start_year=None, new_end_year=None,
                 new_month_list=None):
        print("__ Importing zonal and meridional velocities and computing total velocity.")
        return self.get(np.sqrt(
            (xr.open_mfdataset(self.paths, combine='by_coords').vcurrTot_mm_dpth.rename({'depth_1': 'z'})
             .rename({'longitude_1': 'longitude'}).rename({'latitude_1': 'latitude'})) ** 2 +
            (xr.open_mfdataset(self.paths, combine='by_coords').vcurrTot_mm_dpth.rename({'depth_1': 'z'})
             .rename({'longitude_1': 'longitude'}).rename({'latitude_1': 'latitude'})) ** 2),
            zone, mode_lon, value_lon, mode_lat, value_lat, mode_z, value_z, mode_t, value_t,
            new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)


class OCNYDS(HadCM3RDS):
    """
    PG
    """
    
    def __init__(self, experiment, start_year, end_year, verbose=False, logger="print"):
        expt_id = input_file[experiment][0]
        file_name = f"pg/{expt_id}o#pg"
        super(OCNYDS, self).__init__(experiment, start_year, end_year, file_name=file_name,
                                     verbose=verbose, logger=logger, month_list=None)
    
    @staticmethod
    def transform(array_r):
        array = array_r
        if "longitude" in array.dims:
            array = xr.concat([array, array.isel(longitude=0)], dim="longitude")
        if "longitude_1" in array.dims:
            array = xr.concat([array, array.isel(longitude_1=0)], dim="longitude_1")
        if "latitude" in array.dims:
            array.isel(latitude=-1).values = array.isel(latitude=-2).values
            array = xr.concat([array.isel(latitude=np.arange(0, len(array.latitude) - 1, 1)), array.isel(latitude=-2),
                               array.isel(latitude=-2)], dim="latitude")
        if "latitude_1" in array.dims:
            array.isel(latitude_1=-1).values = array.isel(latitude_1=-2).values
            array = xr.concat(
                [array.isel(latitude_1=0), array.isel(latitude_1=np.arange(0, len(array.latitude_1) - 1, 1)),
                 array.isel(latitude_1=-2), array.isel(latitude_1=-2)], dim="latitude_1")
        return array.transpose(*array_r.dims)
    
    def import_coordinates(self):
        self.lon, self.lonb = np.sort(self.sample_data.longitude.values), np.sort(self.sample_data.longitude_1.values)
        self.lons = self.lonb[1:] - self.lonb[0:-1]
        self.lon_p = np.append(self.lon, self.lon[-1] + self.lons[-1])
        self.lonb_p = np.append(self.lonb, 2 * self.lonb[-1] - self.lonb[-2])
        self.lons_p = self.lonb_p[1:] - self.lonb_p[0:-1]
        
        self.lat, self.latb = np.sort(self.sample_data.latitude.values), np.sort(self.sample_data.latitude_1.values)
        self.lats = self.latb[1:] - self.latb[0:-1]
        self.lat_p = np.append(self.lat, self.lat[-1] + self.lats[-1])
        self.latb_p = np.concatenate(
            ([2 * self.latb[0] - self.latb[1]], self.latb, [2 * self.latb[-1] - self.latb[-2]]))
        self.lats_p = self.latb_p[1:] - self.latb_p[0:-1]
        
        self.z, self.zb = -np.sort(self.sample_data.depth.values), -np.sort(self.sample_data.depth_1.values)
        self.zs = self.zb[1:] - self.zb[0:-1]
        self.z_p = self.z
        self.zb_p = self.zb
        self.zs_p = self.zb_p[1:] - self.zb_p[0:-1]
        
        self.t = [cftime.Datetime360Day(year, 6, 1) for year in np.arange(int(self.start_year), int(self.end_year) + 1)]
        
        super(OCNYDS, self).import_coordinates()
    
    def temperature(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                    mode_z=None, value_z=None, mode_t=None, value_t=None, new_start_year=None, new_end_year=None):
        print("__ Importing temperature.")
        return self.get(xr.open_mfdataset(self.paths, combine='by_coords').temp_ym_dpth.rename({'depth_1': 'z'}),
                        zone, mode_lon, value_lon, mode_lat, value_lat, mode_z, value_z, mode_t, value_t,
                        new_start_year=new_start_year, new_end_year=new_end_year)
    
    def salinity(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                 mode_z=None, value_z=None, mode_t=None, value_t=None, new_start_year=None, new_end_year=None):
        print("__ Importing salinity.")
        return self.get(xr.open_mfdataset(self.paths, combine='by_coords').salinity_ym_dpth.rename({'depth_1': 'z'}),
                        zone, mode_lon, value_lon, mode_lat, value_lat, mode_z, value_z, mode_t, value_t,
                        new_start_year=new_start_year, new_end_year=new_end_year)
    
    def u_velocity(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                   mode_z=None, value_z=None, mode_t=None, value_t=None, new_start_year=None, new_end_year=None):
        print("__ Importing meridional (eastward) velocity.")
        return self.get(xr.open_mfdataset(self.paths, combine='by_coords').ucurrTot_mm_dpth.rename({'depth_1': 'z'})
                        .rename({'longitude_1': 'longitude'}).rename({'latitude_1': 'latitude'}),
                        zone, mode_lon, value_lon, mode_lat, value_lat, mode_z, value_z, mode_t, value_t,
                        new_start_year=new_start_year, new_end_year=new_end_year)
    
    def v_velocity(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                   mode_z=None, value_z=None, mode_t=None, value_t=None, new_start_year=None, new_end_year=None):
        print("__ Importing zonal (northward) velocity.")
        return self.get(xr.open_mfdataset(self.paths, combine='by_coords').vcurrTot_mm_dpth.rename({'depth_1': 'z'})
                        .rename({'longitude_1': 'longitude'}).rename({'latitude_1': 'latitude'}),
                        zone, mode_lon, value_lon, mode_lat, value_lat, mode_z, value_z, mode_t, value_t,
                        new_start_year=new_start_year, new_end_year=new_end_year)
    
    def velocity(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                 mode_z=None, value_z=None, mode_t=None, value_t=None, new_start_year=None, new_end_year=None):
        print("__ Importing zonal and meridional velocities and computing total velocity.")
        return self.get(np.sqrt(
            (xr.open_mfdataset(self.paths, combine='by_coords').vcurrTot_mm_dpth.rename({'depth_1': 'z'})
             .rename({'longitude_1': 'longitude'}).rename({'latitude_1': 'latitude'})) ** 2 +
            (xr.open_mfdataset(self.paths, combine='by_coords').vcurrTot_mm_dpth.rename({'depth_1': 'z'})
             .rename({'longitude_1': 'longitude'}).rename({'latitude_1': 'latitude'})) ** 2),
            zone, mode_lon, value_lon, mode_lat, value_lat, mode_z, value_z, mode_t, value_t,
            new_start_year=new_start_year, new_end_year=new_end_year)


class LNDMDS(HadCM3RDS):
    """
    PT
    """
    
    def import_coordinates(self):
        self.lon = self.sample_data.longitude.values
        self.lat = self.sample_data.latitude.values
        self.z = self.sample_data.pseudo.values
        self.t = self.sample_data.t.values
        
        super(LNDMDS, self).import_coordinates()
        
        # What to do with pseudo, pseudo_2 and pseudo_3?


# ***********
# TIME SERIES
# ***********


class HadCM3TS(HadCM3DS):
    
    def __init__(self, experiment, start_year, end_year, file_name, month_list, verbose, logger):
        
        self.data = None
        self.file_name = file_name
        super(HadCM3TS, self).__init__(experiment, start_year, end_year, month_list, verbose, logger)
    
    def __repr__(self):
        return f"{util.print_coordinates('lon', self.lon)}; {util.print_coordinates('lon_p', self.lon_p)}\n" \
               f"{util.print_coordinates('lonb', self.lonb)}; {util.print_coordinates('lonb_p', self.lonb_p)}\n" \
               f"{util.print_coordinates('lons', self.lons)}; {util.print_coordinates('lons_p', self.lons_p)}\n" \
               f"{util.print_coordinates('lat', self.lat)}; {util.print_coordinates('lat_p', self.lat_p)}\n" \
               f"{util.print_coordinates('latb', self.latb)}; {util.print_coordinates('latb_p', self.latb_p)}\n" \
               f"{util.print_coordinates('lats', self.lats)}; {util.print_coordinates('lats_p', self.lats_p)}\n" \
               f"{util.print_coordinates('z', self.z)}; {util.print_coordinates('z_p', self.z_p)}\n" \
               f"{util.print_coordinates('zb', self.zb)}; {util.print_coordinates('zb_p', self.zb_p)}\n" \
               f"{util.print_coordinates('zs', self.zs)}; {util.print_coordinates('zs_p', self.zs_p)}\n" \
               f"{util.print_coordinates('t', self.t)}\n" \
               f"DATA: {self.data}"
    
    def import_data(self):
        
        path = ""
        try:
            print(
                f"__ Importation of {type(self)} : {self.experiment} between "
                f"years {self.start_year} and {self.end_year}.")
            
            path = input_file[self.experiment][2]
            
            start = time.time()
            self.data = xr.open_dataset(f"{path}{self.experiment}.{self.file_name}.nc")
            print(f"Time elapsed for open_dataset : {time.time() - start}")
            
            if min(self.data.t.values).year > self.start_year or max(self.data.t.values).year < self.end_year:
                raise ValueError(f"Inavlid start_year or end_year. Please check that they fit the valid range\n"
                                 f"Valid range : start_year = {min(self.data.t.values).year}, "
                                 f"end_year = {max(self.data.t.values).year}")
            
            # The where+lamda structure is not working (GitHub?) so each steps are done individually
            # .where(lambda x: x.t >= cftime.Datetime360Day(self.start_year, 1, 1), drop=True) \
            # .where(lambda x: x.t >= cftime.Datetime360Day(self.end_year, 12, 30), drop=True)
            # .where(lambda x: x.t.month in util.months_to_number(self.months), drop=True)
            start = time.time()
            self.data = self.data.where(self.data.t >= cftime.Datetime360Day(self.start_year, 1, 1), drop=True)
            print(f"Time elapsed for crop start year : {time.time() - start}")
            self.data = self.data.where(self.data.t <= cftime.Datetime360Day(self.end_year, 12, 30), drop=True)
            print(f"Time elapsed for crop start and end years : {time.time() - start}")
            self.data = proc.filter_months(self.data, self.months)
            print(f"Time elapsed for crop start and end years and months : {time.time() - start}")
            
            # data is a xarray.Dataset -> not possible to use xarray.GeoDataArray methods. How to change that?
            
            print("____ Import succeeded.")
        
        except FileNotFoundError as error:
            print(f"**** {path}{self.experiment}.{self.file_name}.nc was not found. Data import aborted.")
            raise error
        except KeyError as error:
            print("**** This experiment was not found in \"Experiment_to_filename\". Data importation aborted.")
            raise error
    
    def import_coordinates(self):
        super(HadCM3TS, self).import_coordinates()
        self.t = self.data.t.values
    
    def processing_array(self):
        return util.cycle_lon(self.data.values)


class SAL01MTS(HadCM3TS):
    
    def __init__(self, experiment, start_year, end_year, month_list="full", verbose=False, logger="print"):
        self.data = None
        super(SAL01MTS, self).__init__(experiment, start_year, end_year, file_name="oceansalipf01.monthly",
                                       month_list=month_list, verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lon = self.data.longitude.values
        self.lat = self.data.latitude.values
        
        super(SAL01MTS, self).import_coordinates()
    
    def salinity(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                 mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing sea water salinity at 5m (monthly).")
        return self.get(self.data.salinity_mm_dpth.isel(depth_1=0), zone, mode_lon, value_lon, mode_lat, value_lat,
                        None, None, mode_t, value_t, new_start_year=new_start_year, new_end_year=new_end_year,
                        new_month_list=new_month_list)


class SAL01ATS(HadCM3TS):
    
    def __init__(self, experiment, start_year, end_year, month_list=None, verbose=False, logger="print"):
        self.data = None
        super(SAL01ATS, self).__init__(experiment, start_year, end_year, file_name="oceansalipg01.annual",
                                       month_list=month_list, verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lon = self.data.longitude.values
        self.lat = self.data.latitude.values
        
        super(SAL01ATS, self).import_coordinates()
    
    def salinity(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                 mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing sea water salinity at 5m (annual).")
        return self.get(self.data.salinity_ym_dpth.isel(depth_1=0), zone, mode_lon, value_lon, mode_lat, value_lat,
                        None, None, mode_t, value_t, new_start_year=new_start_year, new_end_year=new_end_year,
                        new_month_list=new_month_list)


class SAL12ATS(HadCM3TS):
    
    def __init__(self, experiment, start_year, end_year, month_list=None, verbose=False, logger="print"):
        self.data = None
        super(SAL12ATS, self).__init__(experiment, start_year, end_year, file_name="oceansalipg12.annual",
                                       month_list=month_list, verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lon = self.data.longitude.values
        self.lat = self.data.latitude.values
        
        super(SAL12ATS, self).import_coordinates()
    
    def salinity(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                 mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing sea water salinity at 666m (annual).")
        return self.get(self.data.salinity_ym_dpth.isel(depth_1=0), zone, mode_lon, value_lon, mode_lat, value_lat,
                        None, None, mode_t, value_t, new_start_year=new_start_year, new_end_year=new_end_year,
                        new_month_list=new_month_list)


class SAL16ATS(HadCM3TS):
    
    def __init__(self, experiment, start_year, end_year, month_list=None, verbose=False, logger="print"):
        self.data = None
        super(SAL16ATS, self).__init__(experiment, start_year, end_year, file_name="oceansalipg16.annual",
                                       month_list=month_list, verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lon = self.data.longitude.values
        self.lat = self.data.latitude.values
        
        super(SAL16ATS, self).import_coordinates()
    
    def salinity(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                 mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing sea water salinity at 2730m (annual).")
        return self.get(self.data.salinity_ym_dpth.isel(depth_1=0), zone, mode_lon, value_lon, mode_lat, value_lat,
                        None, None, mode_t, value_t, new_start_year=new_start_year, new_end_year=new_end_year,
                        new_month_list=new_month_list)


class SALATS(HadCM3TS):
    
    def __init__(self, experiment, start_year, end_year, month_list=None, verbose=False, logger="print"):
        self.data = None
        super(SALATS, self).__init__(experiment, start_year, end_year, file_name="oceansalipg.annual",
                                     month_list=month_list, verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lon = self.data.longitude.values
        self.lat = self.data.latitude.values
        self.data = self.data.assign_coords(depth_1=-self.data.depth_1)
        self.z = self.data.depth_1.values
        
        super(SALATS, self).import_coordinates()
    
    def salinity(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None, mode_z=None,
                 value_z=None, mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing sea water salinity (annual).")
        return self.get(self.data.salinity_ym_dpth.rename({"depth_1": "z"}), zone, mode_lon, value_lon, mode_lat,
                        value_lat, mode_z, value_z, mode_t, value_t, new_start_year=new_start_year,
                        new_end_year=new_end_year, new_month_list=new_month_list)


class SSTMTS(HadCM3TS):
    
    def __init__(self, experiment, start_year=None, end_year=None, month_list="full", verbose=False, logger="print"):
        self.data = None
        super(SSTMTS, self).__init__(experiment, start_year, end_year, file_name="oceansurftemppf.monthly",
                                     month_list=month_list, verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lon = self.data.longitude.values
        self.lat = self.data.latitude.values
        
        super(SSTMTS, self).import_coordinates()
    
    def sst(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
            mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing SST.")
        return self.get(self.data.temp_mm_uo.isel(unspecified=0).drop("unspecified"), zone, mode_lon, value_lon,
                        mode_lat, value_lat, None, None, mode_t, value_t, new_start_year=new_start_year,
                        new_end_year=new_end_year, new_month_list=new_month_list)


class OCNT01MTS(HadCM3TS):
    
    def __init__(self, experiment, start_year=None, end_year=None, month_list="full", verbose=False, logger="print"):
        self.data = None
        super(OCNT01MTS, self).__init__(experiment, start_year, end_year, file_name="oceantemppf01.monthly",
                                        month_list=month_list, verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lon = self.data.longitude.values
        self.lat = self.data.latitude.values
        
        super(OCNT01MTS, self).import_coordinates()
    
    def temperature(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                    mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing sea water temperature at 5m (monthly).")
        return self.get(self.data.temp_mm_dpth.isel(depth_1=0), zone, mode_lon, value_lon, mode_lat, value_lat, None,
                        None, mode_t, value_t, new_start_year=new_start_year, new_end_year=new_end_year,
                        new_month_list=new_month_list)


class OCNT01ATS(HadCM3TS):
    
    def __init__(self, experiment, start_year=None, end_year=None, month_list=None, verbose=False, logger="print"):
        self.data = None
        super(OCNT01ATS, self).__init__(experiment, start_year, end_year, file_name="oceantemppg01.annual",
                                        month_list=month_list, verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lon = self.data.longitude.values
        self.lat = self.data.latitude.values
        
        super(OCNT01ATS, self).import_coordinates()
    
    def temperature(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                    mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing sea water temperature at 5m (annual).")
        return self.get(self.data.temp_ym_dpth.isel(depth_1=0), zone, mode_lon, value_lon, mode_lat, value_lat, None,
                        None, mode_t, value_t, new_start_year=new_start_year, new_end_year=new_end_year,
                        new_month_list=new_month_list)


class OCNT12ATS(HadCM3TS):
    
    def __init__(self, experiment, start_year=None, end_year=None, month_list=None, verbose=False, logger="print"):
        self.data = None
        super(OCNT12ATS, self).__init__(experiment, start_year, end_year, file_name="oceantemppg12.annual",
                                        month_list=month_list, verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lon = self.data.longitude.values
        self.lat = self.data.latitude.values
        
        super(OCNT12ATS, self).import_coordinates()
    
    def temperature(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                    mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing sea water temperature at 666m (annual).")
        return self.get(self.data.temp_ym_dpth.isel(depth_1=0), zone, mode_lon, value_lon, mode_lat, value_lat, None,
                        None, mode_t, value_t, new_start_year=new_start_year, new_end_year=new_end_year,
                        new_month_list=new_month_list)


class OCNT16ATS(HadCM3TS):
    
    def __init__(self, experiment, start_year=None, end_year=None, month_list=None, verbose=False, logger="print"):
        self.data = None
        super(OCNT16ATS, self).__init__(experiment, start_year, end_year, file_name="oceantemppg16.annual",
                                        month_list=month_list, verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lon = self.data.longitude.values
        self.lat = self.data.latitude.values
        
        super(OCNT16ATS, self).import_coordinates()
    
    def temperature(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                    mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing sea water temperature at 2730m (annual).")
        return self.get(self.data.temp_ym_dpth.isel(depth_1=0), zone, mode_lon, value_lon, mode_lat, value_lat, None,
                        None, mode_t, value_t, new_start_year=new_start_year, new_end_year=new_end_year,
                        new_month_list=new_month_list)


class OCNTATS(HadCM3TS):
    
    def __init__(self, experiment, start_year=None, end_year=None, month_list=None, verbose=False, logger="print"):
        self.data = None
        super(OCNTATS, self).__init__(experiment, start_year, end_year, file_name="oceantemppg.annual",
                                      month_list=month_list, verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lon = self.data.longitude.values
        self.lat = self.data.latitude.values
        self.data = self.data.assign_coords(depth_1=-self.data.depth_1)
        self.z = self.data.depth_1.values
        
        super(OCNTATS, self).import_coordinates()
    
    def temperature(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                    mode_z=None,
                    value_z=None, mode_t=None, value_t=None, new_start_year=None, new_end_year=None,
                    new_month_list=None):
        print("__ Importing sea water temperature (annual).")
        return self.get(self.data.temp_ym_dpth.rename({"depth_1": "z"}), zone, mode_lon, value_lon, mode_lat, value_lat,
                        mode_z, value_z, mode_t, value_t, new_start_year=new_start_year, new_end_year=new_end_year,
                        new_month_list=new_month_list)


class OCNUVEL01MTS(HadCM3TS):
    
    def __init__(self, experiment, start_year=None, end_year=None, month_list="full", verbose=False, logger="print"):
        self.data = None
        super(OCNUVEL01MTS, self).__init__(experiment, start_year, end_year, file_name="oceanuvelpf01.monthly",
                                           month_list=month_list, verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lon = self.data.longitude_1.values
        self.lat = self.data.latitude_1.values
        
        super(OCNUVEL01MTS, self).import_coordinates()
    
    def u_vel(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
              mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing eastward sea water velocity at 5m (monthly).")
        return self.get(self.data.ucurrTot_mm_dpth.isel(depth_1=0).rename({'longitude_1': 'longitude'}).rename(
            {'latitude_1': 'latitude'}), zone, mode_lon, value_lon, mode_lat, value_lat, None, None, mode_t, value_t,
            new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)


class OCNUVELATS(HadCM3TS):
    
    def __init__(self, experiment, start_year=None, end_year=None, verbose=False, logger="print"):
        self.data = None
        super(OCNUVELATS, self).__init__(experiment, start_year, end_year, file_name="oceanuvelpg.annual",
                                         month_list=None, verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lon = self.data.longitude_1.values
        self.lat = self.data.latitude_1.values
        self.data = self.data.assign_coords(depth_1=-self.data.depth_1)
        self.z = self.data.depth_1.values
        
        super(OCNUVELATS, self).import_coordinates()
    
    def u_vel(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None, mode_z=None,
              value_z=None, mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing eastward sea water velocity (annual).")
        return self.get(
            self.data.ucurrTot_mm_dpth.rename({'longitude_1': 'longitude'}).rename({'latitude_1': 'latitude'}).rename(
                {"depth_1": "z"}), zone, mode_lon, value_lon, mode_lat, value_lat, mode_z, value_z, mode_t, value_t,
            new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)


class OCNVVEL01MTS(HadCM3TS):
    
    def __init__(self, experiment, start_year=None, end_year=None, month_list="full", verbose=False, logger="print"):
        self.data = None
        super(OCNVVEL01MTS, self).__init__(experiment, start_year, end_year, file_name="oceanuvelpf01.monthly",
                                           month_list=month_list, verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lon = self.data.longitude_1.values
        self.lat = self.data.latitude_1.values
        
        super(OCNVVEL01MTS, self).import_coordinates()
    
    def v_vel(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
              mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing westward sea water velocity at 5m (monthly).")
        return self.get(self.data.vcurrTot_mm_dpth.isel(depth_1=0).rename({'longitude_1': 'longitude'}).rename(
            {'latitude_1': 'latitude'}), zone, mode_lon, value_lon, mode_lat, value_lat, None, None, mode_t, value_t,
            new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)


class OCNVVELATS(HadCM3TS):
    
    def __init__(self, experiment, start_year=None, end_year=None, month_list=None, verbose=False, logger="print"):
        self.data = None
        super(OCNVVELATS, self).__init__(experiment, start_year, end_year, file_name="oceanuvelpg.annual",
                                         month_list=month_list, verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lon = self.data.longitude_1.values
        self.lat = self.data.latitude_1.values
        self.data = self.data.assign_coords(depth_1=-self.data.depth_1)
        self.z = self.data.depth_1.values
        
        super(OCNVVELATS, self).import_coordinates()
    
    def v_vel(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None, mode_z=None,
              value_z=None, mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing westward sea water velocity (annual).")
        return self.get(
            self.data.vcurrTot_mm_dpth.rename({'longitude_1': 'longitude'}).rename({'latitude_1': 'latitude'}).rename(
                {"depth_1": "z"}), zone, mode_lon, value_lon, mode_lat, value_lat, mode_z, value_z, mode_t, value_t,
            new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)


class MLDMTS(HadCM3TS):
    
    def __init__(self, experiment, start_year, end_year, month_list="full", verbose=False, logger="print"):
        self.data = None
        super(MLDMTS, self).__init__(experiment, start_year, end_year, file_name="oceanmixedpf.monthly",
                                     month_list=month_list, verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lon = self.data.longitude.values
        self.lat = self.data.latitude.values
        
        super(MLDMTS, self).import_coordinates()
    
    def mld(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
            mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing MLD.")
        return self.get(self.data.mixLyrDpth_mm_uo.isel(unspecified=0).drop("unspecified"), zone, mode_lon, value_lon,
                        mode_lat, value_lat, None, None, mode_t, value_t, new_start_year=new_start_year,
                        new_end_year=new_end_year, new_month_list=new_month_list)


class MERIDATS(HadCM3TS):
    
    def __init__(self, experiment, start_year, end_year, month_list=None, verbose=False, logger="print"):
        self.data = None
        super(MERIDATS, self).__init__(experiment, start_year, end_year, file_name="merid.annual",
                                       month_list=month_list, verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lat = self.data.latitude.values
        self.data = self.data.assign_coords(depth=-self.data.depth)
        self.z = self.data.depth.values
        
        super(MERIDATS, self).import_coordinates()
    
    def atlantic(self, zone=zones.NoZone(), mode_lat=None, value_lat=None, mode_z=None, value_z=None, mode_t=None,
                 value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing meridional Overturning Stream Function (Atlantic).")
        return self.get(self.data.Merid_Atlantic.rename({'depth': 'z'}), zone, None, None, mode_lat, value_lat, mode_z,
                        value_z, mode_t, value_t, new_start_year=new_start_year, new_end_year=new_end_year,
                        new_month_list=new_month_list)
    
    def globalx(self, zone=zones.NoZone(), mode_lat=None, value_lat=None, mode_z=None, value_z=None, mode_t=None,
                value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing meridional Overturning Stream Function (Global).")
        return self.get(self.data.Merid_Global.rename({'depth': 'z'}), zone, None, None, mode_lat, value_lat, mode_z,
                        value_z, mode_t, value_t, new_start_year=new_start_year, new_end_year=new_end_year,
                        new_month_list=new_month_list)
    
    def indian(self, zone=zones.NoZone(), mode_lat=None, value_lat=None, mode_z=None, value_z=None, mode_t=None,
               value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing meridional Overturning Stream Function (Indian).")
        return self.get(self.data.Merid_Indian.rename({'depth': 'z'}), zone, None, None, mode_lat, value_lat, mode_z,
                        value_z, mode_t, value_t, new_start_year=new_start_year, new_end_year=new_end_year,
                        new_month_list=new_month_list)
    
    def pacific(self, zone=zones.NoZone(), mode_lat=None, value_lat=None, mode_z=None, value_z=None, mode_t=None,
                value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing meridional Overturning Stream Function (Pacific).")
        return self.get(self.data.Merid_Pacific.rename({'depth': 'z'}), zone, None, None, mode_lat, value_lat, mode_z,
                        value_z, mode_t, value_t, new_start_year=new_start_year, new_end_year=new_end_year,
                        new_month_list=new_month_list)


class OCNSTREAMMTS(HadCM3TS):
    
    def __init__(self, experiment, start_year=None, end_year=None, month_list="full", verbose=False, logger="print"):
        self.data = None
        super(OCNSTREAMMTS, self).__init__(experiment, start_year, end_year, file_name="streamFnpf01.monthly",
                                           month_list=month_list, verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lon = self.data.longitude.values
        self.lat = self.data.latitude.values
        
        super(OCNSTREAMMTS, self).import_coordinates()
    
    def streamfunction(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                       mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing ocean barotropic streamfunction.")
        return self.get(self.data.streamFn_mm_uo.isel(unspecified=0).drop("unspecified"), zone, mode_lon, value_lon,
                        mode_lat, value_lat, None, None, mode_t, value_t, new_start_year=new_start_year,
                        new_end_year=new_end_year, new_month_list=new_month_list)


class PRECIPMTS(HadCM3TS):
    
    def __init__(self, experiment, start_year=None, end_year=None, month_list="full", verbose=False, logger="print"):
        self.data = None
        super(PRECIPMTS, self).__init__(experiment, start_year, end_year, file_name="precip.monthly",
                                        month_list=month_list, verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lon = self.data.longitude.values
        self.lat = self.data.latitude.values
        
        super(PRECIPMTS, self).import_coordinates()
    
    def precip(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
               mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing precipitation flux.")
        return self.get(self.data.precip_mm_srf.isel(surface=0).drop("surface"), zone, mode_lon, value_lon, mode_lat,
                        value_lat, None, None, mode_t, value_t, new_start_year=new_start_year,
                        new_end_year=new_end_year, new_month_list=new_month_list)


class EVAPMTS(HadCM3TS):
    
    def __init__(self, experiment, start_year, end_year, month_list="full", verbose=False, logger="print"):
        self.data = None
        super(EVAPMTS, self).__init__(experiment, start_year, end_year, file_name="evap2.monthly",
                                      month_list=month_list, verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lon = self.data.longitude.values
        self.lonb = self.data.longitude_1.values
        self.lat = self.data.latitude.values
        self.latb = self.data.latitude_1.values
        
        super(EVAPMTS, self).import_coordinates()
    
    def total_evap(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None, mode_t=None,
                   value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing evaporation flux.")
        return self.get(self.data.total_evap.isel(surface=0).drop("surface"), zone, mode_lon, value_lon, mode_lat,
                        value_lat, None, None, mode_t, value_t, new_start_year=new_start_year,
                        new_end_year=new_end_year, new_month_list=new_month_list)


class Q2MMTS(HadCM3TS):
    
    def __init__(self, experiment, start_year=None, end_year=None, month_list="full", verbose=False, logger="print"):
        self.data = None
        super(Q2MMTS, self).__init__(experiment, start_year, end_year, file_name="q2m.monthly",
                                     month_list=month_list, verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lon = self.data.longitude.values
        self.lat = self.data.latitude.values
        
        super(Q2MMTS, self).import_coordinates()
    
    def humidity(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                 mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing specific humidity at 1.5m.")
        return self.get(self.data.q_mm_1_5m.isel(ht=0).drop("ht"), zone, mode_lon, value_lon, mode_lat, value_lat, None,
                        None, mode_t, value_t, new_start_year=new_start_year, new_end_year=new_end_year,
                        new_month_list=new_month_list)


class RH2MMTS(HadCM3TS):
    
    def __init__(self, experiment, start_year=None, end_year=None, month_list="full", verbose=False, logger="print"):
        self.data = None
        super(RH2MMTS, self).__init__(experiment, start_year, end_year, file_name="rh2m.monthly",
                                      month_list=month_list, verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lon = self.data.longitude.values
        self.lat = self.data.latitude.values
        
        super(RH2MMTS, self).import_coordinates()
    
    def humidity(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                 mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing relative humidity at 1.5m.")
        return self.get(self.data.rh_mm_1_5m.isel(ht=0).drop("ht"), zone, mode_lon, value_lon, mode_lat, value_lat,
                        None, None, mode_t, value_t, new_start_year=new_start_year, new_end_year=new_end_year,
                        new_month_list=new_month_list)


class SHMTS(HadCM3TS):
    
    def __init__(self, experiment, start_year=None, end_year=None, month_list="full", verbose=False, logger="print"):
        self.data = None
        super(SHMTS, self).__init__(experiment, start_year, end_year, file_name="sh.monthly",
                                    month_list=month_list, verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lon = self.data.longitude.values
        self.lat = self.data.latitude.values
        
        super(SHMTS, self).import_coordinates()
    
    def heat_flux(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                  mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing surface & b.layer heat fluxes.")
        return self.get(self.data.sh_mm_hyb.isel(hybrid_p_x1000_1=0).drop("hybrid_p_x1000_1"), zone, mode_lon,
                        value_lon, mode_lat, value_lat, None, None, mode_t, value_t, new_start_year=new_start_year,
                        new_end_year=new_end_year, new_month_list=new_month_list)


class LHMTS(HadCM3TS):
    
    def __init__(self, experiment, start_year, end_year, month_list="full", verbose=False, logger="print"):
        self.data = None
        super(LHMTS, self).__init__(experiment, start_year, end_year, file_name="lh.monthly",
                                    month_list=month_list, verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lon = self.data.longitude.values
        self.lat = self.data.latitude.values
        
        super(LHMTS, self).import_coordinates()
    
    def heat_flux(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None, mode_t=None,
                  value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing surface latent heat fluxes.")
        return self.get(self.data.lh_mm_srf.isel(surface=0).drop("surface"), zone, mode_lon, value_lon, mode_lat,
                        value_lat, None, None, mode_t, value_t, new_start_year=new_start_year,
                        new_end_year=new_end_year, new_month_list=new_month_list)


class ICECONCMTS(HadCM3TS):
    
    def __init__(self, experiment, start_year, end_year, month_list="full", verbose=False, logger="print"):
        self.data = None
        super(ICECONCMTS, self).__init__(experiment, start_year, end_year, file_name="iceconc.monthly",
                                         month_list=month_list, verbose=verbose, logger=logger)
    
    @staticmethod
    def transform(array_r):
        return ATMSURFMDS.transform(array_r)
    
    def import_coordinates(self):
        
        self.lon = np.sort(self.data.longitude.values)
        self.lonb = util.guess_bounds(self.lon)
        self.lons = self.lonb[1:] - self.lonb[0:-1]
        self.lon_p = np.append(self.lon, self.lon[-1] + self.lons[-1])
        self.lonb_p = util.guess_bounds(self.lon_p)
        self.lons_p = self.lonb_p[1:] - self.lonb_p[0:-1]

        self.lat = np.sort(self.data.latitude.values)
        self.latb = util.guess_bounds(self.lat)
        self.lats = self.latb[1:] - self.latb[0:-1]
        self.lat_p = self.lat
        self.latb_p = util.guess_bounds(self.lat_p)
        self.lats_p = self.latb_p[1:] - self.latb_p[0:-1]
        
        super(ICECONCMTS, self).import_coordinates()
    
    def ice_conc(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None, mode_t=None,
                 value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing sea ice fraction.")
        return self.get(self.data.iceconc_mm_srf.isel(surface=0).drop("surface"), zone, mode_lon, value_lon, mode_lat,
                        value_lat, None, None, mode_t, value_t, new_start_year=new_start_year,
                        new_end_year=new_end_year, new_month_list=new_month_list)


class ICEDEPTHMTS(HadCM3TS):
    
    def __init__(self, experiment, start_year, end_year, month_list="full", verbose=False, logger="print"):
        self.data = None
        super(ICEDEPTHMTS, self).__init__(experiment, start_year, end_year, file_name="icedepth.monthly",
                                          month_list=month_list, verbose=verbose, logger=logger)
    
    @staticmethod
    def transform(array_r):
        return ATMSURFMDS.transform(array_r)
    
    def import_coordinates(self):
        
        self.lon = np.sort(self.data.longitude.values)
        self.lonb = util.guess_bounds(self.lon)
        self.lons = self.lonb[1:] - self.lonb[0:-1]
        self.lon_p = np.append(self.lon, self.lon[-1] + self.lons[-1])
        self.lonb_p = util.guess_bounds(self.lon_p)
        self.lons_p = self.lonb_p[1:] - self.lonb_p[0:-1]
    
        self.lat = np.sort(self.data.latitude.values)
        self.latb = util.guess_bounds(self.lat)
        self.lats = self.latb[1:] - self.latb[0:-1]
        self.lat_p = self.lat
        self.latb_p = util.guess_bounds(self.lat_p)
        self.lats_p = self.latb_p[1:] - self.latb_p[0:-1]
    
        super(ICEDEPTHMTS, self).import_coordinates()
    
    def ice_depth(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None, mode_t=None,
                  value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing sea ice depth.")
        return self.get(self.data.icedepth_mm_srf.isel(surface=0).drop("surface"), zone, mode_lon, value_lon, mode_lat,
                        value_lat, None, None, mode_t, value_t, new_start_year=new_start_year,
                        new_end_year=new_end_year, new_month_list=new_month_list)


class SNOWMTS(HadCM3TS):
    
    def __init__(self, experiment, start_year=None, end_year=None, month_list="full", verbose=False, logger="print"):
        self.data = None
        super(SNOWMTS, self).__init__(experiment, start_year, end_year, file_name="snowdepth.monthly",
                                      month_list=month_list, verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lon = self.data.longitude.values
        self.lat = self.data.latitude.values
        
        super(SNOWMTS, self).import_coordinates()
    
    def snow_depth(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                   mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing snow amount.")
        return self.get(self.data.snowdepth_mm_srf.isel(surface=0).drop("surface"), zone, mode_lon, value_lon, mode_lat,
                        value_lat, None, None, mode_t, value_t, new_start_year=new_start_year,
                        new_end_year=new_end_year, new_month_list=new_month_list)


class SATMTS(HadCM3TS):
    
    def __init__(self, experiment, start_year, end_year, month_list="full", verbose=False, logger="print"):
        self.data = None
        super(SATMTS, self).__init__(experiment, start_year, end_year, file_name="tempsurf.monthly",
                                     month_list=month_list, verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lon = np.sort(self.data.longitude.values)
        self.lonb = util.guess_bounds(self.lon)
        self.lons = self.lonb[1:] - self.lonb[0:-1]
        self.lon_p = np.append(self.lon, self.lon[-1] + self.lons[-1])
        self.lonb_p = util.guess_bounds(self.lon_p)
        self.lons_p = self.lonb_p[1:] - self.lonb_p[0:-1]

        self.lat = np.sort(self.data.latitude.values)
        self.latb = util.guess_bounds(self.lat)
        self.lats = self.latb[1:] - self.latb[0:-1]
        self.lat_p = self.lat
        self.latb_p = util.guess_bounds(self.lat_p)
        self.lats_p = self.latb_p[1:] - self.latb_p[0:-1]
        
        super(SATMTS, self).import_coordinates()
    
    @staticmethod
    def transform(array_r):
        return ATMSURFMDS.transform(array_r)
    
    def sat(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None, mode_t=None,
            value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing SAT.")
        return self.get(self.kelvin_to_celsius(self.data.temp_mm_srf.isel(surface=0).drop("surface")), zone,
                        mode_lon, value_lon, mode_lat, value_lat, None, None, mode_t, value_t,
                        new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)
    
    @staticmethod
    def kelvin_to_celsius(data_array):
        # Dirty!
        data_array.attrs['valid_min'] = data_array.attrs['valid_min'] - 273.15
        data_array.attrs['valid_max'] = data_array.attrs['valid_max'] - 273.15
        data_array.values = util.kelvin_to_celsius(data_array.values)
        return data_array


class ATMT2MMTS(HadCM3TS):
    
    def __init__(self, experiment, start_year, end_year, month_list="full", verbose=False, logger="print"):
        self.data = None
        super(ATMT2MMTS, self).__init__(experiment, start_year, end_year, file_name="temp2m.monthly",
                                        month_list=month_list, verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lon = self.data.longitude.values
        self.lat = self.data.latitude.values
        
        super(ATMT2MMTS, self).import_coordinates()
    
    def temperature(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                    mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing atmosphere temperature at 1.5m.")
        return self.get(self.data.temp_mm_1_5m.isel(ht=0).drop("ht"), zone, mode_lon, value_lon, mode_lat, value_lat,
                        None, None, mode_t, value_t, new_start_year=new_start_year, new_end_year=new_end_year,
                        new_month_list=new_month_list)


class SOLNETSURFMTS(HadCM3TS):
    
    def __init__(self, experiment, start_year, end_year, month_list="full", verbose=False, logger="print"):
        self.data = None
        super(SOLNETSURFMTS, self).__init__(experiment, start_year, end_year, file_name="net_downsolar_surf.monthly",
                                            month_list=month_list, verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lon = self.data.longitude.values
        self.lat = self.data.latitude.values
        
        super(SOLNETSURFMTS, self).import_coordinates()
    
    def solar_flux(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                   mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing net incoming SW solar flux (surface).")
        return self.get(self.data.solar_mm_s3_srf.isel(surface=0).drop("surface"), zone, mode_lon, value_lon, mode_lat,
                        value_lat, None, None, mode_t, value_t, new_start_year=new_start_year,
                        new_end_year=new_end_year, new_month_list=new_month_list)


class SOLTOTSMTS(HadCM3TS):
    
    def __init__(self, experiment, start_year, end_year, month_list="full", verbose=False, logger="print"):
        self.data = None
        super(SOLTOTSMTS, self).__init__(experiment, start_year, end_year, file_name="total_downsolar_surf.monthly",
                                         month_list=month_list, verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lon = self.data.longitude.values
        self.lat = self.data.latitude.values
        
        super(SOLTOTSMTS, self).import_coordinates()
    
    def solar_flux(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                   mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing total incoming SW solar flux (Surface).")
        return self.get(self.data.downSol_Seaice_mm_s3_srf.isel(surface=0).drop("surface"), zone, mode_lon, value_lon,
                        mode_lat, value_lat, None, None, mode_t, value_t, new_start_year=new_start_year,
                        new_end_year=new_end_year, new_month_list=new_month_list)


class SOLTOAMTS(HadCM3TS):
    
    def __init__(self, experiment, start_year, end_year, month_list="full", verbose=False, logger="print"):
        self.data = None
        super(SOLTOAMTS, self).__init__(experiment, start_year, end_year, file_name="downsolar_toa.monthly",
                                        month_list=month_list, verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lon = self.data.longitude.values
        self.lat = self.data.latitude.values
        
        super(SOLTOAMTS, self).import_coordinates()
    
    def solar_flux(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                   mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing incoming SW solar flux (TOA).")
        return self.get(self.data.downSol_mm_TOA.isel(toa=0).drop("toa"), zone, mode_lon, value_lon, mode_lat,
                        value_lat, None, None, mode_t, value_t, new_start_year=new_start_year,
                        new_end_year=new_end_year, new_month_list=new_month_list)


class SOLUPMTS(HadCM3TS):
    
    def __init__(self, experiment, start_year, end_year, month_list="full", verbose=False, logger="print"):
        self.data = None
        super(SOLUPMTS, self).__init__(experiment, start_year, end_year, file_name="upsolar_toa.monthly",
                                       month_list=month_list, verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lon = self.data.longitude.values
        self.lat = self.data.latitude.values
        
        super(SOLUPMTS, self).import_coordinates()
    
    def solar_flux(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                   mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing outgoing SW solar flux (TOA).")
        return self.get(self.data.upSol_mm_s3_TOA.isel(toa=0).drop("toa"), zone, mode_lon, value_lon, mode_lat,
                        value_lat, None, None, mode_t, value_t, new_start_year=new_start_year,
                        new_end_year=new_end_year, new_month_list=new_month_list)


class OLRMTS(HadCM3TS):
    
    def __init__(self, experiment, start_year=None, end_year=None, month_list="full", verbose=False, logger="print"):
        self.data = None
        super(OLRMTS, self).__init__(experiment, start_year, end_year, file_name="olr.monthly",
                                     month_list=month_list, verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lon = self.data.longitude.values
        self.lat = self.data.latitude.values
        
        super(OLRMTS, self).import_coordinates()
    
    def solar_flux(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                   mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing outgoing LW solar flux (TOA).")
        return self.get(self.data.olr_mm_s3_TOA.isel(toa=0).drop("toa"), zone, mode_lon, value_lon, mode_lat, value_lat,
                        None, None, mode_t, value_t, new_start_year=new_start_year, new_end_year=new_end_year,
                        new_month_list=new_month_list)


class U10MTS(HadCM3TS):
    
    def __init__(self, experiment, start_year, end_year, month_list="full", verbose=False, logger="print"):
        self.data = None
        super(U10MTS, self).__init__(experiment, start_year, end_year, file_name="u10m.monthly",
                                     month_list=month_list, verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lon = self.data.longitude_1.values
        self.lat = self.data.latitude_1.values
        
        super(U10MTS, self).import_coordinates()
    
    def wind(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
             mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing eastward component of wind at 10m.")
        return self.get(
            self.data.u_mm_10m.isel(ht=0).drop("ht").rename({'longitude_1': 'longitude'}).rename(
                {'latitude_1': 'latitude'}), zone, mode_lon, value_lon, mode_lat, value_lat, None, None, mode_t,
            value_t, new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)


class U200MTS(HadCM3TS):
    
    def __init__(self, experiment, start_year, end_year, month_list="full", verbose=False, logger="print"):
        self.data = None
        super(U200MTS, self).__init__(experiment, start_year, end_year, file_name="u200.monthly",
                                      month_list=month_list, verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lon = self.data.longitude.values
        self.lat = self.data.latitude.values
        
        super(U200MTS, self).import_coordinates()
    
    def wind(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
             mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing eastward component of wind at 200m.")
        return self.get(self.data.u_mm_p.isel(p=0), zone, mode_lon, value_lon, mode_lat, value_lat, None, None, mode_t,
                        value_t, new_start_year=new_start_year, new_end_year=new_end_year,
                        new_month_list=new_month_list)


class U850MTS(HadCM3TS):
    
    def __init__(self, experiment, start_year, end_year, month_list="full", verbose=False, logger="print"):
        self.data = None
        super(U850MTS, self).__init__(experiment, start_year, end_year, file_name="u850.monthly",
                                      month_list=month_list, verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lon = self.data.longitude.values
        self.lat = self.data.latitude.values
        
        super(U850MTS, self).import_coordinates()
    
    def wind(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
             mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing eastward component of wind at 850m.")
        return self.get(self.data.u_mm_p.isel(p=0), zone, mode_lon, value_lon, mode_lat, value_lat, None, None, mode_t,
                        value_t, new_start_year=new_start_year, new_end_year=new_end_year,
                        new_month_list=new_month_list)


class V10MTS(HadCM3TS):
    
    def __init__(self, experiment, start_year, end_year, month_list="full", verbose=False, logger="print"):
        self.data = None
        super(V10MTS, self).__init__(experiment, start_year, end_year, file_name="v10m.monthly",
                                     month_list=month_list, verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lon = self.data.longitude_1.values
        self.lat = self.data.latitude_1.values
        
        super(V10MTS, self).import_coordinates()
    
    def wind(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
             mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing westward component of wind at 10m.")
        return self.get(
            self.data.v_mm_10m.isel(ht=0).drop("ht").rename({'longitude_1': 'longitude'}).rename(
                {'latitude_1': 'latitude'}), zone, mode_lon, value_lon, mode_lat, value_lat, None, None, mode_t,
            value_t, new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)


class V200MTS(HadCM3TS):
    
    def __init__(self, experiment, start_year, end_year, month_list="full", verbose=False, logger="print"):
        self.data = None
        super(V200MTS, self).__init__(experiment, start_year, end_year, file_name="v200.monthly",
                                      month_list=month_list, verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lon = self.data.longitude.values
        self.lat = self.data.latitude.values
        
        super(V200MTS, self).import_coordinates()
    
    def wind(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
             mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing westward component of wind at 200m.")
        return self.get(self.data.v_mm_p.isel(p=0), zone, mode_lon, value_lon, mode_lat, value_lat, None, None, mode_t,
                        value_t, new_start_year=new_start_year, new_end_year=new_end_year,
                        new_month_list=new_month_list)


class V850MTS(HadCM3TS):
    
    def __init__(self, experiment, start_year, end_year, month_list="full", verbose=False, logger="print"):
        self.data = None
        super(V850MTS, self).__init__(experiment, start_year, end_year, file_name="v850.monthly",
                                      month_list=month_list, verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lon = self.data.longitude.values
        self.lat = self.data.latitude.values
        
        super(V850MTS, self).import_coordinates()
    
    def wind(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
             mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing westward component of wind at 850m.")
        return self.get(self.data.v_mm_p.isel(p=0), zone, mode_lon, value_lon, mode_lat, value_lat, None, None, mode_t,
                        value_t, new_start_year=new_start_year, new_end_year=new_end_year,
                        new_month_list=new_month_list)


class MSLPMTS(HadCM3TS):
    
    def __init__(self, experiment, start_year, end_year, month_list="full", verbose=False, logger="print"):
        self.data = None
        super(MSLPMTS, self).__init__(experiment, start_year, end_year, file_name="mslp.monthly",
                                      month_list=month_list, verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lon = self.data.longitude.values
        self.lat = self.data.latitude.values
        
        super(MSLPMTS, self).import_coordinates()
    
    def mslp(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None, mode_t=None,
             value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing pressure at mean sea level.")
        return self.get(self.data.p_mm_msl.isel(msl=0).drop("msl"), zone, mode_lon, value_lon, mode_lat, value_lat,
                        None, None, mode_t, value_t, new_start_year=new_start_year, new_end_year=new_end_year,
                        new_month_list=new_month_list)


class Z500MTS(HadCM3TS):
    
    def __init__(self, experiment, start_year, end_year, month_list="full", verbose=False, logger="print"):
        self.data = None
        super(Z500MTS, self).__init__(experiment, start_year, end_year, file_name="z500.monthly",
                                      month_list=month_list, verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lon = self.data.longitude_1.values
        self.lat = self.data.latitude_1.values
        
        super(Z500MTS, self).import_coordinates()
    
    def z500(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
             mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing geopotential height z500.")
        return self.get(
            self.data.ht_mm_p.isel(p=0).rename({'longitude_1': 'longitude'}).rename({'latitude_1': 'latitude'}), zone,
            mode_lon, value_lon, mode_lat, value_lat, None, None, mode_t, value_t, new_start_year=new_start_year,
            new_end_year=new_end_year, new_month_list=new_month_list)


class SMMTS(HadCM3TS):
    
    def __init__(self, experiment, start_year=None, end_year=None, month_list="full", verbose=False, logger="print"):
        self.data = None
        super(SMMTS, self).__init__(experiment, start_year, end_year, file_name="sm.monthly",
                                    month_list=month_list, verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lon = self.data.longitude.values
        self.lat = self.data.latitude.values
        self.z = self.data.level6.values
        
        super(SMMTS, self).import_coordinates()
    
    def moisture(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None, mode_z=None,
                 value_z=None, mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing soil moisture content in a layer.")
        return self.get(self.data.sm_mm_soil.rename({"level6": "z"}), zone, mode_lon, value_lon, mode_lat, value_lat,
                        mode_z, value_z, mode_t, value_t, new_start_year=new_start_year, new_end_year=new_end_year,
                        new_month_list=new_month_list)


class SOILTMTS(HadCM3TS):
    
    def __init__(self, experiment, start_year=None, end_year=None, month_list="full", verbose=False, logger="print"):
        self.data = None
        super(SOILTMTS, self).__init__(experiment, start_year, end_year, file_name="soiltemp.monthly",
                                       month_list=month_list, verbose=verbose, logger=logger)
    
    def import_coordinates(self):
        self.lon = self.data.longitude.values
        self.lat = self.data.latitude.values
        self.z = self.data.level6.values
        
        super(SOILTMTS, self).import_coordinates()
    
    def temperature(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                    mode_z=None, value_z=None, mode_t=None, value_t=None, new_start_year=None, new_end_year=None,
                    new_month_list=None):
        print("__ Importing soil temperature in a layer.")
        return self.get(self.data.soiltemp_mm_soil.rename({"level6": "z"}), zone, mode_lon, value_lon, mode_lat,
                        value_lat, mode_z, value_z, mode_t, value_t, new_start_year=new_start_year,
                        new_end_year=new_end_year, new_month_list=new_month_list)


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
