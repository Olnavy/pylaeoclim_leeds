import pylaeoclim_leeds.processing as proc
import pylaeoclim_leeds.zones as zones
import numpy as np
import xarray as xr
import pylaeoclim_leeds.util_hadcm3 as util
import abc
import cftime
import os
import time
import pathlib
import netCDF4

input_file = util.generate_input(str(pathlib.Path(__file__).parent.absolute()) + "/resources/hadcm3_input")


class HadCM3DS(proc.ModelDS):
    """
    Mother function for all HadCM3 model datasets (raw datasets and time series). Abstract class.
    Inherit from ModelDS. Implement abstract methods and time processing
    Get method defined here temporarly. To factorise in ModelDS in the future?
    """
    
    # Default month_list in HadCM3.
    MONTHS = ['ja', 'fb', 'mr', 'ar', 'my', 'jn', 'jl', 'ag', 'sp', 'ot', 'nv', 'dc']
    
    def __init__(self, exp_name, start_year, end_year, month_list, chunks, verbose, debug, logger):
        """
        Init function, with all parameters common to all dataset. Not to be called.
        De fine docstring in child classes, and leave this one blank.
        
        Parameters
        ----------
        exp_name: string
            Name of the experiment. Has to fit the input file.
        start_year: int
            First year of the dataset.
        end_year: int
            Last year of the dataset.
        month_list:
            Months to be implemented in the dataset.
        chunks: int
            If not None, activate parallel computing on t coordinate
        verbose: bool
            Not implemented yet.
        debug: bool
            Print times of execution and important waypoints.
        logger: bool
            Not implemented yet.
        """
        
        super(HadCM3DS, self).__init__(verbose, debug, logger)
        self.exp_name = exp_name
        self.start_year, self.end_year = start_year, end_year
        self.months = month_list
        self.chunks = chunks
        # Import data or sample data.
        self.import_data()
        # Import available coordinates and compute the others
        self.import_coordinates()
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        """
        Abstract and static method: processing method.
        """
        pass
    
    def processed_time(self, new_start_year=None):
        return np.linspace(0, self.end_year - self.start_year, len(self.t)) + \
               (new_start_year if new_start_year is not None else self.start_year)
    
    @abc.abstractmethod
    def import_data(self):
        pass
    
    @abc.abstractmethod
    def import_coordinates(self):
        print("____ Coordinates imported in the HadCM3DS dataset.")
    
    def get(self, data, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
            mode_z=None, value_z=None, mode_t=None, value_t=None, new_start_year=None, new_end_year=None,
            new_month_list=None):
        
        start = time.time()
        geo_da = proc.GeoDataArray(data, ds=self, process=self.process)  # add the GeoDataArray wrapper
        if self.debug: print(f"** Time elapsed for creating GeoDataArray : {time.time() - start}")
        
        start = time.time()
        geo_da = zone.compact(geo_da)
        if self.debug: print(f"** Time elapsed to compact the zone : {time.time() - start}")
        
        if any([new_start_year is not None, new_end_year is not None, new_month_list is not None]) and mode_t is None:
            print("____ Truncation to new time coordinates.")
            self.t = [cftime.Datetime360Day(year, month, 1)
                      for year in np.arange(int(new_start_year), int(new_end_year) + 1)
                      for month in util.months_to_number(new_month_list)]
        try:
            start = time.time()
            if new_start_year is not None and new_start_year < self.start_year:
                raise ValueError("!!!! The new start year is smaller than the imported one.")
            elif new_end_year is not None and new_end_year > self.end_year:
                raise ValueError("!!!! The new end year is larger than the imported one.")
            elif new_start_year is None and new_end_year is not None:
                geo_da.crop_years(self.start_year, new_end_year)
            elif new_start_year is not None and new_end_year is None:
                geo_da.crop_years(new_start_year, self.end_year)
            elif new_start_year is not None and new_end_year is not None:
                geo_da.crop_years(new_start_year, new_end_year)
            else:
                pass
            if self.debug: print(f"* Time elapsed for crop_years: {time.time() - start}")
            
            start = time.time()
            if new_month_list is not None and self.months is None:
                raise ValueError(f"!!!! The month cropping is not available with {type(self)}.")
            elif new_month_list is not None and \
                not all(
                    month in util.months_to_number(self.months) for month in util.months_to_number(new_month_list)):
                raise ValueError("!!!! The new month list includes months not yet imported.")
            elif new_month_list is not None:
                geo_da.crop_months(new_month_list)
            else:
                pass
            if self.debug: print(f"* Time elapsed for crop_months: {time.time() - start}")
        
        except ValueError as error:
            print(error)
            print("____ The crop was not performed.")
        
        start = time.time()
        geo_da.get_lon(mode_lon, value_lon)
        if self.debug: print(f"* Time elapsed for get_lon: {time.time() - start}")
        
        start = time.time()
        geo_da.get_lat(mode_lat, value_lat)
        if self.debug: print(f"* Time elapsed for get_lat: {time.time() - start}")
        
        start = time.time()
        geo_da.get_z(mode_z, value_z)
        if self.debug: print(f"* Time elapsed for get_z: {time.time() - start}")
        
        start = time.time()
        geo_da.get_t(mode_t, value_t)
        if self.debug: print(f"* Time elapsed for get_t: {time.time() - start}")
        
        # Rebuilding the data:
        if self.chunks is not None:
            print("____ Rebuilding the data_array")
            geo_da.data.load()
        
        return geo_da


# ************
# RAW DATASETS
# ************

class HadCM3RDS(HadCM3DS):
    
    def __init__(self, exp_name, start_year, end_year, file_name, month_list, chunks, verbose, debug, logger):
        self.sample_data = None
        self.file_name = file_name
        self.paths = []
        super(HadCM3RDS, self).__init__(exp_name, start_year, end_year, month_list, chunks, verbose, debug, logger)
    
    def import_data(self):
        print(f"__ Importing {type(self)}")
        print(f"____ Paths generated for {self.exp_name} between years {self.start_year} and {self.end_year}.")
        
        # ADD A METHOD TO CHECK THE VALID RANGE
        
        start = time.time()
        try:
            if self.debug: start = time.time()
            path = input_file[self.exp_name][1]
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
            if self.debug: print(f"* Time elapsed for creating paths : {time.time() - start}")
        except KeyError as error:
            print("!!!! This experiment was not found in \"Experiment_to_filename\". Data import aborted.")
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
    
    def __init__(self, exp_name, start_year, end_year, month_list=None, chunks=None, verbose=True, debug=False,
                 logger="print"):
        month_list = HadCM3DS.MONTHS if month_list is None else month_list  # To overcome mutable argument error
        expt_id = input_file[exp_name][0]
        file_name = f"pcpd/{expt_id}a#pc"
        super(ATMUPMDS, self).__init__(exp_name, start_year, end_year, file_name=file_name, month_list=month_list,
                                       chunks=chunks, verbose=verbose, debug=debug, logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        array = array_r
        if "longitude" in array.dims and proc_lon:
            array = xr.concat([array, array.isel(longitude=0)], dim="longitude")
        if "longitudeb" in array.dims and proc_lon:
            array = xr.concat([array, array.isel(longitudeb=0)], dim="longitudeb")
        if "latitude" in array.dims and proc_lat:
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
    
    def temperature(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                    mode_z=None, value_z=None, mode_t=None, value_t=None,
                    new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing atmosphere temperaure.")
        return self.get(xr.open_mfdataset(
            self.paths, combine='by_coords').temp_mm_p.rename({'p': 'z'}).rename({'longitude_1': 'longitudeb'}).
                        rename({'latitude_1': 'latitudeb'}), zone,
                        mode_lon, value_lon, mode_lat, value_lat, mode_z, value_z, mode_t, value_t,
                        new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)


class ATMSURFMDS(HadCM3RDS):
    """
    PD
    """
    
    def __init__(self, exp_name, start_year, end_year, month_list=None, chunks=None, verbose=True, debug=False,
                 logger="print"):
        month_list = HadCM3DS.MONTHS if month_list is None else month_list  # To overcome mutable argument error
        expt_id = input_file[exp_name][0]
        file_name = f"pcpd/{expt_id}a#pd"
        super(ATMSURFMDS, self).__init__(exp_name, start_year, end_year, file_name=file_name, month_list=month_list,
                                         chunks=chunks, verbose=verbose, debug=debug, logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        array = array_r
        if "longitude" in array.dims and proc_lon:
            array = xr.concat([array, array.isel(longitude=0)], dim="longitude")
        if "longitudeb" in array.dims and proc_lon:
            array = xr.concat([array, array.isel(longitudeb=0)], dim="longitudeb")
        if "latitudeb" in array.dims and proc_lat:
            array.isel(latitudeb=-1).values = array.isel(latitudeb=-2).values
            array = xr.concat([array.isel(latitudeb=0), array, array.isel(latitudeb=-2)], dim="latitudeb")
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
        return self.get(
            xr.open_mfdataset(self.paths, combine='by_coords').temp_mm_srf.isel(surface=0), zone,
            mode_lon, value_lon, mode_lat, value_lat, None, None, mode_t, value_t,
            new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)
    
    def u_wind(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
               mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing eastward component of wind at 10m.")
        return self.get(xr.open_mfdataset(self.paths, combine='by_coords').u_mm_10m.isel(ht=0).drop('ht').
                        rename({'longitude_1': 'longitudeb'}).rename({'latitude_1': 'latitudeb'}), zone,
                        mode_lon, value_lon, mode_lat, value_lat, None, None, mode_t, value_t,
                        new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)
    
    def v_wind(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
               mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing northward component of wind at 10m.")
        return self.get(xr.open_mfdataset(self.paths, combine='by_coords').v_mm_10m.isel(ht=0).drop('ht').
                        rename({'longitude_1': 'longitudeb'}).rename({'latitude_1': 'latitudeb'}), zone,
                        mode_lon, value_lon, mode_lat, value_lat, None, None, mode_t, value_t,
                        new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)
    
    def mslp(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
             mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing mean sea level pressure.")
        return self.get(xr.open_mfdataset(self.paths, combine='by_coords').p_mm_msl.isel(msl=0).drop('msl'), zone,
                        mode_lon, value_lon, mode_lat, value_lat, None, None, mode_t, value_t,
                        new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)
    
    def surfp(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
              mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing sea level pressure.")
        return self.get(xr.open_mfdataset(self.paths, combine='by_coords').p_mm_srf.isel(surface=0).drop('surface'),
                        zone, mode_lon, value_lon, mode_lat, value_lat, None, None, mode_t, value_t,
                        new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)
    
    def downsol_toa(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                    mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing incoming shortwave solar radiation.")
        return self.get(xr.open_mfdataset(
            self.paths, combine='by_coords').downSol_mm_TOA.isel(toa=0).drop('toa'), zone,
                        mode_lon, value_lon, mode_lat, value_lat, None, None, mode_t, value_t,
                        new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)


class OCNMDS(HadCM3RDS):
    """
    PF
    """
    
    def __init__(self, exp_name, start_year, end_year, month_list=None, chunks=None, verbose=True, debug=False,
                 logger="print"):
        month_list = HadCM3DS.MONTHS if month_list is None else month_list  # To overcome mutable argument error
        expt_id = input_file[exp_name][0]
        file_name = f"pf/{expt_id}o#pf"
        super(OCNMDS, self).__init__(exp_name, start_year, end_year, file_name=file_name, month_list=month_list,
                                     chunks=chunks, verbose=verbose, debug=debug, logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        array = array_r
        if "longitude" in array.dims and proc_lon:
            array = xr.concat([array, array.isel(longitude=0)], dim="longitude")
        if "longitudeb" in array.dims and proc_lon:
            array = xr.concat([array, array.isel(longitudeb=0)], dim="longitudeb")
        if "latitude" in array.dims and proc_lat:
            array.isel(latitude=-1).values = array.isel(latitude=-2).values
            array = xr.concat([array.isel(latitude=np.arange(0, len(array.latitude) - 1, 1)), array.isel(latitude=-2),
                               array.isel(latitude=-2)], dim="latitude")
        if "latitudeb" in array.dims and proc_lat:
            array.isel(latitudeb=-1).values = array.isel(latitudeb=-2).values
            array = xr.concat(
                [array.isel(latitudeb=0), array.isel(latitudeb=np.arange(0, len(array.latitudeb) - 1, 1)),
                 array.isel(latitudeb=-2), array.isel(latitudeb=-2)], dim="latitudeb")
        if "zb" in array.dims and proc_z:
            array = xr.concat([array, array.isel(zb=-1)], dim="zb")
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
        
        self.z, self.zb = np.sort(-self.sample_data.depth.values), np.sort(-self.sample_data.depth_1.values)
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
        return self.get(xr.open_mfdataset(self.paths, combine='by_coords').temp_mm_dpth.
                        assign_coords(depth_1=-self.sample_data.depth_1).rename({'depth_1': 'zb'}), zone,
                        mode_lon, value_lon, mode_lat, value_lat, mode_z, value_z, mode_t, value_t,
                        new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)
    
    def salinity(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                 mode_z=None, value_z=None, mode_t=None, value_t=None, new_start_year=None, new_end_year=None,
                 new_month_list=None, convert=True):
        print("__ Importing salinity.")
        if convert:
            data = self.convert_salinity(xr.open_mfdataset(self.paths, combine='by_coords', chunks={"t": self.chunks}))
        else:
            data = xr.open_mfdataset(self.paths, combine='by_coords', chunks={"t": self.chunks})
        return self.get(data.salinity_mm_dpth.
                        assign_coords(depth_1=-self.sample_data.depth_1).rename({'depth_1': 'zb'}),
                        zone, mode_lon, value_lon, mode_lat, value_lat, mode_z, value_z, mode_t, value_t,
                        new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)
    
    @staticmethod
    def convert_salinity(data_array):
        return data_array * 1000 + 35
    
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
        return self.get(xr.open_mfdataset(self.paths, combine='by_coords').ucurrTot_mm_dpth.
                        assign_coords(depth_1=-self.sample_data.depth_1).rename({'depth_1': 'zb'})
                        .rename({'longitude_1': 'longitudeb'}).rename({'latitude_1': 'latitudeb'}),
                        zone, mode_lon, value_lon, mode_lat, value_lat, mode_z, value_z, mode_t, value_t,
                        new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)
    
    def v_velocity(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                   mode_z=None, value_z=None, mode_t=None, value_t=None, new_start_year=None, new_end_year=None,
                   new_month_list=None):
        print("__ Importing zonal (northward) velocity.")
        return self.get(xr.open_mfdataset(self.paths, combine='by_coords').vcurrTot_mm_dpth.
                        assign_coords(depth_1=-self.sample_data.depth_1).rename({'depth_1': 'zb'})
                        .rename({'longitude_1': 'longitudeb'}).rename({'latitude_1': 'latitudeb'}),
                        zone, mode_lon, value_lon, mode_lat, value_lat, mode_z, value_z, mode_t, value_t,
                        new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)
    
    def velocity(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                 mode_z=None, value_z=None, mode_t=None, value_t=None, new_start_year=None, new_end_year=None,
                 new_month_list=None):
        print("__ Importing zonal and meridional velocities and computing total velocity.")
        return self.get(np.sqrt(
            (xr.open_mfdataset(self.paths, combine='by_coords').vcurrTot_mm_dpth.
             assign_coords(depth_1=-self.sample_data.depth_1).rename({'depth_1': 'zb'})
             .rename({'longitude_1': 'longitudeb'}).rename({'latitude_1': 'latitudeb'})) ** 2 +
            (xr.open_mfdataset(self.paths, combine='by_coords').vcurrTot_mm_dpth.
             assign_coords(depth_1=-self.sample_data.depth_1).rename({'depth_1': 'zb'})
             .rename({'longitude_1': 'longitudeb'}).rename({'latitude_1': 'latitudeb'})) ** 2),
            zone, mode_lon, value_lon, mode_lat, value_lat, mode_z, value_z, mode_t, value_t,
            new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)


class OCNYDS(HadCM3RDS):
    """
    PG
    """
    
    def __init__(self, exp_name, start_year, end_year, month_list=None, chunks=None, verbose=True, debug=False,
                 logger="print"):
        expt_id = input_file[exp_name][0]
        file_name = f"pg/{expt_id}o#pg"
        super(OCNYDS, self).__init__(exp_name, start_year, end_year, file_name=file_name, month_list=month_list,
                                     chunks=chunks, verbose=verbose, debug=debug, logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        array = array_r
        if "longitude" in array.dims and proc_lon:
            array = xr.concat([array, array.isel(longitude=0)], dim="longitude")
        if "longitudeb" in array.dims and proc_lon:
            array = xr.concat([array, array.isel(longitudeb=0)], dim="longitudeb")
        if "latitude" in array.dims and proc_lat:
            array.isel(latitude=-1).values = array.isel(latitude=-2).values
            array = xr.concat([array.isel(latitude=np.arange(0, len(array.latitude) - 1, 1)), array.isel(latitude=-2),
                               array.isel(latitude=-2)], dim="latitude")
        if "latitudeb" in array.dims and proc_lat:
            array.isel(latitudeb=-1).values = array.isel(latitudeb=-2).values
            array = xr.concat(
                [array.isel(latitudeb=0), array.isel(latitudeb=np.arange(0, len(array.latitudeb) - 1, 1)),
                 array.isel(latitudeb=-2), array.isel(latitudeb=-2)], dim="latitudeb")
        if "zb" in array.dims and proc_z:
            pass
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
        
        self.z, self.zb = np.sort(-self.sample_data.depth.values), np.sort(-self.sample_data.depth_1.values)
        self.zs = self.zb[1:] - self.zb[0:-1]
        self.z_p = self.z
        self.zb_p = self.zb
        self.zs_p = self.zb_p[1:] - self.zb_p[0:-1]
        
        self.t = [cftime.Datetime360Day(year, 6, 1) for year in np.arange(int(self.start_year), int(self.end_year) + 1)]
        
        super(OCNYDS, self).import_coordinates()
    
    def temperature(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                    mode_z=None, value_z=None, mode_t=None, value_t=None, new_start_year=None, new_end_year=None):
        print("__ Importing temperature.")
        return self.get(xr.open_mfdataset(self.paths, combine='by_coords').temp_ym_dpth.
                        assign_coords(depth_1=-self.sample_data.depth_1).rename({'depth_1': 'zb'}),
                        zone, mode_lon, value_lon, mode_lat, value_lat, mode_z, value_z, mode_t, value_t,
                        new_start_year=new_start_year, new_end_year=new_end_year)
    
    def salinity(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                 mode_z=None, value_z=None, mode_t=None, value_t=None, new_start_year=None, new_end_year=None):
        print("__ Importing salinity.")
        return self.get(xr.open_mfdataset(self.paths, combine='by_coords').salinity_ym_dpth.
                        assign_coords(depth_1=-self.sample_data.depth_1).rename({'depth_1': 'zb'}),
                        zone, mode_lon, value_lon, mode_lat, value_lat, mode_z, value_z, mode_t, value_t,
                        new_start_year=new_start_year, new_end_year=new_end_year)
    
    def u_velocity(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                   mode_z=None, value_z=None, mode_t=None, value_t=None, new_start_year=None, new_end_year=None):
        print("__ Importing meridional (eastward) velocity.")
        return self.get(xr.open_mfdataset(self.paths, combine='by_coords').ucurrTot_mm_dpth.
                        assign_coords(depth_1=-self.sample_data.depth_1).rename({'depth_1': 'zb'})
                        .rename({'longitude_1': 'longitudeb'}).rename({'latitude_1': 'latitudeb'}),
                        zone, mode_lon, value_lon, mode_lat, value_lat, mode_z, value_z, mode_t, value_t,
                        new_start_year=new_start_year, new_end_year=new_end_year)
    
    def v_velocity(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                   mode_z=None, value_z=None, mode_t=None, value_t=None, new_start_year=None, new_end_year=None):
        print("__ Importing zonal (northward) velocity.")
        return self.get(xr.open_mfdataset(self.paths, combine='by_coords').vcurrTot_mm_dpth.
                        assign_coords(depth_1=-self.sample_data.depth_1).rename({'depth_1': 'zb'})
                        .rename({'longitude_1': 'longitudeb'}).rename({'latitude_1': 'latitudeb'}),
                        zone, mode_lon, value_lon, mode_lat, value_lat, mode_z, value_z, mode_t, value_t,
                        new_start_year=new_start_year, new_end_year=new_end_year)
    
    def velocity(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                 mode_z=None, value_z=None, mode_t=None, value_t=None, new_start_year=None, new_end_year=None):
        print("__ Importing zonal and meridional velocities and computing total velocity.")
        return self.get(np.sqrt(
            (xr.open_mfdataset(self.paths, combine='by_coords').vcurrTot_mm_dpth.
             assign_coords(depth_1=-self.sample_data.depth_1).rename({'depth_1': 'zb'})
             .rename({'longitude_1': 'longitudeb'}).rename({'latitude_1': 'latitudeb'})) ** 2 +
            (xr.open_mfdataset(self.paths, combine='by_coords').vcurrTot_mm_dpth.
             assign_coords(depth_1=-self.sample_data.depth_1).rename({'depth_1': 'zb'})
             .rename({'longitude_1': 'longitudeb'}).rename({'latitude_1': 'latitudeb'})) ** 2),
            zone, mode_lon, value_lon, mode_lat, value_lat, mode_z, value_z, mode_t, value_t,
            new_start_year=new_start_year, new_end_year=new_end_year)
    
    def stream(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
               mode_z=None, value_z=None, mode_t=None, value_t=None, new_start_year=None, new_end_year=None):
        print("__ Importing salinity.")
        return self.get(xr.open_mfdataset(self.paths, combine='by_coords').streamFn_ym_uo.
                        isel(unspecified=0).drop("unspecified"),
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
    
    def __init__(self, exp_name, start_year, end_year, file_name, month_list, chunks, verbose, debug, logger):
        self.data = None
        self.file_name = file_name
        start_year = self.get_start_year(exp_name, file_name) if start_year is None else start_year
        end_year = self.get_end_year(exp_name, file_name) if end_year is None else end_year
        self.chunks = chunks
        super(HadCM3TS, self).__init__(exp_name, start_year, end_year, month_list, chunks, verbose, debug, logger)
    
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
    
    def get_start_year(self, exp_name=None, file_name=None):
        # To sort
        exp_name = exp_name if exp_name is not None else self.exp_name
        file_name = file_name if file_name is not None else self.file_name
        
        path = input_file[exp_name][2]
        times = netCDF4.Dataset(f"{path}{exp_name}.{file_name}.nc").variables['t']
        return netCDF4.num2date(np.sort(times[:]), units=times.units, calendar=times.calendar)[0].year
    
    def get_end_year(self, exp_name=None, file_name=None):
        # To sort
        exp_name = exp_name if exp_name is not None else self.exp_name
        file_name = file_name if file_name is not None else self.file_name
        
        path = input_file[exp_name][2]
        times = netCDF4.Dataset(f"{path}{exp_name}.{file_name}.nc").variables['t']
        return netCDF4.num2date(np.sort(times[:]), units=times.units, calendar=times.calendar)[-1].year
    
    def import_data(self):
        
        start = time.time()
        path = ""
        
        try:
            print(
                f"__ Importation of {type(self)} : {self.exp_name} between years "
                f"{self.start_year} and {self.end_year}.")
            
            path = input_file[self.exp_name][2]
            
            if self.debug: start = time.time()
            if self.chunks is not None:
                self.data = xr.open_dataset(f"{path}{self.exp_name}.{self.file_name}.nc", chunks={"t": self.chunks})
            else:
                self.data = xr.open_dataset(f"{path}{self.exp_name}.{self.file_name}.nc")
            if self.debug: print(f"* Time elapsed for open_dataset : {time.time() - start}")
            
            if min(self.data.t.values).year > self.start_year or max(self.data.t.values).year < self.end_year:
                raise ValueError(f"Inavlid start_year or end_year. Please check that they fit the valid range\n"
                                 f"Valid range : start_year = {min(self.data.t.values).year}, "
                                 f"end_year = {max(self.data.t.values).year}")
            
            if self.debug: start = time.time()
            if self.start_year != self.get_start_year():
                self.data = self.data.where(self.data.t >= cftime.Datetime360Day(self.start_year, 1, 1), drop=True)
            if self.debug: print(f"* Time elapsed for crop start year : {time.time() - start}")
            
            if self.debug: start = time.time()
            if self.end_year != self.get_end_year():
                self.data = self.data.where(self.data.t <= cftime.Datetime360Day(self.end_year, 12, 30), drop=True)
            if self.debug: print(f"* Time elapsed for crop end years : {time.time() - start}")
            
            if self.debug: start = time.time()
            if self.months is not self.MONTHS and self.months is not None:
                self.data = self.filter_months(self.data, self.months)
            if self.debug: print(f"* Time elapsed for crop months : {time.time() - start}")
            
            print("____ Import succeeded.")
        
        except FileNotFoundError as error:
            print(f"!!!! {path}{self.exp_name}.{self.file_name}.nc was not found. Data import aborted.")
            raise error
        except KeyError as error:
            print("!!!! This experiment was not found in \"Experiment_to_filename\". Data importation aborted.")
            raise error
    
    def import_coordinates(self):
        super(HadCM3TS, self).import_coordinates()
        self.t = np.sort(self.data.t.values)
    
    def processing_array(self):
        return util.cycle_lon(self.data.values)


class SAL01MTS(HadCM3TS):
    
    def __init__(self, exp_name, start_year=None, end_year=None, month_list=None, chunks=None, verbose=True,
                 debug=False, logger="print"):
        month_list = HadCM3DS.MONTHS if month_list is None else month_list  # To overcome mutable argument error
        super(SAL01MTS, self).__init__(exp_name, start_year, end_year, file_name="oceansalipf01.monthly",
                                       month_list=month_list, chunks=chunks, verbose=verbose, debug=debug,
                                       logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        return OCNMDS.process(array_r, proc_lon, proc_lat, proc_z)
    
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
        self.lat_p = np.append(self.lat, self.lat[-1] + self.lats[-1])
        self.latb_p = util.guess_bounds(self.lat_p)
        self.lats_p = self.latb_p[1:] - self.latb_p[0:-1]
        
        super(SAL01MTS, self).import_coordinates()
    
    def salinity(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                 mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing sea water salinity at 5m (monthly).")
        return self.get(self.data.salinity_mm_dpth.isel(depth_1=0), zone, mode_lon, value_lon, mode_lat, value_lat,
                        None, None, mode_t, value_t, new_start_year=new_start_year, new_end_year=new_end_year,
                        new_month_list=new_month_list)


class SAL01ATS(HadCM3TS):
    
    def __init__(self, exp_name, start_year=None, end_year=None, chunks=None, verbose=True, debug=False,
                 logger="print"):
        super(SAL01ATS, self).__init__(exp_name, start_year, end_year, file_name="oceansalipg01.annual",
                                       month_list=None, chunks=chunks, verbose=verbose, debug=debug, logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        return OCNYDS.process(array_r, proc_lon, proc_lat, proc_z)
    
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
        self.lat_p = np.append(self.lat, self.lat[-1] + self.lats[-1])
        self.latb_p = util.guess_bounds(self.lat_p)
        self.lats_p = self.latb_p[1:] - self.latb_p[0:-1]
        
        super(SAL01ATS, self).import_coordinates()
    
    def salinity(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                 mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing sea water salinity at 5m (annual).")
        return self.get(self.data.salinity_ym_dpth.isel(depth_1=0), zone, mode_lon, value_lon, mode_lat, value_lat,
                        None, None, mode_t, value_t, new_start_year=new_start_year, new_end_year=new_end_year,
                        new_month_list=new_month_list)


class SAL12ATS(HadCM3TS):
    
    def __init__(self, exp_name, start_year=None, end_year=None, chunks=None, verbose=True, debug=False,
                 logger="print"):
        super(SAL12ATS, self).__init__(exp_name, start_year, end_year, file_name="oceansalipg12.annual",
                                       month_list=None, chunks=chunks, verbose=verbose, debug=debug, logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        return OCNYDS.process(array_r, proc_lon, proc_lat, proc_z)
    
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
        self.lat_p = np.append(self.lat, self.lat[-1] + self.lats[-1])
        self.latb_p = util.guess_bounds(self.lat_p)
        self.lats_p = self.latb_p[1:] - self.latb_p[0:-1]
        
        super(SAL12ATS, self).import_coordinates()
    
    def salinity(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                 mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing sea water salinity at 666m (annual).")
        return self.get(self.data.salinity_ym_dpth.isel(depth_1=0), zone, mode_lon, value_lon, mode_lat, value_lat,
                        None, None, mode_t, value_t, new_start_year=new_start_year, new_end_year=new_end_year,
                        new_month_list=new_month_list)


class SAL16ATS(HadCM3TS):
    
    def __init__(self, exp_name, start_year=None, end_year=None, chunks=None, verbose=True, debug=False,
                 logger="print"):
        super(SAL16ATS, self).__init__(exp_name, start_year, end_year, file_name="oceansalipg16.annual",
                                       month_list=None, chunks=chunks, verbose=verbose, debug=debug, logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        return OCNYDS.process(array_r, proc_lon, proc_lat, proc_z)
    
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
        self.lat_p = np.append(self.lat, self.lat[-1] + self.lats[-1])
        self.latb_p = util.guess_bounds(self.lat_p)
        self.lats_p = self.latb_p[1:] - self.latb_p[0:-1]
        
        super(SAL16ATS, self).import_coordinates()
    
    def salinity(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                 mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing sea water salinity at 2730m (annual).")
        return self.get(self.data.salinity_ym_dpth.isel(depth_1=0), zone, mode_lon, value_lon, mode_lat, value_lat,
                        None, None, mode_t, value_t, new_start_year=new_start_year, new_end_year=new_end_year,
                        new_month_list=new_month_list)


class SALATS(HadCM3TS):
    
    def __init__(self, exp_name, start_year=None, end_year=None, chunks=None, verbose=True, debug=False,
                 logger="print"):
        super(SALATS, self).__init__(exp_name, start_year, end_year, file_name="oceansalipg.annual",
                                     month_list=None, chunks=chunks, verbose=verbose, debug=debug, logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        return OCNYDS.process(array_r, proc_lon, proc_lat, proc_z)
    
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
        self.lat_p = np.append(self.lat, self.lat[-1] + self.lats[-1])
        self.latb_p = util.guess_bounds(self.lat_p)
        self.lats_p = self.latb_p[1:] - self.latb_p[0:-1]
        
        self.data = self.data.assign_coords(depth_1=-self.data.depth_1)
        self.zb = np.sort(self.data.depth_1.values)
        self.z = util.guess_from_bounds(self.zb)
        self.zs = self.zb[1:] - self.zb[0:-1]
        self.z_p = self.z
        self.zb_p = self.zb
        self.zs_p = self.zb_p[1:] - self.zb_p[0:-1]
        
        super(SALATS, self).import_coordinates()
    
    def salinity(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None, mode_z=None,
                 value_z=None, mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None,
                 convert=True):
        print("__ Importing sea water salinity (annual).")
        data = self.convert() if convert else self.data
        return self.get(data.salinity_ym_dpth.rename({"depth_1": "zb"}), zone, mode_lon, value_lon, mode_lat,
                        value_lat, mode_z, value_z, mode_t, value_t, new_start_year=new_start_year,
                        new_end_year=new_end_year, new_month_list=new_month_list)
    
    def budget(self, zone=zones.NoZone(), dimensions="all"):
        """
        To factorise
        Returns
        -------

        """
        print("__ Budget sea water salinity (annual).")
        
        geo_da = proc.GeoDataArray(self.data.salinity_ym_dpth.rename({"depth_1": "zb"}), ds=self, process=self.process)
        geo_da = zone.compact(geo_da)
        
        mass_matrix = util.volume_matrix(geo_da.lon, geo_da.lat, geo_da.zb) * 1000
        geo_da.data = geo_da.data * np.resize(mass_matrix, geo_da.data.shape)
        
        if dimensions == "all":
            geo_da.data = geo_da.data.sum(skipna=True)
        elif any([dimension not in geo_da.data.dims for dimension in dimensions]):
            raise KeyError(f"This coordinate was not recognized. Available coordinates: {geo_da.data.dims}")
        else:
            for dimension in dimensions:
                print(f"____ Summing over dimension: {dimension}")
                geo_da.data = geo_da.data.sum(dim=dimension, skipna=True)
        
        # Update coordinates:
        if "longitude" in dimensions or "longitudeb" in dimensions:
            geo_da.update_lon(mode_lon="sum", value_lon=None)
        if "latitude" in dimensions or "latitudeb" in dimensions:
            geo_da.update_lat(mode_lat="sum", value_lat=None)
        if "z" in dimensions or "zb" in dimensions:
            geo_da.update_z(mode_z="sum", value_z=None)
        
        return geo_da
    
    def convert(self):
        return self.data * 1000 + 35


class SSTMTS(HadCM3TS):
    
    def __init__(self, exp_name, start_year=None, end_year=None, month_list=None, chunks=None, verbose=True,
                 debug=False,
                 logger="print"):
        month_list = HadCM3DS.MONTHS if month_list is None else month_list  # To overcome mutable argument error
        super(SSTMTS, self).__init__(exp_name, start_year, end_year, file_name="oceansurftemppf.monthly",
                                     month_list=month_list, chunks=chunks, verbose=verbose, debug=debug, logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        return OCNMDS.process(array_r, proc_lon, proc_lat, proc_z)
    
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
        self.lat_p = np.append(self.lat, self.lat[-1] + self.lats[-1])
        self.latb_p = util.guess_bounds(self.lat_p)
        self.lats_p = self.latb_p[1:] - self.latb_p[0:-1]
        
        super(SSTMTS, self).import_coordinates()
    
    def sst(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
            mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing SST.")
        return self.get(self.data.temp_mm_uo.isel(unspecified=0).drop("unspecified"), zone, mode_lon, value_lon,
                        mode_lat, value_lat, None, None, mode_t, value_t, new_start_year=new_start_year,
                        new_end_year=new_end_year, new_month_list=new_month_list)


class OCNT01MTS(HadCM3TS):
    
    def __init__(self, exp_name, start_year=None, end_year=None, month_list=None, chunks=None, verbose=True,
                 debug=False,
                 logger="print"):
        month_list = HadCM3DS.MONTHS if month_list is None else month_list  # To overcome mutable argument error
        super(OCNT01MTS, self).__init__(exp_name, start_year, end_year, file_name="oceantemppf01.monthly",
                                        month_list=month_list, chunks=chunks, verbose=verbose, debug=debug,
                                        logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        return OCNMDS.process(array_r, proc_lon, proc_lat, proc_z)
    
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
        self.lat_p = np.append(self.lat, self.lat[-1] + self.lats[-1])
        self.latb_p = util.guess_bounds(self.lat_p)
        self.lats_p = self.latb_p[1:] - self.latb_p[0:-1]
        
        super(OCNT01MTS, self).import_coordinates()
    
    def temperature(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                    mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing sea water temperature at 5m (monthly).")
        return self.get(self.data.temp_mm_dpth.isel(depth_1=0), zone, mode_lon, value_lon, mode_lat, value_lat, None,
                        None, mode_t, value_t, new_start_year=new_start_year, new_end_year=new_end_year,
                        new_month_list=new_month_list)


class OCNT01ATS(HadCM3TS):
    
    def __init__(self, exp_name, start_year=None, end_year=None, chunks=None, verbose=True, debug=False,
                 logger="print"):
        super(OCNT01ATS, self).__init__(exp_name, start_year, end_year, file_name="oceantemppg01.annual",
                                        month_list=None, chunks=chunks, verbose=verbose, debug=debug, logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        return OCNYDS.process(array_r, proc_lon, proc_lat, proc_z)
    
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
        self.lat_p = np.append(self.lat, self.lat[-1] + self.lats[-1])
        self.latb_p = util.guess_bounds(self.lat_p)
        self.lats_p = self.latb_p[1:] - self.latb_p[0:-1]
        
        super(OCNT01ATS, self).import_coordinates()
    
    def temperature(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                    mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing sea water temperature at 5m (annual).")
        return self.get(self.data.temp_ym_dpth.isel(depth_1=0), zone, mode_lon, value_lon, mode_lat, value_lat, None,
                        None, mode_t, value_t, new_start_year=new_start_year, new_end_year=new_end_year,
                        new_month_list=new_month_list)


class OCNT12ATS(HadCM3TS):
    
    def __init__(self, exp_name, start_year=None, end_year=None, chunks=None, verbose=True, debug=False,
                 logger="print"):
        super(OCNT12ATS, self).__init__(exp_name, start_year, end_year, file_name="oceantemppg12.annual",
                                        month_list=None, chunks=chunks, verbose=verbose, debug=debug, logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        return OCNYDS.process(array_r, proc_lon, proc_lat, proc_z)
    
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
        self.lat_p = np.append(self.lat, self.lat[-1] + self.lats[-1])
        self.latb_p = util.guess_bounds(self.lat_p)
        self.lats_p = self.latb_p[1:] - self.latb_p[0:-1]
        
        super(OCNT12ATS, self).import_coordinates()
    
    def temperature(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                    mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing sea water temperature at 666m (annual).")
        return self.get(self.data.temp_ym_dpth.isel(depth_1=0), zone, mode_lon, value_lon, mode_lat, value_lat, None,
                        None, mode_t, value_t, new_start_year=new_start_year, new_end_year=new_end_year,
                        new_month_list=new_month_list)


class OCNT16ATS(HadCM3TS):
    
    def __init__(self, exp_name, start_year=None, end_year=None, chunks=None, verbose=True, debug=False,
                 logger="print"):
        super(OCNT16ATS, self).__init__(exp_name, start_year, end_year, file_name="oceantemppg16.annual",
                                        month_list=None, chunks=chunks, verbose=verbose, debug=debug, logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        return OCNYDS.process(array_r, proc_lon, proc_lat, proc_z)
    
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
        self.lat_p = np.append(self.lat, self.lat[-1] + self.lats[-1])
        self.latb_p = util.guess_bounds(self.lat_p)
        self.lats_p = self.latb_p[1:] - self.latb_p[0:-1]
        
        super(OCNT16ATS, self).import_coordinates()
    
    def temperature(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                    mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing sea water temperature at 2730m (annual).")
        return self.get(self.data.temp_ym_dpth.isel(depth_1=0), zone, mode_lon, value_lon, mode_lat, value_lat, None,
                        None, mode_t, value_t, new_start_year=new_start_year, new_end_year=new_end_year,
                        new_month_list=new_month_list)


class OCNTATS(HadCM3TS):
    
    def __init__(self, exp_name, start_year=None, end_year=None, chunks=None, verbose=True, debug=False,
                 logger="print"):
        super(OCNTATS, self).__init__(exp_name, start_year, end_year, file_name="oceantemppg.annual",
                                      month_list=None, chunks=chunks, verbose=verbose, debug=debug, logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        return OCNYDS.process(array_r, proc_lon, proc_lat, proc_z)
    
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
        self.lat_p = np.append(self.lat, self.lat[-1] + self.lats[-1])
        self.latb_p = util.guess_bounds(self.lat_p)
        self.lats_p = self.latb_p[1:] - self.latb_p[0:-1]
        
        self.data = self.data.assign_coords(depth_1=-self.data.depth_1)
        self.zb = np.sort(self.data.depth_1.values)
        self.z = util.guess_from_bounds(self.zb)
        self.zs = self.zb[1:] - self.zb[0:-1]
        self.z_p = self.z
        self.zb_p = self.zb
        self.zs_p = self.zb_p[1:] - self.zb_p[0:-1]
        
        super(OCNTATS, self).import_coordinates()
    
    def temperature(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                    mode_z=None, value_z=None, mode_t=None, value_t=None,
                    new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing sea water temperature (annual).")
        if self.debug:
            start = time.time()
            data = self.data.temp_ym_dpth.rename({"depth_1": "zb"})
            print(f"** Time elapsed for data computation: {time.time() - start}")
            return self.get(data, zone, mode_lon, value_lon, mode_lat, value_lat, mode_z, value_z, mode_t, value_t,
                            new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)
        else:
            return self.get(self.data.temp_ym_dpth.rename({"depth_1": "zb"}), zone, mode_lon, value_lon, mode_lat,
                            value_lat, mode_z, value_z, mode_t, value_t,
                            new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)


class OCNUVEL01MTS(HadCM3TS):
    
    def __init__(self, exp_name, start_year=None, end_year=None, month_list=None, chunks=None, verbose=True,
                 debug=False,
                 logger="print"):
        month_list = HadCM3DS.MONTHS if month_list is None else month_list  # To overcome mutable argument error
        super(OCNUVEL01MTS, self).__init__(exp_name, start_year, end_year, file_name="oceanuvelpf01.monthly",
                                           month_list=month_list, chunks=chunks, verbose=verbose, debug=debug,
                                           logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        return OCNMDS.process(array_r, proc_lon, proc_lat, proc_z)
    
    def import_coordinates(self):
        self.lonb = np.sort(self.data.longitude1.values)
        self.lon = util.guess_from_bounds(self.lonb)
        self.lons = self.lonb[1:] - self.lonb[0:-1]
        self.lon_p = np.append(self.lon, self.lon[-1] + self.lons[-1])
        self.lonb_p = util.guess_bounds(self.lon_p)
        self.lons_p = self.lonb_p[1:] - self.lonb_p[0:-1]
        
        self.latb = np.sort(self.data.latitude1.values)
        self.lat = util.guess_from_bounds(self.latb)
        self.lats = self.latb[1:] - self.latb[0:-1]
        self.lat_p = np.append(self.lat, self.lat[-1] + self.lats[-1])
        self.latb_p = util.guess_bounds(self.lat_p)
        self.lats_p = self.latb_p[1:] - self.latb_p[0:-1]
        
        super(OCNUVEL01MTS, self).import_coordinates()
    
    def u_vel(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
              mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing eastward sea water velocity at 5m (monthly).")
        return self.get(self.data.ucurrTot_mm_dpth.isel(depth_1=0).rename({'longitude_1': 'longitudeb'}).rename(
            {'latitude_1': 'latitudeb'}), zone, mode_lon, value_lon, mode_lat, value_lat, None, None, mode_t, value_t,
            new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)


class OCNUVELATS(HadCM3TS):
    
    def __init__(self, exp_name, start_year=None, end_year=None, chunks=None, verbose=True, debug=False,
                 logger="print"):
        super(OCNUVELATS, self).__init__(exp_name, start_year, end_year, file_name="oceanuvelpg.annual",
                                         month_list=None, chunks=chunks, verbose=verbose, debug=debug, logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        return OCNYDS.process(array_r, proc_lon, proc_lat, proc_z)
    
    def import_coordinates(self):
        self.lonb = np.sort(self.data.longitude1.values)
        self.lon = util.guess_from_bounds(self.lonb)
        self.lons = self.lonb[1:] - self.lonb[0:-1]
        self.lon_p = np.append(self.lon, self.lon[-1] + self.lons[-1])
        self.lonb_p = util.guess_bounds(self.lon_p)
        self.lons_p = self.lonb_p[1:] - self.lonb_p[0:-1]
        
        self.latb = np.sort(self.data.latitude1.values)
        self.lat = util.guess_from_bounds(self.latb)
        self.lats = self.latb[1:] - self.latb[0:-1]
        self.lat_p = np.append(self.lat, self.lat[-1] + self.lats[-1])
        self.latb_p = util.guess_bounds(self.lat_p)
        self.lats_p = self.latb_p[1:] - self.latb_p[0:-1]
        
        self.data = self.data.assign_coords(depth_1=-self.data.depth_1)
        self.zb = np.sort(self.data.depth_1.values)
        self.z = util.guess_from_bounds(self.zb)
        self.zs = self.zb[1:] - self.zb[0:-1]
        self.z_p = self.z
        self.zb_p = self.zb
        self.zs_p = self.zb_p[1:] - self.zb_p[0:-1]
        
        super(OCNUVELATS, self).import_coordinates()
    
    def u_vel(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None, mode_z=None,
              value_z=None, mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing eastward sea water velocity (annual).")
        return self.get(
            self.data.ucurrTot_ym_dpth.rename({'longitude_1': 'longitudeb'}).rename({'latitude_1': 'latitudeb'})
                .rename({"depth_1": "zb"}), zone,
            mode_lon, value_lon, mode_lat, value_lat, mode_z, value_z, mode_t, value_t,
            new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)


class OCNVVEL01MTS(HadCM3TS):
    
    def __init__(self, exp_name, start_year=None, end_year=None, month_list=None, chunks=None, verbose=True,
                 debug=False,
                 logger="print"):
        month_list = HadCM3DS.MONTHS if month_list is None else month_list  # To overcome mutable argument error
        super(OCNVVEL01MTS, self).__init__(exp_name, start_year, end_year, file_name="oceanuvelpf01.monthly",
                                           month_list=month_list, chunks=chunks, verbose=verbose, debug=debug,
                                           logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        return OCNMDS.process(array_r, proc_lon, proc_lat, proc_z)
    
    def import_coordinates(self):
        self.lonb = np.sort(self.data.longitude1.values)
        self.lon = util.guess_from_bounds(self.lonb)
        self.lons = self.lonb[1:] - self.lonb[0:-1]
        self.lon_p = np.append(self.lon, self.lon[-1] + self.lons[-1])
        self.lonb_p = util.guess_bounds(self.lon_p)
        self.lons_p = self.lonb_p[1:] - self.lonb_p[0:-1]
        
        self.latb = np.sort(self.data.latitude1.values)
        self.lat = util.guess_from_bounds(self.latb)
        self.lats = self.latb[1:] - self.latb[0:-1]
        self.lat_p = np.append(self.lat, self.lat[-1] + self.lats[-1])
        self.latb_p = util.guess_bounds(self.lat_p)
        self.lats_p = self.latb_p[1:] - self.latb_p[0:-1]
        
        super(OCNVVEL01MTS, self).import_coordinates()
    
    def v_vel(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
              mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing westward sea water velocity at 5m (monthly).")
        return self.get(self.data.vcurrTot_mm_dpth.isel(depth_1=0).rename({'longitude_1': 'longitudeb'}).rename(
            {'latitude_1': 'latitudeb'}), zone, mode_lon, value_lon, mode_lat, value_lat, None, None, mode_t, value_t,
            new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)


class OCNVVELATS(HadCM3TS):
    
    def __init__(self, exp_name, start_year=None, end_year=None, chunks=None, verbose=True, debug=False,
                 logger="print"):
        super(OCNVVELATS, self).__init__(exp_name, start_year, end_year, file_name="oceanuvelpg.annual",
                                         month_list=None, chunks=chunks, verbose=verbose, debug=debug, logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        return OCNYDS.process(array_r, proc_lon, proc_lat, proc_z)
    
    def import_coordinates(self):
        self.lonb = np.sort(self.data.longitude1.values)
        self.lon = util.guess_from_bounds(self.lonb)
        self.lons = self.lonb[1:] - self.lonb[0:-1]
        self.lon_p = np.append(self.lon, self.lon[-1] + self.lons[-1])
        self.lonb_p = util.guess_bounds(self.lon_p)
        self.lons_p = self.lonb_p[1:] - self.lonb_p[0:-1]
        
        self.latb = np.sort(self.data.latitude1.values)
        self.lat = util.guess_from_bounds(self.latb)
        self.lats = self.latb[1:] - self.latb[0:-1]
        self.lat_p = np.append(self.lat, self.lat[-1] + self.lats[-1])
        self.latb_p = util.guess_bounds(self.lat_p)
        self.lats_p = self.latb_p[1:] - self.latb_p[0:-1]
        
        self.data = self.data.assign_coords(depth_1=-self.data.depth_1)
        self.zb = np.sort(self.data.depth_1.values)
        self.z = util.guess_from_bounds(self.zb)
        self.zs = self.zb[1:] - self.zb[0:-1]
        self.z_p = self.z
        self.zb_p = self.zb
        self.zs_p = self.zb_p[1:] - self.zb_p[0:-1]
        
        super(OCNVVELATS, self).import_coordinates()
    
    def v_vel(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None, mode_z=None,
              value_z=None, mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing westward sea water velocity (annual).")
        return self.get(
            self.data.vcurrTot_mm_dpth.rename({'longitude_1': 'longitudeb'}).rename(
                {'latitude_1': 'latitudeb'}).rename({"depth_1": "zb"}), zone, mode_lon, value_lon, mode_lat, value_lat,
            mode_z, value_z, mode_t, value_t,
            new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)


class MLDMTS(HadCM3TS):
    
    def __init__(self, exp_name, start_year=None, end_year=None, month_list=None, chunks=None, verbose=True,
                 debug=False,
                 logger="print"):
        month_list = HadCM3DS.MONTHS if month_list is None else month_list  # To overcome mutable argument error
        super(MLDMTS, self).__init__(exp_name, start_year, end_year, file_name="oceanmixedpf.monthly",
                                     month_list=month_list, chunks=chunks, verbose=verbose, debug=debug, logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        return OCNMDS.process(array_r, proc_lon, proc_lat, proc_z)
    
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
        self.lat_p = np.append(self.lat, self.lat[-1] + self.lats[-1])
        self.latb_p = util.guess_bounds(self.lat_p)
        self.lats_p = self.latb_p[1:] - self.latb_p[0:-1]
        
        super(MLDMTS, self).import_coordinates()
    
    def mld(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
            mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing MLD.")
        return self.get(self.data.mixLyrDpth_mm_uo.isel(unspecified=0).drop("unspecified"), zone, mode_lon, value_lon,
                        mode_lat, value_lat, None, None, mode_t, value_t, new_start_year=new_start_year,
                        new_end_year=new_end_year, new_month_list=new_month_list)


class MERIDATS(HadCM3TS):
    
    def __init__(self, exp_name, start_year=None, end_year=None, chunks=None, verbose=True, debug=False,
                 logger="print"):
        super(MERIDATS, self).__init__(exp_name, start_year, end_year, file_name="merid.annual",
                                       month_list=None, chunks=chunks, verbose=verbose, debug=debug, logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        return OCNYDS.process(array_r, proc_lon, proc_lat, proc_z)
    
    def import_coordinates(self):
        self.lat = np.sort(self.data.latitude.values)
        self.latb = util.guess_bounds(self.lat)
        self.lats = self.latb[1:] - self.latb[0:-1]
        self.lat_p = np.append(self.lat, self.lat[-1] + self.lats[-1])
        self.latb_p = util.guess_bounds(self.lat_p)
        self.lats_p = self.latb_p[1:] - self.latb_p[0:-1]
        
        self.data = self.data.assign_coords(depth=-self.data.depth)
        self.z = np.sort(self.data.depth.values)
        self.zb = util.guess_bounds(self.z)
        self.zs = self.zb[1:] - self.zb[0:-1]
        self.z_p = self.z
        self.zb_p = self.zb
        self.zs_p = self.zb_p[1:] - self.zb_p[0:-1]
        
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
    
    def __init__(self, exp_name, start_year=None, end_year=None, month_list=None, chunks=None, verbose=True,
                 debug=False,
                 logger="print"):
        month_list = HadCM3DS.MONTHS if month_list is None else month_list  # To overcome mutable argument error
        super(OCNSTREAMMTS, self).__init__(exp_name, start_year, end_year, file_name="streamFnpf01.monthly",
                                           month_list=month_list, chunks=chunks, verbose=verbose, debug=debug,
                                           logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        return OCNMDS.process(array_r, proc_lon, proc_lat, proc_z)
    
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
        self.lat_p = np.append(self.lat, self.lat[-1] + self.lats[-1])
        self.latb_p = util.guess_bounds(self.lat_p)
        self.lats_p = self.latb_p[1:] - self.latb_p[0:-1]
        
        super(OCNSTREAMMTS, self).import_coordinates()
    
    def stream(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
               mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing ocean barotropic streamfunction.")
        return self.get(self.data.streamFn_mm_uo.isel(unspecified=0).drop("unspecified"), zone, mode_lon, value_lon,
                        mode_lat, value_lat, None, None, mode_t, value_t, new_start_year=new_start_year,
                        new_end_year=new_end_year, new_month_list=new_month_list)


class PRECIPMTS(HadCM3TS):
    
    def __init__(self, exp_name, start_year=None, end_year=None, month_list=None, chunks=None, verbose=True,
                 debug=False,
                 logger="print"):
        month_list = HadCM3DS.MONTHS if month_list is None else month_list  # To overcome mutable argument error
        super(PRECIPMTS, self).__init__(exp_name, start_year, end_year, file_name="precip.monthly",
                                        month_list=month_list, chunks=chunks, verbose=verbose, debug=debug,
                                        logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        return ATMSURFMDS.process(array_r, proc_lon, proc_lat, proc_z)
    
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
        
        super(PRECIPMTS, self).import_coordinates()
    
    def precip(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
               mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing precipitation flux.")
        return self.get(self.data.precip_mm_srf.isel(surface=0).drop("surface"), zone, mode_lon, value_lon, mode_lat,
                        value_lat, None, None, mode_t, value_t, new_start_year=new_start_year,
                        new_end_year=new_end_year, new_month_list=new_month_list)


class EVAPMTS(HadCM3TS):
    
    def __init__(self, exp_name, start_year=None, end_year=None, month_list=None, chunks=None, verbose=True,
                 debug=False,
                 logger="print"):
        month_list = HadCM3DS.MONTHS if month_list is None else month_list  # To overcome mutable argument error
        super(EVAPMTS, self).__init__(exp_name, start_year, end_year, file_name="evap2.monthly",
                                      month_list=month_list, chunks=chunks, verbose=verbose, debug=debug, logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        return ATMSURFMDS.process(array_r, proc_lon, proc_lat, proc_z)
    
    def import_coordinates(self):
        self.lon, self.lonb = np.sort(self.data.longitude.values), np.sort(self.data.longitude_1.values)
        self.lons = self.lonb[1:] - self.lonb[0:-1]
        self.lon_p = np.append(self.lon, self.lon[-1] + self.lons[-1])
        self.lonb_p = np.append(self.lonb, [2 * self.lonb[-1] - self.lonb[-2]])
        self.lons_p = self.lonb_p[1:] - self.lonb_p[0:-1]
        
        self.lat, self.latb = np.sort(self.data.latitude.values), np.sort(self.data.latitude_1.values)
        self.lats = self.latb[1:] - self.latb[0:-1]
        self.lat_p = self.lat
        self.latb_p = np.concatenate(([-90], self.latb, [2 * self.latb[-1] - self.latb[-2]]))
        self.lats_p = self.latb_p[1:] - self.latb_p[0:-1]
        
        super(EVAPMTS, self).import_coordinates()
    
    def total_evap(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None, mode_t=None,
                   value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing evaporation flux.")
        return self.get(self.data.total_evap.isel(surface=0).drop("surface"), zone, mode_lon, value_lon, mode_lat,
                        value_lat, None, None, mode_t, value_t, new_start_year=new_start_year,
                        new_end_year=new_end_year, new_month_list=new_month_list)


class Q2MMTS(HadCM3TS):
    
    def __init__(self, exp_name, start_year=None, end_year=None, month_list=None, chunks=None, verbose=True,
                 debug=False,
                 logger="print"):
        month_list = HadCM3DS.MONTHS if month_list is None else month_list  # To overcome mutable argument error
        super(Q2MMTS, self).__init__(exp_name, start_year, end_year, file_name="q2m.monthly",
                                     month_list=month_list, chunks=chunks, verbose=verbose, debug=debug, logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        return ATMSURFMDS.process(array_r, proc_lon, proc_lat, proc_z)
    
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
        
        super(Q2MMTS, self).import_coordinates()
    
    def humidity(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                 mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing specific humidity at 1.5m.")
        return self.get(self.data.q_mm_1_5m.isel(ht=0).drop("ht"), zone, mode_lon, value_lon, mode_lat, value_lat, None,
                        None, mode_t, value_t, new_start_year=new_start_year, new_end_year=new_end_year,
                        new_month_list=new_month_list)


class RH2MMTS(HadCM3TS):
    
    def __init__(self, exp_name, start_year=None, end_year=None, month_list=None, chunks=None, verbose=True,
                 debug=False,
                 logger="print"):
        month_list = HadCM3DS.MONTHS if month_list is None else month_list  # To overcome mutable argument error
        super(RH2MMTS, self).__init__(exp_name, start_year, end_year, file_name="rh2m.monthly",
                                      month_list=month_list, chunks=chunks, verbose=verbose, debug=debug, logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        return ATMSURFMDS.process(array_r, proc_lon, proc_lat, proc_z)
    
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
        
        super(RH2MMTS, self).import_coordinates()
    
    def humidity(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                 mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing relative humidity at 1.5m.")
        return self.get(self.data.rh_mm_1_5m.isel(ht=0).drop("ht"), zone, mode_lon, value_lon, mode_lat, value_lat,
                        None, None, mode_t, value_t, new_start_year=new_start_year, new_end_year=new_end_year,
                        new_month_list=new_month_list)


class SHMTS(HadCM3TS):
    
    def __init__(self, exp_name, start_year=None, end_year=None, month_list=None, chunks=None, verbose=True,
                 debug=False,
                 logger="print"):
        month_list = HadCM3DS.MONTHS if month_list is None else month_list  # To overcome mutable argument error
        super(SHMTS, self).__init__(exp_name, start_year, end_year, file_name="sh.monthly",
                                    month_list=month_list, chunks=chunks, verbose=verbose, debug=debug, logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        return ATMSURFMDS.process(array_r, proc_lon, proc_lat, proc_z)
    
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
        
        super(SHMTS, self).import_coordinates()
    
    def heat_flux(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                  mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing surface & b.layer heat fluxes.")
        return self.get(self.data.sh_mm_hyb.isel(hybrid_p_x1000_1=0).drop("hybrid_p_x1000_1"), zone, mode_lon,
                        value_lon, mode_lat, value_lat, None, None, mode_t, value_t, new_start_year=new_start_year,
                        new_end_year=new_end_year, new_month_list=new_month_list)


class LHMTS(HadCM3TS):
    
    def __init__(self, exp_name, start_year=None, end_year=None, month_list=None, chunks=None, verbose=True,
                 debug=False,
                 logger="print"):
        month_list = HadCM3DS.MONTHS if month_list is None else month_list  # To overcome mutable argument error
        super(LHMTS, self).__init__(exp_name, start_year, end_year, file_name="lh.monthly",
                                    month_list=month_list, chunks=chunks, verbose=verbose, debug=debug, logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        return ATMSURFMDS.process(array_r, proc_lon, proc_lat, proc_z)
    
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
        
        super(LHMTS, self).import_coordinates()
    
    def heat_flux(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None, mode_t=None,
                  value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing surface latent heat fluxes.")
        return self.get(self.data.lh_mm_srf.isel(surface=0).drop("surface"), zone, mode_lon, value_lon, mode_lat,
                        value_lat, None, None, mode_t, value_t, new_start_year=new_start_year,
                        new_end_year=new_end_year, new_month_list=new_month_list)


class ICECONCMTS(HadCM3TS):
    
    def __init__(self, exp_name, start_year=None, end_year=None, month_list=None, chunks=None, verbose=True,
                 debug=False,
                 logger="print"):
        month_list = HadCM3DS.MONTHS if month_list is None else month_list  # To overcome mutable argument error
        super(ICECONCMTS, self).__init__(exp_name, start_year, end_year, file_name="iceconc.monthly",
                                         month_list=month_list, chunks=chunks, verbose=verbose, debug=debug,
                                         logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        return ATMSURFMDS.process(array_r, proc_lon, proc_lat, proc_z)
    
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
    
    def __init__(self, exp_name, start_year=None, end_year=None, month_list=None, chunks=None, verbose=True,
                 debug=False,
                 logger="print"):
        month_list = HadCM3DS.MONTHS if month_list is None else month_list  # To overcome mutable argument error
        super(ICEDEPTHMTS, self).__init__(exp_name, start_year, end_year, file_name="icedepth.monthly",
                                          month_list=month_list, chunks=chunks, verbose=verbose, debug=debug,
                                          logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        return ATMSURFMDS.process(array_r, proc_lon, proc_lat, proc_z)
    
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
    
    def __init__(self, exp_name, start_year=None, end_year=None, month_list=None, chunks=None, verbose=True,
                 debug=False,
                 logger="print"):
        month_list = HadCM3DS.MONTHS if month_list is None else month_list  # To overcome mutable argument error
        super(SNOWMTS, self).__init__(exp_name, start_year, end_year, file_name="snowdepth.monthly",
                                      month_list=month_list, chunks=chunks, verbose=verbose, debug=debug, logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        return ATMSURFMDS.process(array_r, proc_lon, proc_lat, proc_z)
    
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
        
        super(SNOWMTS, self).import_coordinates()
    
    def snow_depth(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                   mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing snow amount.")
        return self.get(self.data.snowdepth_mm_srf.isel(surface=0).drop("surface"), zone, mode_lon, value_lon, mode_lat,
                        value_lat, None, None, mode_t, value_t, new_start_year=new_start_year,
                        new_end_year=new_end_year, new_month_list=new_month_list)


class SATMTS(HadCM3TS):
    
    def __init__(self, exp_name, start_year=None, end_year=None, month_list=None, chunks=None, verbose=True,
                 debug=False,
                 logger="print"):
        month_list = HadCM3DS.MONTHS if month_list is None else month_list  # To overcome mutable argument error
        super(SATMTS, self).__init__(exp_name, start_year, end_year, file_name="tempsurf.monthly",
                                     month_list=month_list, chunks=chunks, verbose=verbose, debug=debug, logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        return ATMSURFMDS.process(array_r, proc_lon, proc_lat, proc_z)
    
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
    
    def sat(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None, mode_t=None,
            value_t=None, new_start_year=None, new_end_year=None, new_month_list=None, convert=True):
        print("__ Importing SAT.")
        data = self.convert() if convert else self.data
        return self.get(data.temp_mm_srf.isel(surface=0).drop("surface"), zone,
                        mode_lon, value_lon, mode_lat, value_lat, None, None, mode_t, value_t,
                        new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)
    
    def convert(self):
        # self.data.attrs['valid_min'] = self.data.attrs['valid_min'] - 273.15
        # self.data.attrs['valid_max'] = self.data.attrs['valid_max'] - 273.15
        return self.data - 273.15


class ATMT2MMTS(HadCM3TS):
    
    def __init__(self, exp_name, start_year=None, end_year=None, month_list=None, chunks=None, verbose=True,
                 debug=False,
                 logger="print"):
        month_list = HadCM3DS.MONTHS if month_list is None else month_list  # To overcome mutable argument error
        super(ATMT2MMTS, self).__init__(exp_name, start_year, end_year, file_name="temp2m.monthly",
                                        month_list=month_list, chunks=chunks, verbose=verbose, debug=debug,
                                        logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        return ATMSURFMDS.process(array_r, proc_lon, proc_lat, proc_z)
    
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
        
        super(ATMT2MMTS, self).import_coordinates()
    
    def temperature(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                    mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing atmosphere temperature at 1.5m.")
        return self.get(self.data.temp_mm_1_5m.isel(ht=0).drop("ht"), zone, mode_lon, value_lon, mode_lat, value_lat,
                        None, None, mode_t, value_t, new_start_year=new_start_year, new_end_year=new_end_year,
                        new_month_list=new_month_list)


class SOLNETSURFMTS(HadCM3TS):
    
    def __init__(self, exp_name, start_year=None, end_year=None, month_list=None, chunks=None, verbose=True,
                 debug=False,
                 logger="print"):
        month_list = HadCM3DS.MONTHS if month_list is None else month_list  # To overcome mutable argument error
        super(SOLNETSURFMTS, self).__init__(exp_name, start_year, end_year, file_name="net_downsolar_surf.monthly",
                                            month_list=month_list, chunks=chunks, verbose=verbose, debug=debug,
                                            logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        return ATMSURFMDS.process(array_r, proc_lon, proc_lat, proc_z)
    
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
        
        super(SOLNETSURFMTS, self).import_coordinates()
    
    def solar_flux(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                   mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing net incoming SW solar flux (surface).")
        return self.get(self.data.solar_mm_s3_srf.isel(surface=0).drop("surface"), zone, mode_lon, value_lon, mode_lat,
                        value_lat, None, None, mode_t, value_t, new_start_year=new_start_year,
                        new_end_year=new_end_year, new_month_list=new_month_list)


class SOLTOTSMTS(HadCM3TS):
    
    def __init__(self, exp_name, start_year=None, end_year=None, month_list=None, chunks=None, verbose=True,
                 debug=False,
                 logger="print"):
        month_list = HadCM3DS.MONTHS if month_list is None else month_list  # To overcome mutable argument error
        super(SOLTOTSMTS, self).__init__(exp_name, start_year, end_year, file_name="total_downsolar_surf.monthly",
                                         month_list=month_list, chunks=chunks, verbose=verbose, debug=debug,
                                         logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        return ATMSURFMDS.process(array_r, proc_lon, proc_lat, proc_z)
    
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
        
        super(SOLTOTSMTS, self).import_coordinates()
    
    def solar_flux(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                   mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing total incoming SW solar flux (Surface).")
        return self.get(self.data.downSol_Seaice_mm_s3_srf.isel(surface=0).drop("surface"), zone, mode_lon, value_lon,
                        mode_lat, value_lat, None, None, mode_t, value_t, new_start_year=new_start_year,
                        new_end_year=new_end_year, new_month_list=new_month_list)


class SOLTOAMTS(HadCM3TS):
    
    def __init__(self, exp_name, start_year=None, end_year=None, month_list=None, chunks=None, verbose=True,
                 debug=False,
                 logger="print"):
        month_list = HadCM3DS.MONTHS if month_list is None else month_list  # To overcome mutable argument error
        super(SOLTOAMTS, self).__init__(exp_name, start_year, end_year, file_name="downsolar_toa.monthly",
                                        month_list=month_list, chunks=chunks, verbose=verbose, debug=debug,
                                        logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        return ATMSURFMDS.process(array_r, proc_lon, proc_lat, proc_z)
    
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
        
        super(SOLTOAMTS, self).import_coordinates()
    
    def solar_flux(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                   mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing incoming SW solar flux (TOA).")
        return self.get(self.data.downSol_mm_TOA.isel(toa=0).drop("toa"), zone, mode_lon, value_lon, mode_lat,
                        value_lat, None, None, mode_t, value_t, new_start_year=new_start_year,
                        new_end_year=new_end_year, new_month_list=new_month_list)


class SOLUPMTS(HadCM3TS):
    
    def __init__(self, exp_name, start_year=None, end_year=None, month_list=None, chunks=None, verbose=True,
                 debug=False,
                 logger="print"):
        month_list = HadCM3DS.MONTHS if month_list is None else month_list  # To overcome mutable argument error
        super(SOLUPMTS, self).__init__(exp_name, start_year, end_year, file_name="upsolar_toa.monthly",
                                       month_list=month_list, chunks=chunks, verbose=verbose, debug=debug,
                                       logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        return ATMSURFMDS.process(array_r, proc_lon, proc_lat, proc_z)
    
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
        
        super(SOLUPMTS, self).import_coordinates()
    
    def solar_flux(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                   mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing outgoing SW solar flux (TOA).")
        return self.get(self.data.upSol_mm_s3_TOA.isel(toa=0).drop("toa"), zone, mode_lon, value_lon, mode_lat,
                        value_lat, None, None, mode_t, value_t, new_start_year=new_start_year,
                        new_end_year=new_end_year, new_month_list=new_month_list)


class OLRMTS(HadCM3TS):
    
    def __init__(self, exp_name, start_year=None, end_year=None, month_list=None, chunks=None, verbose=True,
                 debug=False,
                 logger="print"):
        month_list = HadCM3DS.MONTHS if month_list is None else month_list  # To overcome mutable argument error
        super(OLRMTS, self).__init__(exp_name, start_year, end_year, file_name="olr.monthly",
                                     month_list=month_list, chunks=chunks, verbose=verbose, debug=debug, logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        return ATMSURFMDS.process(array_r, proc_lon, proc_lat, proc_z)
    
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
        
        super(OLRMTS, self).import_coordinates()
    
    def solar_flux(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
                   mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing outgoing LW solar flux (TOA).")
        return self.get(self.data.olr_mm_s3_TOA.isel(toa=0).drop("toa"), zone, mode_lon, value_lon, mode_lat, value_lat,
                        None, None, mode_t, value_t, new_start_year=new_start_year, new_end_year=new_end_year,
                        new_month_list=new_month_list)


class U10MTS(HadCM3TS):
    
    def __init__(self, exp_name, start_year=None, end_year=None, month_list=None, chunks=None, verbose=True,
                 debug=False,
                 logger="print"):
        month_list = HadCM3DS.MONTHS if month_list is None else month_list  # To overcome mutable argument error
        super(U10MTS, self).__init__(exp_name, start_year, end_year, file_name="u10m.monthly",
                                     month_list=month_list, chunks=chunks, verbose=verbose, debug=debug, logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        return ATMSURFMDS.process(array_r, proc_lon, proc_lat, proc_z)
    
    def import_coordinates(self):
        self.lonb = np.sort(self.data.longitude_1.values)
        self.lon = util.guess_from_bounds(self.lonb)
        self.lons = self.lonb[1:] - self.lonb[0:-1]
        self.lon_p = np.append(self.lon, self.lon[-1] + self.lons[-1])
        self.lonb_p = util.guess_bounds(self.lon_p)
        self.lons_p = self.lonb_p[1:] - self.lonb_p[0:-1]
        
        self.latb = np.sort(self.data.latitude_1.values)
        self.lat = util.guess_from_bounds(self.latb)
        self.lats = self.latb[1:] - self.latb[0:-1]
        self.lat_p = self.lat
        self.latb_p = util.guess_bounds(self.lat_p)
        self.lats_p = self.latb_p[1:] - self.latb_p[0:-1]
        
        super(U10MTS, self).import_coordinates()
    
    def u_wind(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
               mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing eastward component of wind at 10m.")
        return self.get(
            self.data.u_mm_10m.isel(ht=0).drop("ht").rename({'longitude_1': 'longitudeb'}).rename(
                {'latitude_1': 'latitudeb'}), zone, mode_lon, value_lon, mode_lat, value_lat, None, None, mode_t,
            value_t, new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)


class U200MTS(HadCM3TS):
    
    def __init__(self, exp_name, start_year=None, end_year=None, month_list=None, chunks=None, verbose=True,
                 debug=False,
                 logger="print"):
        month_list = HadCM3DS.MONTHS if month_list is None else month_list  # To overcome mutable argument error
        super(U200MTS, self).__init__(exp_name, start_year, end_year, file_name="u200.monthly",
                                      month_list=month_list, chunks=chunks, verbose=verbose, debug=debug, logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        return ATMUPMDS.process(array_r, proc_lon, proc_lat, proc_z)
    
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
        self.lat_p = np.concatenate(([-90], self.lat, [90]))
        self.latb_p = util.guess_bounds(self.lat_p)
        self.lats_p = self.latb_p[1:] - self.latb_p[0:-1]
        
        super(U200MTS, self).import_coordinates()
    
    def u_wind(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
               mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing eastward component of wind at 200m.")
        return self.get(self.data.u_mm_p.isel(p=9), zone, mode_lon, value_lon, mode_lat, value_lat, None, None, mode_t,
                        value_t, new_start_year=new_start_year, new_end_year=new_end_year,
                        new_month_list=new_month_list)


class U850MTS(HadCM3TS):
    
    def __init__(self, exp_name, start_year=None, end_year=None, month_list=None, chunks=None, verbose=True,
                 debug=False,
                 logger="print"):
        month_list = HadCM3DS.MONTHS if month_list is None else month_list  # To overcome mutable argument error
        super(U850MTS, self).__init__(exp_name, start_year, end_year, file_name="u850.monthly",
                                      month_list=month_list, chunks=chunks, verbose=verbose, debug=debug, logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        return ATMUPMDS.process(array_r, proc_lon, proc_lat, proc_z)
    
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
        self.lat_p = np.concatenate(([-90], self.lat, [90]))
        self.latb_p = util.guess_bounds(self.lat_p)
        self.lats_p = self.latb_p[1:] - self.latb_p[0:-1]
        
        super(U850MTS, self).import_coordinates()
    
    def u_wind(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
               mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing eastward component of wind at 850m.")
        return self.get(self.data.u_mm_p.isel(p=2), zone, mode_lon, value_lon, mode_lat, value_lat, None, None, mode_t,
                        value_t, new_start_year=new_start_year, new_end_year=new_end_year,
                        new_month_list=new_month_list)


class V10MTS(HadCM3TS):
    
    def __init__(self, exp_name, start_year=None, end_year=None, month_list=None, chunks=None, verbose=True,
                 debug=False,
                 logger="print"):
        month_list = HadCM3DS.MONTHS if month_list is None else month_list  # To overcome mutable argument error
        super(V10MTS, self).__init__(exp_name, start_year, end_year, file_name="v10m.monthly",
                                     month_list=month_list, chunks=chunks, verbose=verbose, debug=debug, logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        return ATMSURFMDS.process(array_r, proc_lon, proc_lat, proc_z)
    
    def import_coordinates(self):
        self.lonb = np.sort(self.data.longitude_1.values)
        self.lon = util.guess_from_bounds(self.lonb)
        self.lons = self.lonb[1:] - self.lonb[0:-1]
        self.lon_p = np.append(self.lon, self.lon[-1] + self.lons[-1])
        self.lonb_p = util.guess_bounds(self.lon_p)
        self.lons_p = self.lonb_p[1:] - self.lonb_p[0:-1]
        
        self.latb = np.sort(self.data.latitude_1.values)
        self.lat = util.guess_from_bounds(self.latb)
        self.lats = self.latb[1:] - self.latb[0:-1]
        self.lat_p = self.lat
        self.latb_p = util.guess_bounds(self.lat_p)
        self.lats_p = self.latb_p[1:] - self.latb_p[0:-1]
        
        super(V10MTS, self).import_coordinates()
    
    def v_wind(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
               mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing westward component of wind at 10m.")
        return self.get(
            self.data.v_mm_10m.isel(ht=0).drop("ht").rename({'longitude_1': 'longitudeb'}).rename(
                {'latitude_1': 'latitudeb'}), zone, mode_lon, value_lon, mode_lat, value_lat, None, None, mode_t,
            value_t, new_start_year=new_start_year, new_end_year=new_end_year, new_month_list=new_month_list)


class V200MTS(HadCM3TS):
    
    def __init__(self, exp_name, start_year=None, end_year=None, month_list=None, chunks=None, verbose=True,
                 debug=False,
                 logger="print"):
        month_list = HadCM3DS.MONTHS if month_list is None else month_list  # To overcome mutable argument error
        super(V200MTS, self).__init__(exp_name, start_year, end_year, file_name="v200.monthly",
                                      month_list=month_list, chunks=chunks, verbose=verbose, debug=debug, logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        return ATMUPMDS.process(array_r, proc_lon, proc_lat, proc_z)
    
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
        self.lat_p = np.concatenate(([-90], self.lat, [90]))
        self.latb_p = util.guess_bounds(self.lat_p)
        self.lats_p = self.latb_p[1:] - self.latb_p[0:-1]
        
        super(V200MTS, self).import_coordinates()
    
    def v_wind(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
               mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing westward component of wind at 200m.")
        return self.get(self.data.v_mm_p.isel(p=9), zone, mode_lon, value_lon, mode_lat, value_lat, None, None, mode_t,
                        value_t, new_start_year=new_start_year, new_end_year=new_end_year,
                        new_month_list=new_month_list)


class V850MTS(HadCM3TS):
    
    def __init__(self, exp_name, start_year=None, end_year=None, month_list=None, chunks=None, verbose=True,
                 debug=False,
                 logger="print"):
        month_list = HadCM3DS.MONTHS if month_list is None else month_list  # To overcome mutable argument error
        super(V850MTS, self).__init__(exp_name, start_year, end_year, file_name="v850.monthly",
                                      month_list=month_list, chunks=chunks, verbose=verbose, debug=debug, logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        return ATMUPMDS.process(array_r, proc_lon, proc_lat, proc_z)
    
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
        self.lat_p = np.concatenate(([-90], self.lat, [90]))
        self.latb_p = util.guess_bounds(self.lat_p)
        self.lats_p = self.latb_p[1:] - self.latb_p[0:-1]
        
        super(V850MTS, self).import_coordinates()
    
    def v_wind(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
               mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing westward component of wind at 850m.")
        return self.get(self.data.v_mm_p.isel(p=2), zone, mode_lon, value_lon, mode_lat, value_lat, None, None, mode_t,
                        value_t, new_start_year=new_start_year, new_end_year=new_end_year,
                        new_month_list=new_month_list)


class MSLPMTS(HadCM3TS):
    
    def __init__(self, exp_name, start_year=None, end_year=None, month_list=None, chunks=None, verbose=True,
                 debug=False,
                 logger="print"):
        month_list = HadCM3DS.MONTHS if month_list is None else month_list  # To overcome mutable argument error
        super(MSLPMTS, self).__init__(exp_name, start_year, end_year, file_name="mslp.monthly",
                                      month_list=month_list, chunks=chunks, verbose=verbose, debug=debug, logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        return ATMSURFMDS.process(array_r, proc_lon, proc_lat, proc_z)
    
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
        
        super(MSLPMTS, self).import_coordinates()
    
    def mslp(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None, mode_t=None,
             value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing pressure at mean sea level.")
        return self.get(self.data.p_mm_msl.isel(msl=0).drop("msl"), zone, mode_lon, value_lon, mode_lat, value_lat,
                        None, None, mode_t, value_t, new_start_year=new_start_year, new_end_year=new_end_year,
                        new_month_list=new_month_list)


class Z500MTS(HadCM3TS):
    
    def __init__(self, exp_name, start_year=None, end_year=None, month_list=None, chunks=None, verbose=True,
                 debug=False,
                 logger="print"):
        month_list = HadCM3DS.MONTHS if month_list is None else month_list  # To overcome mutable argument error
        super(Z500MTS, self).__init__(exp_name, start_year, end_year, file_name="z500.monthly",
                                      month_list=month_list, chunks=chunks, verbose=verbose, debug=debug, logger=logger)
    
    @staticmethod
    def process(array_r, proc_lon, proc_lat, proc_z):
        return ATMUPMDS.process(array_r, proc_lon, proc_lat, proc_z)
    
    def import_coordinates(self):
        self.lonb = np.sort(self.data.longitude_1.values)
        self.lon = util.guess_from_bounds(self.lonb)
        self.lons = self.lonb[1:] - self.lonb[0:-1]
        self.lon_p = np.append(self.lon, self.lon[-1] + self.lons[-1])
        self.lonb_p = util.guess_bounds(self.lon_p)
        self.lons_p = self.lonb_p[1:] - self.lonb_p[0:-1]
        
        self.latb = np.sort(self.data.latitude_1.values)
        self.lat = util.guess_from_bounds(self.latb)
        self.lats = self.latb[1:] - self.latb[0:-1]
        self.lat_p = np.concatenate(([-90], self.lat, [90]))
        self.latb_p = util.guess_bounds(self.lat_p)
        self.lats_p = self.latb_p[1:] - self.latb_p[0:-1]
        
        super(Z500MTS, self).import_coordinates()
    
    def z500(self, zone=zones.NoZone(), mode_lon=None, value_lon=None, mode_lat=None, value_lat=None,
             mode_t=None, value_t=None, new_start_year=None, new_end_year=None, new_month_list=None):
        print("__ Importing geopotential height z500.")
        return self.get(
            self.data.ht_mm_p.isel(p=0).rename({'longitude_1': 'longitudeb'}).rename({'latitude_1': 'latitudeb'}),
            zone, mode_lon, value_lon, mode_lat, value_lat, None, None, mode_t, value_t, new_start_year=new_start_year,
            new_end_year=new_end_year, new_month_list=new_month_list)


class SMMTS(HadCM3TS):
    """
    NOT IMPLEMENTED YET!!
    """
    
    def __init__(self, exp_name, start_year=None, end_year=None, month_list=None, chunks=None, verbose=True,
                 debug=False,
                 logger="print"):
        month_list = HadCM3DS.MONTHS if month_list is None else month_list  # To overcome mutable argument error
        super(SMMTS, self).__init__(exp_name, start_year, end_year, file_name="sm.monthly",
                                    month_list=month_list, chunks=chunks, verbose=verbose, debug=debug, logger=logger)
    
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
    """
    NOT IMPLEMENTED YET!!
    """
    
    def __init__(self, exp_name, start_year=None, end_year=None, month_list=None, chunks=None, verbose=True,
                 debug=False,
                 logger="print"):
        month_list = HadCM3DS.MONTHS if month_list is None else month_list  # To overcome mutable argument error
        super(SOILTMTS, self).__init__(exp_name, start_year, end_year, file_name="soiltemp.monthly",
                                       month_list=month_list, chunks=chunks, verbose=verbose, debug=debug,
                                       logger=logger)
    
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

# class HadCM3LSM(proc.LSM):
#
#     def __init__(self):
#         super(HadCM3LSM, self).__init__()
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
