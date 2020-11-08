import numpy as np
# import numpy.ma as ma
import abc


class Zone:
    
    def __init__(self, verbose):
        self.verbose = verbose
    
    @abc.abstractmethod
    def compact(self, cube):
        return
    
    @abc.abstractmethod
    def import_coordinates(self, data_source, lon, lat, z):
        pass


class NoZone(Zone):
    
    def __init__(self, verbose=False):
        super(NoZone, self).__init__(verbose)
    
    def compact(self, data_array):
        return data_array
    
    def import_coordinates(self, data_source=None, lon=None, lat=None, z=None):
        return self


class Box(Zone):
    
    def __init__(self, lon_min=None, lon_max=None, lat_min=None, lat_max=None, z_min=None, z_max=None,
                 lon=None, lat=None, z=None, data_source=None, verbose=False):
        
        super(Box, self).__init__(verbose)
        
        self.lon, self.lat, self.z = None, None, None
        self.lonb, self.latb, self.zb = None, None, None
        self.lons, self.lats, self.zs = None, None, None
        self.lon_p, self.lat_p, self.z_p = None, None, None
        self.lonb_p, self.latb_p, self.zb_p = None, None, None
        self.lons_p, self.lats_p, self.zs_p = None, None, None
        
        self.import_coordinates(data_source, lon, lat, z)
        
        # self.lon_min = np.min(self.lon) if lon_min is None and self.lon is not None else lon_min
        # self.lon_max = np.max(self.lon) if lon_max is None and self.lon is not None else lon_max
        # self.lat_min = np.min(self.lat) if lat_min is None and self.lat is not None else lat_min
        # self.lat_max = np.max(self.lat) if lat_max is None and self.lat is not None else lat_max
        # self.z_min = np.min(self.z) if z_min is None and self.z is not None else z_min
        # self.z_max = np.max(self.z) if z_max is None and self.z is not None else z_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.z_min = z_min
        self.z_max = z_max
        
        # if lsm is None:
        #     self.lsm = prc.LSM().default_lsm(lon, lat, z)
        # else:
        #     self.lsm = lsm
    
    def import_coordinates(self, data_source=None, lon=None, lat=None, z=None):
        
        if data_source is not None:
            self.lon, self.lat, self.z = data_source.lon, data_source.lat, data_source.z
            self.lonb, self.latb, self.zb = data_source.lonb, data_source.latb, data_source.zb
            self.lons, self.lats, self.zs = data_source.lons, data_source.lats, data_source.zs
            self.lon_p, self.lat_p, self.z_p = data_source.lon_p, data_source.lat_p, data_source.z_p
            self.lonb_p, self.latb_p, self.zb_p = data_source.lonb_p, data_source.latb_p, data_source.zb_p
            self.lons_p, self.lats_p, self.zs_p = data_source.lons_p, data_source.lats_p, data_source.zs_p
        else:
            self.lon = lon
            self.lat = lat
            self.z = z
        
        self.update()
        
        return self
    
    def get_indexes(self):
        
        if any([self.lon is None, self.lat is None, self.z is None]):
            print("Caution : Please import coordinates first")
            raise KeyError("Caution : Please import coordinates first")
        
        else:
            ilon_box = np.where(self.lon_min <= self.lon <= self.lon_max)
            ilat_box = np.where(self.lat_min <= self.lat <= self.lat_max)
            iz_box = np.where(self.z_min <= self.z <= self.z_max)
            return ilon_box, ilat_box, iz_box
    
    def compact(self, geo_da):
        # Test lon etc.
        
        if self.lon_min is not None:
            geo_da.data = geo_da.data.where(geo_da.data.longitude >= self.lon_min, drop=True)
        if self.lon_max is not None:
            geo_da.data = geo_da.data.where(geo_da.data.longitude <= self.lon_max, drop=True)
        if self.lat_min is not None:
            geo_da.data = geo_da.data.where(geo_da.data.latitude >= self.lat_min, drop=True)
        if self.lat_max is not None:
            geo_da.data = geo_da.data.where(geo_da.data.latitude >= self.lat_max, drop=True)
        if self.z_min is not None:
            geo_da.data = geo_da.data.where(geo_da.data.z >= self.z_min, drop=True)
        if self.z_max is not None:
            geo_da.data = geo_da.data.where(geo_da.data.z <= self.z_max, drop=True)
        
        print("____ Data compacted to the zone.")
        geo_da.fit_coordinates_to_data()
        return geo_da
    
    def update(self):
        
        if self.lon is not None and self.lon_min is None:
            self.lon_min = np.min(self.lon)
        if self.lon is not None and self.lon_max is None:
            self.lon_max = np.max(self.lon)
        if self.lat is not None and self.lat_min is None:
            self.lat_min = np.min(self.lat)
        if self.lat is not None and self.lat_max is None:
            self.lat_max = np.max(self.lat)
        if self.z is not None and self.z_min is None:
            self.z_min = np.min(self.z)
        if self.z is not None and self.z_max is None:
            self.z_max = np.max(self.z)
    
    def fit_coordinates_to_data(self):
        if self.lon_min is not None:
            self.lon = self.lon[np.where(self.lon >= self.lon_min)]
            self.lon_p = self.lon_p[np.where(self.lon_p >= self.lon_min)]
            self.lonb = self.lonb[np.where(self.lon >= self.lon_min)]  # not good, find something else.
            self.lonb_p = self.lonb_p[np.where(self.lon >= self.lon_min)]
            self.lons_p = self.lonb_p[1:] - self.lonb_p[0:-1]
        if self.lon_max is not None:
            self.lon = self.lon[np.where(self.lon <= self.lon_max)]
            self.lon_p = self.lon_p[np.where(self.lon_p <= self.lon_max)]
            self.lonb = self.lonb[np.where(self.lon <= self.lon_max)]  # not good, find something else.
            self.lonb_p = self.lonb_p[np.where(self.lon <= self.lon_max)]
            self.lons_p = self.lonb_p[1:] - self.lonb_p[0:-1]