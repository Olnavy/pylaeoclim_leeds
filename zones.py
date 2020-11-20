import numpy as np
# import numpy.ma as ma
import abc
# import pylaeoclim_leeds.util_hadcm3 as util


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
        
        # self.lon, self.lat, self.z = None, None, None
        # self.lonb, self.latb, self.zb = None, None, None
        # self.lons, self.lats, self.zs = None, None, None
        # self.lon_p, self.lat_p, self.z_p = None, None, None
        # self.lonb_p, self.latb_p, self.zb_p = None, None, None
        # self.lons_p, self.lats_p, self.zs_p = None, None, None
        
        # self.import_coordinates(data_source, lon, lat, z)
        
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
    
    def __repr__(self):
        return f"lon_min: {self.lon_min}, lon_max: {self.lon_max}, lat_min: {self.lat_min}, lat_max: {self.lat_max}" \
               f"z_min: {self.z_min}, z_max: {self.z_max}\n" \
            # f"{util.print_coordinates('lon', self.lon)}; {util.print_coordinates('lon_p', self.lon_p)}\n" \
        # f"{util.print_coordinates('lonb', self.lonb)}; {util.print_coordinates('lonb_p', self.lonb_p)}\n" \
        # f"{util.print_coordinates('lons', self.lons)}; {util.print_coordinates('lons_p', self.lons_p)}\n" \
        # f"{util.print_coordinates('lat', self.lat)}; {util.print_coordinates('lat_p', self.lat_p)}\n" \
        # f"{util.print_coordinates('latb', self.latb)}; {util.print_coordinates('latb_p', self.latb_p)}\n" \
        # f"{util.print_coordinates('lats', self.lats)}; {util.print_coordinates('lats_p', self.lats_p)}\n" \
        # f"{util.print_coordinates('z', self.z)}; {util.print_coordinates('z_p', self.z_p)}\n" \
        # f"{util.print_coordinates('zb', self.zb)}; {util.print_coordinates('zb_p', self.zb_p)}\n" \
        # f"{util.print_coordinates('zs', self.zs)}; {util.print_coordinates('zs_p', self.zs_p)}\n" \
    
    def compact(self, geo_da):
        # Test lon etc.
        
        if self.lon_min is not None and 'longitude' in geo_da.data.dims:
            geo_da.data = geo_da.data.where(geo_da.data.longitude >= self.lon_min, drop=True)
        if self.lon_min is not None and 'longitudeb' in geo_da.data.dims:
            geo_da.data = geo_da.data.where(geo_da.data.longitudeb >= self.lon_min, drop=True)
        if self.lon_max is not None and 'longitude' in geo_da.data.dims:
            geo_da.data = geo_da.data.where(geo_da.data.longitude <= self.lon_max, drop=True)
        if self.lon_max is not None and 'longitudeb' in geo_da.data.dims:
            geo_da.data = geo_da.data.where(geo_da.data.longitudeb <= self.lon_max, drop=True)
        if self.lat_min is not None and 'latitude' in geo_da.data.dims:
            geo_da.data = geo_da.data.where(geo_da.data.latitude >= self.lat_min, drop=True)
        if self.lat_min is not None and 'latitudeb' in geo_da.data.dims:
            geo_da.data = geo_da.data.where(geo_da.data.latitudeb >= self.lat_min, drop=True)
        if self.lat_max is not None and 'latitude' in geo_da.data.dims:
            geo_da.data = geo_da.data.where(geo_da.data.latitude <= self.lat_max, drop=True)
        if self.lat_max is not None and 'latitudeb' in geo_da.data.dims:
            geo_da.data = geo_da.data.where(geo_da.data.latitudeb <= self.lat_max, drop=True)
        if self.z_min is not None and 'z' in geo_da.data.dims:
            geo_da.data = geo_da.data.where(geo_da.data.z >= self.z_min, drop=True)
        if self.z_min is not None and 'zb' in geo_da.data.dims:
            geo_da.data = geo_da.data.where(geo_da.data.zb >= self.z_min, drop=True)
        if self.z_max is not None and 'z' in geo_da.data.dims:
            geo_da.data = geo_da.data.where(geo_da.data.z <= self.z_max, drop=True)
        if self.z_max is not None and 'zb' in geo_da.data.dims:
            geo_da.data = geo_da.data.where(geo_da.data.zb <= self.z_max, drop=True)
        
        print("____ Data compacted to the zone.")
        # geo_da.fit_coordinates_to_data()
        return self.fit_coordinates_to_data(geo_da)
    
    def fit_coordinates_to_data(self, geo_da):
        """
        A reprendre
        :param geo_da:
        :return:
        """
        if self.lon_min is not None:
            condition = np.where(geo_da.lon >= self.lon_min)
            conditionb = np.where(geo_da.lonb >= self.lon_min)
            lon = geo_da.lon[condition]
            lon_p = geo_da.lon_p[condition]
            lonb = geo_da.lonb[conditionb]  # not good, find something else.
            lonb_p = geo_da.lonb_p[conditionb]
            lons = geo_da.lonb[1:] - geo_da.lonb[0:-1]
            lons_p = geo_da.lonb_p[1:] - geo_da.lonb_p[0:-1]
            geo_da.lon, geo_da.lonb, geo_da.lons = lon, lonb, lons
            geo_da.lon_p, geo_da.lonb_p, geo_da.lons_p = lon_p, lonb_p, lons_p
            geo_da.proc_lon = False
        if self.lon_max is not None:
            condition = np.where(geo_da.lon <= self.lon_max)
            conditionb = np.where(geo_da.lonb <= self.lon_max)
            lon = geo_da.lon[condition]
            lon_p = geo_da.lon_p[condition]
            lonb = geo_da.lonb[conditionb]  # not good, find something else.
            lonb_p = geo_da.lonb_p[conditionb]
            lons = geo_da.lonb[1:] - geo_da.lonb[0:-1]
            lons_p = geo_da.lonb_p[1:] - geo_da.lonb_p[0:-1]
            geo_da.lon, geo_da.lonb, geo_da.lons = lon, lonb, lons
            geo_da.lon_p, geo_da.lonb_p, geo_da.lons_p = lon_p, lonb_p, lons_p
            geo_da.proc_lon = False
        if self.lat_min is not None:
            condition = np.where(geo_da.lat >= self.lat_min)
            conditionb = np.where(geo_da.latb >= self.lat_min)
            lat = geo_da.lat[condition]
            lat_p = geo_da.lat_p[condition]
            latb = geo_da.latb[conditionb]  # not good, find something else.
            latb_p = geo_da.latb_p[conditionb]
            lats = geo_da.latb[1:] - geo_da.latb[0:-1]
            lats_p = geo_da.latb_p[1:] - geo_da.latb_p[0:-1]
            geo_da.lat, geo_da.latb, geo_da.lats = lat, latb, lats
            geo_da.lat_p, geo_da.latb_p, geo_da.lats_p = lat_p, latb_p, lats_p
            geo_da.proc_lat = False
        if self.lat_max is not None:
            condition = np.where(geo_da.lat <= self.lat_max)
            conditionb = np.where(geo_da.latb <= self.lat_max)
            lat = geo_da.lat[condition]
            lat_p = geo_da.lat_p[condition]
            latb = geo_da.latb[conditionb]  # not good, find something else.
            latb_p = geo_da.latb_p[conditionb]
            lats = geo_da.latb[1:] - geo_da.latb[0:-1]
            lats_p = geo_da.latb_p[1:] - geo_da.latb_p[0:-1]
            geo_da.lat, geo_da.latb, geo_da.lats = lat, latb, lats
            geo_da.lat_p, geo_da.latb_p, geo_da.lats_p = lat_p, latb_p, lats_p
            geo_da.proc_lat = False
        if self.z_min is not None:
            condition = np.where(geo_da.z >= self.z_min)
            conditionb = np.where(geo_da.zb >= self.z_min)
            z = geo_da.z[condition]
            z_p = geo_da.z_p[condition]
            zb = geo_da.zb[conditionb]  # not good, find something else.
            zb_p = geo_da.zb_p[conditionb]
            zs = geo_da.zb[1:] - geo_da.zb[0:-1]
            zs_p = geo_da.zb_p[1:] - geo_da.zb_p[0:-1]
            geo_da.z, geo_da.zb, geo_da.zs = z, zb, zs
            geo_da.z_p, geo_da.zb_p, geo_da.zs_p = z_p, zb_p, zs_p
            geo_da.proc_z = False
        if self.z_max is not None:
            condition = np.where(geo_da.z <= self.z_max)
            conditionb = np.where(geo_da.zb <= self.z_max)
            z = geo_da.z[condition]
            z_p = geo_da.z_p[condition]
            zb = geo_da.zb[conditionb]  # not good, find something else.
            zb_p = geo_da.zb_p[conditionb]
            zs = geo_da.zb[1:] - geo_da.zb[0:-1]
            zs_p = geo_da.zb_p[1:] - geo_da.zb_p[0:-1]
            geo_da.z, geo_da.zb, geo_da.zs = z, zb, zs
            geo_da.z_p, geo_da.zb_p, geo_da.zs_p = z_p, zb_p, zs_p
            geo_da.proc_z = False
        return geo_da
    
    def import_coordinates(self, data_source=None, lon=None, lat=None, z=None):
        pass
    
    # TO RECREATE FOR get_indexes!!
    #     if data_source is not None:
    #         self.lon, self.lat, self.z = data_source.lon, data_source.lat, data_source.z
    #         self.lonb, self.latb, self.zb = data_source.lonb, data_source.latb, data_source.zb
    #         self.lons, self.lats, self.zs = data_source.lons, data_source.lats, data_source.zs
    #         self.lon_p, self.lat_p, self.z_p = data_source.lon_p, data_source.lat_p, data_source.z_p
    #         self.lonb_p, self.latb_p, self.zb_p = data_source.lonb_p, data_source.latb_p, data_source.zb_p
    #         self.lons_p, self.lats_p, self.zs_p = data_source.lons_p, data_source.lats_p, data_source.zs_p
    #     else:
    #         self.lon = lon
    #         self.lat = lat
    #         self.z = z
    #
    #     # self.update()
    #     print(data_source)
    #     return self
    
    def get_indexes(self, lon, lat, z):
        
        if any([lon is None, lat is None, z is None]):
            print("Caution : Please import coordinates first")
            raise KeyError("Caution : Please import coordinates first")
        
        else:
            ilon_box = np.where(self.lon_min <= lon <= self.lon_max)
            ilat_box = np.where(self.lat_min <= lat <= self.lat_max)
            iz_box = np.where(self.z_min <= z <= self.z_max)
            return ilon_box, ilat_box, iz_box
    
    # def update(self):
    #     # A SUPPRIMER?
    #     if self.lon is not None and self.lon_min is None:
    #         self.lon_min = np.min(self.lon)
    #     if self.lon is not None and self.lon_max is None:
    #         self.lon_max = np.max(self.lon)
    #     if self.lat is not None and self.lat_min is None:
    #         self.lat_min = np.min(self.lat)
    #     if self.lat is not None and self.lat_max is None:
    #         self.lat_max = np.max(self.lat)
    #     if self.z is not None and self.z_min is None:
    #         self.z_min = np.min(self.z)
    #     if self.z is not None and self.z_max is None:
    #         self.z_max = np.max(self.z)
