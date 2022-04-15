import numpy as np
# import numpy.ma as ma
import abc
import pylaeoclim_leeds.util_hadcm3 as util
import shapely.geometry


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
    
    def __init__(self, lon_min=None, lon_max=None, lat_min=None, lat_max=None, z_min=None, z_max=None, verbose=False):
        
        super(Box, self).__init__(verbose)
        
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.z_min = z_min
        self.z_max = z_max
        
    def __repr__(self):
        return f"lon_min: {self.lon_min}, lon_max: {self.lon_max}, lat_min: {self.lat_min}, lat_max: {self.lat_max}, " \
               f"z_min: {self.z_min}, z_max: {self.z_max}\n" \
    
    
    def compact(self, geo_da):
        # Test lon etc.

        if 'longitude' in geo_da.data.dims:
            geo_da.data = geo_da.data.isel(longitude=slice(
                util.coordinate_to_index(geo_da.data.longitude, self.lon_min),
                util.coordinate_to_index(geo_da.data.longitude, self.lon_max)))
        if 'longitudeb' in geo_da.data.dims:
            geo_da.data = geo_da.data.isel(longitudeb=slice(
                util.coordinate_to_index(geo_da.data.longitudeb, self.lon_min),
                util.coordinate_to_index(geo_da.data.longitudeb, self.lon_max)))
        if 'latitude' in geo_da.data.dims:
            geo_da.data = geo_da.data.isel(latitude=slice(
                util.coordinate_to_index(geo_da.data.latitude, self.lat_min),
                util.coordinate_to_index(geo_da.data.latitude, self.lat_max)))
        if 'latitudeb' in geo_da.data.dims:
            geo_da.data = geo_da.data.isel(latitudeb=slice(
                util.coordinate_to_index(geo_da.data.latitudeb, self.lat_min),
                util.coordinate_to_index(geo_da.data.latitudeb, self.lat_max)))
        if 'z' in geo_da.data.dims:
            geo_da.data = geo_da.data.isel(z=slice(
                util.coordinate_to_index(geo_da.data.z, self.z_min),
                util.coordinate_to_index(geo_da.data.z, self.z_max)))
        if 'zb' in geo_da.data.dims:
            geo_da.data = geo_da.data.isel(zb=slice(
                util.coordinate_to_index(geo_da.data.zb, self.z_min),
                util.coordinate_to_index(geo_da.data.zb, self.z_max)))

            print("____ Data compacted to the zone.")
        # geo_da.fit_coordinates_to_data()
        return self.fit_coordinates_to_data(geo_da)
    
    def create_cycle(self, dim='2D'):
        """
        Create a 2D numpy array with coordinates for polygon drawing in matplotlib.
        :param dim:
        :return:
        """
        if dim=='2D':
            return [[self.lon_min, self.lon_min, self.lon_max, self.lon_max, self.lon_min],
                    [self.lat_min, self.lat_max, self.lat_max, self.lat_min, self.lat_min]]
        else:
            print("*** Not implemented yet, cycle aborted")
            return None
    
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

    def cycle_box(self):
        return util.cycle_box(self.lon_min, self.lon_max, self.lat_min, self.lat_max)

    def to_plane(self):
        return Box(self.lon_min, self.lon_max, self.lat_min, self.lat_max)


class Polygon:
    
    def __init__(self, bounds):
        self.bounds = self.test_lon_range(bounds)
        self.poly = shapely.geometry.Polygon(self.bounds)
    
    @staticmethod
    def test_lon_range(bounds_in):
        if any([np.abs(bounds_in[i][0]) > 360 for i in range(len(bounds_in))]):
            print("* At least one of the longitudes was outside of the [-360, 360] range. Modulo applied.")
            return [(bounds_in[i][0] % 360, bounds_in[i][1]) for i in range(len(bounds_in))]
        else:
            return bounds_in
    
    def contain(self, lon, lat):
        return np.logical_or(self.poly.contains(shapely.geometry.Point(lon % 360, lat)),
                             self.poly.contains(shapely.geometry.Point(lon % 360 - 360, lat)))
    
    def xy(self):
        return self.poly.exterior.coords.xy
    
    def create_mask(self, lon, lat):
        mask = np.empty((len(lat), len(lon)))
        mask[:] = np.nan
        for i in range(len(lon)):
            for j in range(len(lat)):
                if self.contain(lon[i], lat[j]):
                    mask[j, i] = 1
        return mask
    