import numpy as np
# import pathlib
import cftime
import scipy.signal as signal
import matplotlib.mlab as mlab
import matplotlib.colors
import scipy.ndimage as ndimage
import pandas as pd

# import xarray as xr


# Filtering functions

class ButterLowPass:
    
    def __init__(self, order, fc, fs, mult=1):
        self.order = order
        self.fc = fc * mult
        self.fs = fs
        self.filter = signal.butter(order, fc * mult, 'lp', fs=1, output='sos')
    
    def process(self, data):
        return signal.sosfiltfilt(self.filter, data - np.mean(data)) + np.mean(data)
    
    def plot(self, min_power=None, max_power=None, n_fq=100):
        """
        
        :param min_power:
        :param max_power:
        :param n_fq:
        :return: w : ndarray, The angular frequencies at which `h` was computed.
        :return: h : ndarray, The frequency response.
        """
        nyq = 0.5 * self.fs
        normal_cutoff = self.fc / nyq
        # b: Numerator of a linear filter; a: Denominator of a linear filter
        b, a = signal.butter(self.order, normal_cutoff, btype='low', analog=True)
        if min_power is not None and max_power is not None:
            return signal.freqs(b, a, worN=np.logspace(min_power, max_power, n_fq))
        return signal.freqs(b, a)


def psd(data, fs, scale_by_freq=False):
    """
    :param data: np.ndarray
    :param fs: The sampling frequency (samples per time unit). It is used to calculate the Fourier frequencies, freqs,
    in cycles per time unit.
    :param scale_by_freq: Whether the resulting density values should be scaled by the scaling frequency, which gives
     density in units of Hz^-1. This allows for integration over the returned frequency values.
    :return:
    """
    return mlab.psd(data - np.mean(data), NFFT=len(data), Fs=fs, scale_by_freq=scale_by_freq)


def fundamental_fq(density, fq):
    return fq[np.argmax(density)]


def butter_lowpass(cut_off, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cut_off / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=True)
    return b, a


class Grid:
    def __init__(self, lon, lat, z):
        self.lon = lon
        self.lat = lat
        self.z = z
        self.lon_b = guess_bounds(self.lon)
        self.lat_b = guess_bounds(self.lat)
        self.z_b = guess_bounds(self.z)
    
    def get_surface_matrix(self, n_t=0):
        matrix = surface_matrix(self.lon, self.lat)
        return matrix if n_t <= 0 else np.resize(matrix, (n_t, matrix.shape[0], matrix.shape[1]))
    
    def get_surface_ratio(self, n_t=0):
        matrix = surface_matrix(self.lon, self.lat)
        return matrix / np.sum(matrix) if n_t <= 0 else np.resize(matrix / np.sum(matrix),
                                                                  (n_t, matrix.shape[0], matrix.shape[1]))
    
    def get_volume_matrix(self, n_t=0):
        matrix = volume_matrix(self.lon, self.lat, self.z)
        return matrix if n_t <= 0 else np.resize(matrix, (n_t, matrix.shape[0], matrix.shape[1], matrix.shape[2]))


def cell_area(n_lon, lat1, lat2):
    """
    Area of a cell on a regular lon-lat grid.
    :param n_lon: number of longitude divisions
    :param lat1: bottom of the cell
    :param lat2: top of the cell
    :return:
    """
    r = 6371000
    lat1_rad, lat2_rad = 2 * np.pi * lat1 / 360, 2 * np.pi * lat2 / 360
    return 2 * np.pi * r ** 2 * np.abs(np.sin(lat1_rad) - np.sin(lat2_rad)) / n_lon


def surface_matrix(lon, lat):
    """
    Compute a matrix with all the surfaces values.
    :param lon:
    :param lat:
    :return:
    """
    n_j, n_i = len(lat), len(lon)
    lat_b = guess_bounds(lat)
    surface = np.zeros((n_j, n_i))
    for i in range(n_i - 1):
        for j in range(n_j - 1):
            surface[j, i] = cell_area(n_i, lat_b[j], lat_b[j + 1])
    return surface


def volume_matrix(lon, lat, z):
    n_lat, n_lon, n_z = len(lat), len(lon), len(z)
    if any([n_lat == 1, n_lon == 1, n_z == 1]):
        raise ValueError(f"Dimensions length must be >= 1.")
    lat_b = guess_bounds(lat)
    z_b = guess_bounds(z)
    volume = np.zeros((n_lat, n_lon, n_z))
    for i in range(n_lat):
        for j in range(n_lon):
            for k in range(n_z):
                volume[i, j, k] = cell_area(n_lon, lat_b[i], lat_b[i + 1]) * np.abs(z_b[k + 1] - z_b[k])
    return volume


def rmean(data, n):
    try:
        return pd.Series(data).rolling(window=n, min_periods=1, center=True).mean().values
    except TypeError as error:
        print(error)
        print("Returning initial tab.")
        return data


def running_mean(data, n, axis=0):
    """
    Running mean on n years for a 1D or 2D array. Only use the past values.
    Parameters
    ----------
    data : numpy 1D or 2D array with time as first dimension
        data to process the running mean
    n : int
        number of years to perform the running mean
    axis : int
        axis
    Returns
    -------
    numpy 1D or 2D array
        new averaged data
    """
    
    try:
        if data.ndim == 1:
            # mean = pd.Series(data).rolling(window=n, min_periods=1, center=True).mean().values
            mean = np.convolve(data, np.ones(n), mode="full")
        elif data.ndim == 2:
            n_i = data.shape[axis]
            n_j = data.shape[1] if axis == 0 else data.shape[0]
            mean = np.zeros((n_i, n_j))
            for j in range(n_j):
                # mean[:, j] = pd.Series(data[:, j]).rolling(window=n, min_periods=1, center=True).mean().values
                mean[:, j] = np.convolve(data[:, j], np.ones(n), mode="full")[:len(data)]
        
        else:
            raise ValueError("Dimensions >2 not implemented yet.")
    except TypeError as error:
        print(error)
        print("Returning initial tab.")
        return data
    
    out_mean = np.zeros(data.shape)
    for i in range(len(data)):
        if i + 1 < n:
            out_mean[i] = mean[i] / (i + 1)
        else:
            out_mean[i] = mean[i] / n
    return out_mean


def coordinate_to_index(longitude, latitude, target_lon, target_lat):
    """
        Find the closet -or at least pretty clos- indexes from a coordiantes grid to a point.
        inc should have the magnitude of the grd size
        Parameters
        ----------
        longitude : numpy 2D array
            grid longitudes coordinates
        latitude : numpy 2D array
            grid latitudes coordinates
        target_lon : float?
            point longitude
        target_lat : float?
            point latitude
            step for the research (default is 0,5)
        Returns
        -------
        int, int
            indexes that match the longitudes and the latitudes.
        """
    i_out = (np.abs(latitude - target_lat)).argmin()
    j_out = (np.abs(longitude - target_lon)).argmin()
    
    return j_out, i_out


def lon_to_index(longitude, target_lon):
    return (np.abs(longitude - target_lon)).argmin()


def lat_to_index(latitude, target_lat):
    return (np.abs(latitude - target_lat)).argmin()


def z_to_index(z, target_z):
    return (np.abs(z - target_z)).argmin()


def guess_bounds(coordinate):
    if coordinate is not None:
        if len(coordinate) <= 1:
            coordinateb = coordinate
        else:
            coordinateb = [(coordinate[i] + coordinate[i + 1]) / 2 for i in range(len(coordinate) - 1)]
            coordinateb = np.append((3 * coordinate[0] - coordinate[1]) / 2, coordinateb)
            coordinateb = np.append(coordinateb, (3 * coordinate[-1] - coordinate[-2]) / 2)
        return np.array(coordinateb)
    else:
        raise ValueError("Empty coordinate.")


def guess_from_bounds(coordinateb):
    if coordinateb is not None:
        if len(coordinateb) <= 1:
            coordinate = coordinateb
        else:
            coordinate = [(coordinateb[i] + coordinateb[i + 1]) / 2 for i in range(len(coordinateb) - 1)]
        return np.array(coordinate)
    else:
        raise ValueError("Empty coordinate.")


def compute_steps(coordinate):
    if coordinate is not None:
        if len(coordinate) <= 1:
            coordinates = 0
        else:
            coordinates = [(coordinate[i] - coordinate[i + 1]) for i in range(len(coordinate) - 1)]
        return np.array(coordinates)
    else:
        raise ValueError("Empty coordinate.")


# def guess_bounds_old(coordinate, mode):
#     """
#     DEPRECATED
#     """
#     if mode == "lon":
#         lon_b = []
#         if coordinate is not None:
#             if lon_b is None:
#                 lon_b = [(coordinate[i] + coordinate[i + 1]) / 2 for i in range(len(coordinate) - 1)]
#                 lon_b = np.append((3 * coordinate[0] - coordinate[1]) / 2, lon_b)
#                 lon_b = np.append(lon_b, (3 * coordinate[-1] - coordinate[-2]) / 2)
#             elif len(coordinate) <= 1:
#                 lon_b = coordinate
#             elif len(lon_b) == len(coordinate):
#                 lon_b = np.append(lon_b, 2 * lon_b[-1] - lon_b[-2])
#             elif len(lon_b) == len(coordinate) + 1:
#                 pass
#             elif len(lon_b) == len(coordinate) - 1:
#                 lon_b = np.append(2 * lon_b[1] - lon_b[2], lon_b)
#                 lon_b = np.append(lon_b, 2 * lon_b[-1] - lon_b[-2])
#             else:
#                 lon_b = [(coordinate[i] + coordinate[i + 1]) / 2 for i in range(len(coordinate) - 1)]
#                 lon_b = np.append((3 * coordinate[0] - coordinate[1]) / 2, lon_b)
#                 lon_b = np.append(lon_b, (3 * coordinate[-1] - coordinate[-2]) / 2)
#             return lon_b
#         else:
#             return None
#
#     if mode == "lat":
#         if coordinate is not None:
#             return coordinate
#         else:
#             return None
#
#     if mode == "z":
#         if coordinate is not None:
#             z_b = []
#             if z_b is None:
#                 z_b = [(coordinate[i] + coordinate[i + 1]) / 2 for i in range(len(coordinate) - 1)]
#                 z_b = np.append((3 * coordinate[0] - coordinate[1]) / 2, z_b)
#                 z_b = np.append(z_b, (3 * coordinate[-1] - coordinate[-2]) / 2)
#             elif len(coordinate) <= 1:
#                 z_b = coordinate
#             elif len(z_b) == len(coordinate):
#                 z_b = np.append(z_b, 2 * z_b[-1] - z_b[-2])
#             elif len(z_b) == len(coordinate) + 1:
#                 pass
#             elif len(z_b) == len(coordinate) - 1:
#                 z_b = np.append(2 * z_b[1] - z_b[2], z_b)
#                 z_b = np.append(z_b, 2 * z_b[-1] - z_b[-2])
#             else:
#                 z_b = [(coordinate[i] + coordinate[i + 1]) / 2 for i in range(len(coordinate) - 1)]
#                 z_b = np.append((3 * coordinate[0] - coordinate[1]) / 2, z_b)
#                 z_b = np.append(z_b, (3 * coordinate[-1] - coordinate[-2]) / 2)
#             return z_b
#         else:
#             return None


def generate_filepath(path):
    """
    Generate a filepath dictionary from a txt file.

    DEPRECATED

    Returns
    -------
    dict
        source_name (values) to experiment_name (keys) connections
    """
    result_dict = dict()
    with open(path) as f:
        for line in f:
            (key, val, trash) = line.split(";")  # Certainement mieux à faire que trasher...
            result_dict[key] = val
    return result_dict


def generate_input(path):
    """
    Generate an input dictionary from a txt file.

    Returns
    -------
    dict
        set of inputs (values) for a experiment (keys)
    """
    result_dict = dict()
    with open(path) as f:
        for line in f:
            if line[0] == '#':
                continue
            key = line.split(";")[0]
            val = line.split(";")[1:-1]
            result_dict[key] = val
    return result_dict


# TIME

def t_to_index(t, target_t: cftime.Datetime360Day):
    return (abs(t - target_t)).argmin()


def months_to_number(month_list):
    try:
        conversion = {'ja': 1, 'fb': 2, 'mr': 3, 'ar': 4, 'my': 5, 'jn': 6, 'jl': 7, 'ag': 8, 'sp': 9, 'ot': 10,
                      'nv': 11, 'dc': 12}
        return [int(month) if isinstance(month, int) or month.isdigit() else conversion[month] for month in
                month_list]
    except ValueError as error:
        print(error)


def cycle_lon(array):
    if array.ndim > 1:
        return np.append(array, array[:, 0][:, np.newaxis], axis=1)
    else:
        return np.append(array, array[0])


def cycle_box(lon_min, lon_max, lat_min, lat_max):
    return [[lon_min, lon_min, lon_max, lon_max, lon_min],
            [lat_min, lat_max, lat_max, lat_min, lat_min]]


def print_coordinates(name, coordinate):
    coordinate = np.array(coordinate)
    if coordinate is None:
        return f"{name}: None"
    if len(coordinate.shape) == 1:
        if isinstance(coordinate, np.float32) or isinstance(coordinate, float):
            return f"{name}: [{coordinate}; 1]"
        elif len(coordinate) >= 2:
            return f"{name}: [{coordinate[0]}; {coordinate[1]}; ...; {coordinate[-2]}; {coordinate[-1]}; " \
                   f"{len(coordinate)}]"
        elif len(coordinate) == 1:
            return f"{name}: [{coordinate[0]}; {len(coordinate)}]"
        else:
            return f"{name}: Null"
    elif len(coordinate.shape) == 2:
        if isinstance(coordinate, np.float32) or isinstance(coordinate, float):
            return f"{name}: [{coordinate}; 1]"
        elif len(coordinate[0]) >= 2:
            return f"{name}: [{coordinate[0, 0]}; {coordinate[0, 1]}; ...; {coordinate[-1, -2]}; " \
                   f"{coordinate[-1, -1]}; {coordinate.shape}]"
        elif len(coordinate) == 1:
            return f"{name}: [{coordinate[0, 0]}; {coordinate.shape}]"
        else:
            return f"{name}: Null"


def trunc_cmap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def get_com(cube, lon, lat):
    if cube.ndim == 2:
        com = ndimage.measurements.center_of_mass(np.where(np.isnan(cube), 0, cube))
        return lon[int(com[1])], lat[int(com[0])]
    elif cube.ndim == 3:
        com = []
        maxmean = np.nanmax(np.nanmean(cube, axis=(1, 2)))
        for t in range(cube.shape[0]):
            weight = np.nanmean(cube[t]) / maxmean
            coordinates = get_com(cube[t], lon, lat)
            com.append((coordinates[0], coordinates[1], weight))
        return com


def get_hs(cube, lon, lat):
    if cube.ndim == 2:
        hs = np.unravel_index(np.argmax(np.where(np.isnan(cube), 0, cube), axis=None), cube.shape)
        hmax = np.nanmax(cube)
        return lon[int(hs[1])], lat[int(hs[0])], hmax
    elif cube.ndim == 3:
        hs = []
        for t in range(cube.shape[0]):
            coordinates = get_hs(cube[t], lon, lat)
            hs.append((coordinates[0], coordinates[1], coordinates[2]))
        return hs


def sub_average(array, chunks):
    array_split = np.array_split(array, len(array) / 100, axis=0)
    array_mean = np.zeros((len(array_split), array_split[0].shape[1], array_split[0].shape[2]))
    for t in range(len(array_mean)):
        array_mean[t] = np.mean(array_split[t], axis=0)
    return array_mean


def vector_stress(uarray, varray):
    return np.sqrt(uarray ** 2 + varray ** 2)

def density(t, s, order=2):
    '''
    Fofonoff 1985
    :param t:
    :param s:
    :param order:
    :return:
    '''
    A = [999.842594, 3.79e-2, -9.09e-3, 1.00e-4]
    B = [8.24e-1, -4.09e-3, 7.64e-5]
    C = [-5.72e-3, 1.02e-4]
    D = [4.8314e-4]
    
    if order == 3:
        return A[0] + A[1] * t + A[2] * t ** 2 + A[3] * t ** 3 + \
               B[0] * s + B[1] * s * t + B[2] * s * t ** 2+ \
               C[0] * np.abs(s) ** (3 / 2) + C[1] * np.abs(s) ** (3 / 2) * t + C[1] * np.abs(s) ** (3 / 2) * t ** 2 + \
               D[0] * s ** 2
    else:
        return A[0] + A[1] * t + A[2] * t ** 2 + \
               B[0] * s + B[1] * s * t + \
               C[0] * np.abs(s) ** (3 / 2) + C[1] * np.abs(s) ** (3 / 2) * t +\
               D[0] * s ** 2


def density_cube(temp, sal):
    n_z, n_lat, n_lon = temp.shape
    density_out = np.zeros(temp.shape)
    for i_z in range(n_z):
        for i_lat in range(n_lat):
            for i_lon in range(n_lon):
                density_out[i_z, i_lat, i_lon] = density(temp[i_z, i_lat, i_lon], sal[i_z, i_lat, i_lon])
    return density_out

# Generate
# path2lsm = generate_filepath(str(pathlib.Path(__file__).parent.absolute()) + "/resources/path2lsm")
