import numpy as np
import pathlib
import cftime
from typing import List


def running_mean(data, n):
    """
    Running mean on n years for a 1D array. Only use the past values.
    Parameters
    ----------
    data : list of numpy 1D array
        data to process the running mean
    n : int
        number of years to perform the running mean
    Returns
    -------
    list of numpy 1D array
        new averaged data
    """
    mean = np.convolve(data, np.ones(n), mode="full")
    out_mean = np.zeros((len(data)))
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


def guess_bounds(coordinate, mode):
    """
    """
    if mode == "lon":
        lon_b = []
        if coordinate is not None:
            if lon_b is None:
                lon_b = [(coordinate[i] + coordinate[i + 1]) / 2 for i in range(len(coordinate) - 1)]
                lon_b = np.append((3 * coordinate[0] - coordinate[1]) / 2, lon_b)
                lon_b = np.append(lon_b, (3 * coordinate[-1] - coordinate[-2]) / 2)
            elif len(coordinate) <= 1:
                lon_b = coordinate
            elif len(lon_b) == len(coordinate):
                lon_b = np.append(lon_b, 2 * lon_b[-1] - lon_b[-2])
            elif len(lon_b) == len(coordinate) + 1:
                pass
            elif len(lon_b) == len(coordinate) - 1:
                lon_b = np.append(2 * lon_b[1] - lon_b[2], lon_b)
                lon_b = np.append(lon_b, 2 * lon_b[-1] - lon_b[-2])
            else:
                lon_b = [(coordinate[i] + coordinate[i + 1]) / 2 for i in range(len(coordinate) - 1)]
                lon_b = np.append((3 * coordinate[0] - coordinate[1]) / 2, lon_b)
                lon_b = np.append(lon_b, (3 * coordinate[-1] - coordinate[-2]) / 2)
            return lon_b
        else:
            return None
    
    if mode == "lat":
        if coordinate is not None:
            return coordinate.append(2*coordinate[-1] - coordinate[-2])
        else:
            return None
    
    if mode == "z":
        if coordinate is not None:
            z_b = []
            if z_b is None:
                z_b = [(coordinate[i] + coordinate[i + 1]) / 2 for i in range(len(coordinate) - 1)]
                z_b = np.append((3 * coordinate[0] - coordinate[1]) / 2, z_b)
                z_b = np.append(z_b, (3 * coordinate[-1] - coordinate[-2]) / 2)
            elif len(coordinate) <= 1:
                z_b = coordinate
            elif len(z_b) == len(coordinate):
                z_b = np.append(z_b, 2 * z_b[-1] - z_b[-2])
            elif len(z_b) == len(coordinate) + 1:
                pass
            elif len(z_b) == len(coordinate) - 1:
                z_b = np.append(2 * z_b[1] - z_b[2], z_b)
                z_b = np.append(z_b, 2 * z_b[-1] - z_b[-2])
            else:
                z_b = [(coordinate[i] + coordinate[i + 1]) / 2 for i in range(len(coordinate) - 1)]
                z_b = np.append((3 * coordinate[0] - coordinate[1]) / 2, z_b)
                z_b = np.append(z_b, (3 * coordinate[-1] - coordinate[-2]) / 2)
            return z_b
        else:
            return None


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


def surface_matrix(lat, lon):
    """
    Compute a matrix with all the surfaces values.
    :param lon:
    :param lat:
    :return:
    """
    n_i, n_j = len(lat), len(lon)
    lat_b = guess_bounds(lat, "lat")
    surface = np.zeros((n_i, n_j))
    for i in range(n_i-1):
        for j in range(n_j):
            surface[i, j] = cell_area(n_j, lat_b[i], lat_b[i + 1])
    return surface


def generate_filepath(path):
    """
    Generate a filepath dictionary from a txt file.

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


# TIME

def t_to_index(t: List[cftime.Datetime360Day], target_t: cftime.Datetime360Day):
    return (abs(t - target_t)).argmin()


def months_to_number(month_list):
    try:
        conversion = {'ja': 1, 'fb': 2, 'mr': 3, 'ar': 4, 'my': 5, 'jn': 6, 'jl': 7, 'ag': 8, 'sp': 9, 'ot': 10,
                      'nv': 11, 'dc': 12}
        return [int(month) if isinstance(month, int) or month.isdigit() else conversion[month] for month in month_list]
    except ValueError as error:
        print(error)


def kelvin_to_celsius(array):
    return array - 273.15


def cycle_lon(array):
    return np.append(array, array[:, 0][:, np.newaxis], axis=1)


# Generate
path2expds = generate_filepath(str(pathlib.Path(__file__).parent.absolute()) + "/resources/path2expds")
path2expts = generate_filepath(str(pathlib.Path(__file__).parent.absolute()) + "/resources/path2expts")
path2lsm = generate_filepath(str(pathlib.Path(__file__).parent.absolute()) + "/resources/path2lsm")
