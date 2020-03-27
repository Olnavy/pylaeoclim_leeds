import numpy as np
import pathlib


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
    mean = np.convolve(data, np.ones((n)), mode="full")
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
        lon : numpy 2D array
            grid longitudes coordinates
        lat : numpy 2D array
            grid latitudes coordinates
        lon_p : float?
            point longitude
        lat_p : float?
            point latitude
        inc :  float, optional
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


def coordinate_bounds(coordinates):
    """
    Create bound file for regriding
    :param coordinates: 1D and regular!
    :return:
    """
    step = coordinates[1] - coordinates[0]
    return [coordinates[0] - step / 2 + i * step for i in range(len(coordinates) + 1)]


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
    lat_b = coordinate_bounds(lat)
    surface = np.zeros((n_j, n_i))
    for i in range(n_i):
        for j in range(n_j):
            surface[j, i] = cell_area(n_i, lat_b[j], lat_b[j + 1])
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


# Generate
path2exp = generate_filepath(str(pathlib.Path(__file__).parent.absolute()) + "/path2exp")
path2lsm = generate_filepath(str(pathlib.Path(__file__).parent.absolute()) + "/path2lsm")
