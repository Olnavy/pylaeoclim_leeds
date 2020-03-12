import processing as proc
import xarray as xr
import util_hadcm3 as util


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

    def fit_lsm_ds(self):
        pass

    def lsm_from_ds(self):
        pass
