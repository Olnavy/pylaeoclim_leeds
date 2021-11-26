import matplotlib.pyplot as plt
import abc
import numpy as np
import matplotlib.colors
import matplotlib.cm as cm


class PlotTemplate:
    
    def __init__(self, sav_path):
        self.sav_path = sav_path
    
    @abc.abstractmethod
    def core(self):
        pass
    
    def plot(self):
        self.core()
        plt.plot()
    
    def save(self):
        self.core()
        plt.savefig(self.sav_path)


# *****
# Norms
# *****

class Normalize(matplotlib.colors.Normalize):
    
    def __init__(self, clip=False, verbose=False, **kwargs):
        
        if 'vmin' in kwargs and 'vmax' in kwargs:
            print("__ Initialising the norm from vmin and vmax values")
            vmin = np.nanmin(kwargs['vmin'])
            vmax = np.nanmax(kwargs['vmax'])
        elif 'in_values' in kwargs:
            print("__ Initialsing the norm from the array sequence")
            vmin = np.nanmin([np.nanmin(a) for a in kwargs['in_values']])
            vmax = np.nanmax([np.nanmax(a) for a in kwargs['in_values']])
        else:
            raise KeyError("Please indicate either in_values or vmin and vmax")
        
        if verbose: print(f"vmin:{vmin}; vmax{vmax}")
        super(Normalize, self).__init__(vmin, vmax, clip)
    
    def generate_mappable(self, cmap):
        return cm.ScalarMappable(norm=self, cmap=cmap)


class TwoSlopeNorm(matplotlib.colors.TwoSlopeNorm):
    
    def __init__(self, vcenter, symetrical=False, verbose=False, **kwargs):
        
        if 'vmin' in kwargs and 'vmax' in kwargs:
            print("__ Initialising the norm from vmin and vmax values")
            vmin = np.nanmin(kwargs['vmin'])
            vmax = np.nanmax(kwargs['vmax'])
        elif 'in_values' in kwargs:
            print("__ Initialsing the norm from the array sequence")
            vmin = np.nanmin([np.nanmin(a) for a in kwargs['in_values']])
            vmax = np.nanmax([np.nanmax(a) for a in kwargs['in_values']])
        else:
            raise KeyError("Please indicate either in_values or vmin and vmax")
        if verbose: print(f"vmin:{vmin}; vmax{vmax}")
        super(TwoSlopeNorm, self).__init__(vcenter, vmin, vmax)
        if symetrical:
            self.make_symetrical()
    
    def generate_mappable(self, cmap):
        return cm.ScalarMappable(norm=self, cmap=cmap)
    
    def make_symetrical(self):
        half_range = np.max([np.abs(self.vmin - self.vcenter), np.abs(self.vmax - self.vcenter)])
        self.vmax = self.vcenter + half_range
        self.vmin = self.vcenter - half_range
