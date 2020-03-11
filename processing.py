class GeoDS:
    """
    Mother class to treat all files (proxies, model outputs...).
    Should fill it after treating an example of proxy file..

    ...

    Attributes
    ----------
    verbose : bool
          Determine whether to print the outputs in a logfile or in directly on the console.
          True is console - debug mode.  (default is False)

    Methods
    -------
    """
    
    def __init__(self, verbose=False):
        """
        Parameters
        ----------
        verbose : bool, optional
              Determine whether to print the outputs in a logfile or in directly on the console.
              True is console - debug mode.  (default is False)
        """
        
        self.verbose = verbose


class ModelDS(GeoDS):
    
    def __init__(self, verbose=False):
        """
        Parameters
        ----------
        verbose : bool, optional
            whether or not to display details about the computation.
            Outputs are printed.  (default is False)
        """
        
        super(ModelDS, self).__init__(verbose)


    def to_ncdf(self):
        """
        Save the dataset as a netcdf file
        :return:
        """
        
        pass
    
    def get(self,cube,zone):
        
        return zone.compact(cube)
    