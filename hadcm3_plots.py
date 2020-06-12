import numpy as np
import hadcm3_processing as hcm3
import zones
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import paleoclim_leeds.plots as plots


class OcnTtz(plots.PlotTemplate):
    
    def __init__(self, experiment, start_year, end_year, zone, sav_path=None):
        super().__init__(sav_path)
        self.experiment = experiment
        self.start_year = start_year
        self.end_year = end_year
        self.zone = zone
    
    def core(self):
        ts = hcm3.OCNTATS(self.experiment, self.start_year, self.end_year)
        
        zone_up = zones.Box(lon_min=self.zone.lon_min, lon_max=self.zone.lon_max,
                            lat_min=self.zone.lat_min, lat_max=self.zone.lat_max,
                            z_min=-1000)
        zone_down = zones.Box(lon_min=self.zone.lon_min, lon_max=self.zone.lon_max,
                              lat_min=self.zone.lat_min, lat_max=self.zone.lat_max,
                              z_max=-1000)
        
        temp_up = ts.temperature(zone=zone_up, mode_lon="mean", mode_lat="mean")
        values_up = temp_up.values(processing=False)
        temp_down = ts.temperature(zone=zone_down, mode_lon="mean", mode_lat="mean")
        values_down = temp_down.values(processing=False)
        
        norm = colors.Normalize(vmin=min(np.nanmin(values_up), np.nanmin(values_down)),
                                vmax=max(np.nanmax(values_up), np.nanmax(values_down)))
        
        years = np.linspace(self.start_year, self.end_year, len(temp_up.t))
        
        figMap = plt.figure(figsize=(15, 7), dpi=200)
        
        ax = figMap.add_subplot(111)
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
        ax.set_ylabel("Temperature Nordic Seas (°C)", labelpad=30)
        
        axUp = figMap.add_subplot(211)
        axDown = figMap.add_subplot(212, sharex=axUp)
        
        figMap.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.01)
        # figMap.tight_layout()
        
        pc_up = axUp.pcolormesh(years, temp_up.z, np.transpose(temp_up.values(processing=False)), cmap="RdYlBu_r",
                                norm=norm)
        axUp.axes.xaxis.set_visible(False)
        
        pc_down = axDown.pcolormesh(years, temp_down.z, np.transpose(temp_down.values(processing=False)),
                                    cmap="RdYlBu_r", norm=norm)
        axDown.ticklabel_format(style="sci")
        axDown.set_xlabel("Simulation Years")
        
        figMap.colorbar(mappable=pc_up, ax=[axUp, axDown], label="°C")
