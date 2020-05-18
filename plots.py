import matplotlib.pyplot as plt
import abc

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

