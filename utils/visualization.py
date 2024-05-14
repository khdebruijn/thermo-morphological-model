import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Disable
def block_print():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enable_print():
    sys.stdout = sys.__stdout__

class BaseSimulationPlot(object):
    """
    ...
    """
    def __init__(self, sim):
        self.sim = sim
        
    def plot_transect(ax=None):
        pass

