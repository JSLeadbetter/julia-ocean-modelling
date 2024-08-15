import numpy as np
from matplotlib import pyplot as plt

import pyqg
from pyqg import diagnostic_tools as tools

from time import perf_counter

YEAR = 24*60*60*360.0
MINUTE = 60

dt = 15.0*MINUTE
M = 128
T = 1*YEAR

m = pyqg.QGModel(
    tmax=T,
    twrite=10000,
    tavestart=5*YEAR,
    nx=M,
    dt=dt,
    log_level=2
)

t_start = perf_counter()
m.run()
t_end = perf_counter()
t_elapsed = t_end - t_start

print("Simulation time: ", t_elapsed)