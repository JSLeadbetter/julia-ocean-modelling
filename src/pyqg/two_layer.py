import numpy as np
import pyqg
from time import perf_counter

YEAR = 24*60*60*360.0
MINUTE = 60

dt = 15.0*MINUTE
M = 128
T = 1*YEAR

M_list = [16, 32, 64, 128]

for M in M_list:
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

    print(f"{M = }, Simulation time: ", t_elapsed)
