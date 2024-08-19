import pyqg
from time import perf_counter
import pandas as pd

YEAR = 24*60*60*365.0
MINUTE = 60
DAY = 60 * 60 * 24

dt = 30.0*MINUTE
T = 30*DAY
M_list = [8, 16, 32, 64, 128]

sample_size = 1
min_runtimes = []

for M in M_list:
    
    runtimes = []

    for i in range(sample_size):
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

        runtimes.append(t_elapsed)

    min_runtime = min(runtimes)
    min_runtimes.append(min_runtime)


df = pd.DataFrame({
    "M": M_list,
    "Time": min_runtimes
})

df.to_csv("python_data.csv")

for i in range(len(M_list)):
    M = M_list[i]
    t = min_runtimes[i]
    
    print(f"{M = }, Min runtime: {t = }")

