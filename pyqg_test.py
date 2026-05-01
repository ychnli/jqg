import numpy as np
from matplotlib import pyplot as plt
import pyqg
import os

def main():
    year = 24*60*60*360.
    m = pyqg.QGModel(tmax=10*year, twrite=10000, tavestart=5*year, nx=256, ny=256, dt=7200/4)
    m.run()

    m_ds = m.to_dataset()
    # os.makedirs("output", exist_ok=True)
    # m_ds.to_netcdf("output/pyqg_test.nc")
    # print("Saved to output/pyqg_test.nc")

    m_ds = m_ds.isel(time=-1)
    m_ds['q_upper'] = m_ds.q.isel(lev=0) + m_ds.Qy.isel(lev=0)*m_ds.y
    m_ds['q_upper'].attrs = {'long_name': 'upper layer PV anomaly'}
    m_ds.q_upper.plot()
    plt.show()


if __name__ == "__main__":
    main()
