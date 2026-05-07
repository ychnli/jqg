import numpy as np
from matplotlib import pyplot as plt
import pyqg
import os

def main():
    year = 24*60*60*360.
    m = pyqg.QGModel(tmax=10*year, twrite=10000, tavestart=5*year, nx=128, ny=128, dt=3600)
    m.run()

    m_ds = m.to_dataset()
    os.makedirs("output", exist_ok=True)
    m_ds.to_zarr("output/pyqg_128_reference.zarr")
    print("Saved to output/pyqg_128_reference.zarr")

    m_ds = m_ds.isel(time=-1)
    m_ds['q_upper'] = m_ds.q.isel(lev=0) + m_ds.Qy.isel(lev=0)*m_ds.y
    m_ds['q_upper'].attrs = {'long_name': 'upper layer PV anomaly'}
    m_ds.q_upper.plot()
    plt.savefig("output/pyqg_128_reference_q_upper.png", dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
