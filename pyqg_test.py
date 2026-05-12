from matplotlib import pyplot as plt
import pyqg
import os
import timeit


def main():
    year = 24 * 60 * 60 * 360  # seconds
    day = 24 * 60 * 60  # seconds
    m = pyqg.QGModel(
        tmax=5 * year, twrite=10000, tavestart=0, taveint=day, nx=128, ny=128, dt=3600
    )
    start_time = timeit.default_timer()
    m.run()
    end_time = timeit.default_timer()
    print(f"INFO: elapsed runtime = {end_time - start_time} seconds")

    m_ds = m.to_dataset()
    print("info: length of time dim = ", m_ds.time.size)
    os.makedirs("output", exist_ok=True)
    m_ds.to_zarr("output/pyqg_128_reference.zarr")
    print("Saved to output/pyqg_128_reference.zarr")

    m_ds = m_ds.isel(time=-1)
    m_ds["q_upper"] = m_ds.q.isel(lev=0) + m_ds.Qy.isel(lev=0) * m_ds.y
    m_ds["q_upper"].attrs = {"long_name": "upper layer PV anomaly"}
    m_ds.q_upper.plot()
    plt.savefig("output/pyqg_128_reference_q_upper.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
