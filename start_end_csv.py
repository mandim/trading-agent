import numpy as np

cache_dir = "cache_fx_EURUSD_D1"  # or whatever you used

bar_times = np.load(f"{cache_dir}/bar_times.npy", mmap_mode="r")
print("First bar:", bar_times[0])
print("Last bar:", bar_times[-1])
