import numpy as np

srcfile = "/home/ian/workplace/DevoBench/devo_bench_data/inv_depths_baseline.npy"
dstfile = "/home/ian/workplace/DevoBench/devo_bench_data/depths.npy"

inv_depth = np.load(srcfile)

depth = 1.0 / (inv_depth + 1e-4)
depth[np.isinf(depth)] = 0
np.save(dstfile, depth)
