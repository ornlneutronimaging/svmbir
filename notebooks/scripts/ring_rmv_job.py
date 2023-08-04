# %%
import os
import math
import numpy as np
import glob
import algotom.prep.calculation as calc
import bm3d_streak_removal as bm3d
from pathlib import Path
import multiprocessing as mp
import logging
import timeit
# %%
logging.basicConfig(filename='./output_ang21_ring.log', format='%(asctime)s %(message)s', 
                    level=logging.DEBUG)
# %%
def task(i):
    _t0 = timeit.default_timer()
    _norm_bm3d = bm3d.extreme_streak_attenuation(norm_rm_tilt[:,i])
    x_rm = bm3d.multiscale_streak_removal(_norm_bm3d)
    np.save(os.path.join(save_path, "sino_20200822_Femur_0010_{:0>5}".format(i+20)), x_rm)
    _t1 = timeit.default_timer()
    logging.info("{}th slice is done, costed {} s".format(i, (_t1-_t0)))
# %%


# main code
base_path = "/netdisk/imaging/data_hfir/for_shimin/rat_femur/Aug21_2020_restart"
logging.info("start loading from {} ...".format(base_path))
norm_rm_tilt = np.load(os.path.join(base_path, "norm_180_notilt_4sum.npy"))
logging.info("data is loaded")
# %%
save_path = os.path.join(base_path, "norm_final_4sum")
Path(save_path).mkdir(parents=True, exist_ok=True)
logging.info("Remove sinogram streak noise using multi-scale BM3D-based denoising procedure.")

t0 = timeit.default_timer()
with mp.Pool(4) as pool:
    # create a set of word hashes
    pool.map(task, range(norm_rm_tilt.shape[1]))
t1 = timeit.default_timer()
logging.info("Time cost {} min".format((t1-t0)/60))

