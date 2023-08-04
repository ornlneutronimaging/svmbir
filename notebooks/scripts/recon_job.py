# %%
import os
import math
import numpy as np
import svmbir
import glob
from tqdm import tqdm, trange
from skimage.io import imread
import dxchange

import algotom.prep.calculation as calc
import algotom.rec.reconstruction as rec
import multiprocessing as mp
import matplotlib.pyplot as plt
import logging
import timeit
# %%
logging.basicConfig(filename='./output_september4_recon.log', format='%(asctime)s %(message)s', 
                    level=logging.DEBUG)
# %%
# functions define
def load_sino_stack(z_start, z_numSlice, sino_lst, norm_path):
    sino_temp = np.load(os.path.join(norm_path, sino_lst[0]))
    num_ang, _, num_cols = sino_temp.shape
    sino = np.zeros((num_ang, z_numSlice, num_cols))
    for i in range(z_start, z_start + z_numSlice):
        sino[:, i-z_start] = np.load(os.path.join(norm_path, sino_lst[i]))[:,0]
    return sino

def recon_task(z_start):
    _t0 = timeit.default_timer()
    sino = load_sino_stack(z_start, z_numSlice, sino_lst, norm_path)[:1596, ]
    _sino = np.moveaxis(sino, 1, 0)
    rot_center = calc.find_center_vo(_sino[int(z_numSlice/2-1),])
    logging.info('Estimated center of rotation:{}'.format(rot_center) )
    _, _, num_cols = sino.shape
    center_offset = rot_center-num_cols/2
    """
    x_hat = rec.gridrec_reconstruction(sino, rot_center, angles=angles_rad, apply_log=False,
                                       ratio=1.0, filter_name='shepp', pad=100,)
    """
    """
    x_hat = rec.fbp_reconstruction(sino, rot_center, angles=angles_rad, apply_log=False, ramp_win=None,
                                   filter_name='hann', pad=None, pad_mode='edge', 
                                   gpu=True, block=(16, 16))
    x_hat= np.moveaxis(x_hat, 1, 0)
    """
    #"""
    x_hat = svmbir.recon(sino, angles_rad,
                         num_rows = num_cols, num_cols = num_cols, center_offset=center_offset, 
                         sharpness=sharpness,  verbose = 0, positivity = False, num_threads = 128,
                         svmbir_lib_path = '/netdisk/imaging/data_hfir/for_shimin/', 
                         max_resolutions=4)
    #"""
    
    dxchange.writer.write_tiff_stack(x_hat[25:,], 
                                     fname=os.path.join(recon_path, file_prefix),
                                     start=z_start+25, overwrite=True)
    _t1 = timeit.default_timer()
    logging.info("...batch starts from {} done... costed {} s".format(z_start, (_t1-_t0)))
# %%
## load angle list
sharpness = 0.0
snr_db = 40
base_path = "/netdisk/imaging/data_hfir/for_shimin/rat_femur/September4_2020"
angles = np.load(os.path.join(base_path,"angles.npy"))
angles_rad = np.pi - np.deg2rad(angles[:1596])

norm_path = os.path.join(base_path, 'norm_final')
sino_lst = os.listdir(norm_path)
sino_lst.sort()

# %%
z_numSlice = 30
file_prefix = '_'.join(sino_lst[0].split('_')[1:-1])
recon_path = os.path.join(base_path, "recon")
# %%
logging.info("start Recon...")
t0 = timeit.default_timer()
with mp.Pool(1) as pool:
    # create a set of word hashes
    pool.map(recon_task, np.arange(2130, 2140, 20))
t1 = timeit.default_timer()
logging.info("Total time cost: {} min".format((t1-t0)/60))
# %%
## Debugging code
"""
im1 = imread(os.path.join(base_path, "recon", "Sample_SevenFiveFive_0010_01300.tiff"))
plt.imshow(im1[500:1400, 400:-100], cmap = 'gray', vmin = 0)
plt.colorbar()
"""

# %%
