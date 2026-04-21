import os
from pathlib import Path
import shutil
import torch
import mrcfile
import numpy as np
from PIL import Image

from ddw.utils.mrctools2 import load_data
standardize_full_tomos = True
tomo0_tensorfiles = Path(f"/home/nathan/Desktop/Ange-Louis/DDW2/testing/tomos/tomo0").glob("*.pt")
tomo1_tensorfiles = Path(f"/home/nathan/Desktop/Ange-Louis/DDW2/testing/tomos/tomo1").glob("*.pt")

fitting_counter, val_counter = 0, 0
# sans masque car je ne sais pas le faire
for k, (tomo0_tensorfile, tomo1_tensorfile) in enumerate(zip(tomo0_tensorfiles, tomo1_tensorfiles)):
	#print(k, tomo0_tensorfile, tomo1_tensorfile)
    tomo0 = load_data(tomo0_tensorfile).float()
    tomo1 = load_data(tomo1_tensorfile).float()
    if standardize_full_tomos:
        print(
            f"Standardizing tomogram '{Path(tomo0_tensorfile).stem}' before extracting sub-tomograms."
        )
        tomo0 -= tomo0.mean(); tomo1 -= tomo1.mean()
        tomo0 /= tomo0.std(); tomo1 /= tomo1.std()
    else:
        std = tomo0.std()
        if std < 1e-3:
            print(f"\
                WARNING: Standard deviation of '{Path(tomo0_tensorfile).stem}' is low ({std}), which may lead to issues during model fitting!\
                \nConsider setting 'standardize_full_tomos=True'.\
                \nIf you do so, you must also set 'standardize_full_tomos=True' for 'ddw refine-tomogram'.\
        ")

