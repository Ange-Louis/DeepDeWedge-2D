import os
from pathlib import Path

import numpy as np
import time
import torch
from scipy import spatial
from torch.utils.data import Dataset
from torchvision.transforms.functional import rotate

from src.ddw.utils.fourier2 import apply_fourier_mask_to_tomo
from src.ddw.utils.missing_wedge2 import (get_missing_wedge_mask,
                            get_rotated_missing_wedge_mask)
from src.ddw.utils.rotation2 import rotate_area

from src.ddw.utils.mrctools2 import save_mrc_data

BASE_SEED = 888


def safe_load(file_path, max_retries=3, delay=1):
    for attempt in range(max_retries):
        try:
            return torch.load(file_path)
        # except everything to catch all exceptions
        except Exception as e:
            print(f"Error loading {file_path}")
            if attempt == max_retries - 1:
                raise e  # Reraise if it's the last attempt
            print(f"Error message is: {e}")
            print(f"Retrying in {delay} seconds")
            time.sleep(delay)  # Wait before retrying



class SubtomoDataset(Dataset):
    """
    A torch dataset which produces the input-target sub-tomogram pairs used for model fitting. The directory 'subtomo_dir' must have the same structure as the output of the 'ddw prepare-data' command.
    """

    def __init__(
        self,
        subtomo_dir,
        mw_angle,
        crop_subtomos_to_size,
        rotate_subtomos=True,
        deterministic_rotations=False,
    ):
        super().__init__()
        self.subtomo_dir = subtomo_dir
        self.crop_subtomos_to_size = crop_subtomos_to_size
        self.mw_angle = mw_angle
        self.rotate_subtomos = rotate_subtomos
        self.deterministic_rotations = deterministic_rotations

        self.subtomo0_path = Path(self.subtomo_dir) / "subtomo0"
        self.subtomo1_path = Path(self.subtomo_dir) / "subtomo1"

        self.subtomo0_files = sorted(list(self.subtomo0_path.rglob("*.pt")))
        self.subtomo1_files = sorted(list(self.subtomo1_path.rglob("*.pt")))

    @property
    def rotate_subtomos(self):
        return self._rotate_subtomos

    @rotate_subtomos.setter
    def rotate_subtomos(self, rotate_subtomos):
        if not isinstance(rotate_subtomos, bool):
            raise ValueError("rotate_subtomos must be a boolean")
        self._rotate_subtomos = rotate_subtomos

    def _sample_rot_angle(self, index):
        seed = BASE_SEED + index if self.deterministic_rotations else None
        
        rng = np.random.default_rng(seed)

        rot_angle = torch.tensor(rng.uniform(0, 360), dtype=torch.float32)
        return rot_angle

    def __len__(self):
        return len(self.subtomo0_files)

    def __getitem__(self, index):
        # load subtomos
        subtomo0_file = str(self.subtomo0_files[index])
        subtomo0 = safe_load(subtomo0_file)
        subtomo1_file = str(self.subtomo1_files[index])
        subtomo1 = safe_load(subtomo1_file)
        # rotate subtomos
        if self.rotate_subtomos == True:
            rot_angle = self._sample_rot_angle(index)
            subtomo0 = rotate_area(
                subtomo0,
                rot_angle=rot_angle,
                output_shape=2 * [self.crop_subtomos_to_size],
            )
            subtomo1 = rotate_area(
                subtomo1,
                rot_angle=rot_angle,
                output_shape=2 * [self.crop_subtomos_to_size],
            )

            # save_mrc_data(subtomo0, f"testing/subtomos_rotated/subtomo0/{Path(subtomo0_file).stem}.mrc")
            # save_mrc_data(subtomo1, f"testing/subtomos_rotated/subtomo1/{Path(subtomo1_file).stem}.mrc")


            # add missing wedge
            mw_mask = get_missing_wedge_mask(
                grid_size=2 * [self.crop_subtomos_to_size],
                mw_angle=self.mw_angle,
                device=subtomo0.device,
            )
            rot_mw_mask = get_rotated_missing_wedge_mask(
                grid_size=2 * [self.crop_subtomos_to_size],
                mw_angle=self.mw_angle,
                rot_angle=rot_angle,
                device=subtomo0.device,
            )
        else:
            mw_mask = get_missing_wedge_mask(
                grid_size=subtomo0.shape,
                mw_angle=self.mw_angle,
                device=subtomo0.device,
            )
            rot_mw_mask = mw_mask
            rot_angle = 0

        model_input = apply_fourier_mask_to_tomo(subtomo0, mw_mask)
        item = {
            "model_input": model_input,
            "model_target": subtomo1,
            "mw_mask": mw_mask,
            "rot_mw_mask": rot_mw_mask,
            "subtomo0_file": subtomo0_file,
            "subtomo1_file": subtomo1_file,
            "rot_angle": rot_angle,
        }
        return item


# %%
