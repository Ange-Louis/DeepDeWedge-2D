import os
import shutil
import numpy as np
from PIL import Image
import mrcfile
import torch


def load_data(file_path):
    """
    Loads a .mrc or .rec file or a picture (.png, .tif, .jpeg, etc.) as a torch tensors.
    """
    file_type = os.path.splitext(file_path)[1].lower()

    if (file_type == '.rec' or file_type == '.mrc'):
        with mrcfile.open(file_path, permissive=True) as mrc:
            try:
                data = torch.tensor(mrc.data)
            except TypeError:
                data = torch.tensor(mrc.data.astype(float))
        return data
    else:
        try:
            img = Image.open(file_path)
            img_array = np.array(img)
            data = torch.tensor(img_array.astype(float))
            return data
        except Exception as e:
            raise ValueError(f"Unsupported file format or corrupted file: {e}")


def save_mrc_data(data, mrc_file, save=False):
    """
    Saves a torch tensor as an .mrc file.
    """
    if save:
        if os.path.exists(mrc_file):
            print(f"File '{mrc_file}' already exists! Moving it to '{mrc_file}~'")
            shutil.move(mrc_file, f"{mrc_file}~")
    with mrcfile.new(mrc_file, overwrite=True) as mrc:
        mrc.set_data(data.numpy())
