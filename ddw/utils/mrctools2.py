import os
import shutil
import torch
import mrcfile
import numpy as np
from PIL import Image

def load_data(image_file):
    """
    Load an image from a file (MRC or an other supported by PIL).
    """
    try:
        with mrcfile.open(image_file, permissive=True) as mrc:
            try:
                data = torch.tensor(mrc.data)
            except TypeError:
                data = torch.tensor(mrc.data.astype(float))
        return [data[z, :, :] for z in range(data.shape[0])]

    except: 
        try:
            # Essayer de charger avec PIL (pour les formats standards)
            img = Image.open(image_file)
            data = np.array(img)
            data = torch.tensor(data)
            return data
        except Exception as e:
            raise ValueError(f"Error: {e}")



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
