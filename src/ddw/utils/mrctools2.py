import os
import shutil
from pathlib import Path
import torch
import mrcfile
import numpy as np
from PIL import Image

def collect_data(image_file, output_dir):
    """
    Collects data from an file (MRC or other supported by PIL) and saves it as .pt files.
    """
    try:
        with mrcfile.open(image_file, permissive=True) as mrc:
            try:
                data = torch.tensor(mrc.data)
            except TypeError:
                data = torch.tensor(mrc.data.astype(float))
            for y in range(data.shape[1]):
                torch.save(data[:, y, :].clone(), f"{output_dir}/{y}.pt")

    except: 
        try:
            # Essayer de charger avec PIL (pour les formats standards)
            img = Image.open(image_file).convert('L')
            data = np.array(img)
            data = torch.flip(torch.tensor(data), dims=[0])
            torch.save(data.clone(), f"{output_dir}/0.pt")
            
        except Exception as e:
            raise ValueError(f"Error: {e}")

def load_data(tensor_file):
    """
    Load an torch.tensor from a tensor file .
    """
    try:
        data = torch.load(tensor_file)
        return data

    except Exception as e: 
        raise ValueError(f"Error: {e}")
    
def load_2d_data(file_path):
    """
    Loads a picture (.png, .tif, .jpeg, etc.) as a torch tensors.
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
