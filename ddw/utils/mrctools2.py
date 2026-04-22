import os
import shutil
from pathlib import Path
import torch
import mrcfile
import numpy as np
from PIL import Image

def collect_data(image_file, output_dir, verbose = True):
    """
    Collects data from an file (MRC or other supported by PIL) and saves it as .pt files.
    """
    try:
        with mrcfile.open(image_file, permissive=True) as mrc:
            try:
                data = torch.tensor(mrc.data)
            except TypeError:
                data = torch.tensor(mrc.data.astype(float))
            file_name = Path(image_file).stem
            os.makedirs(f"{output_dir}/{file_name}", exist_ok=False)
            if verbose:
                print(f"Collecting data from {file_name}. 3D tomogram shape: {data.shape}")
            for y in range(data.shape[1]):
                torch.save(data[:, y, :].clone(), f"{output_dir}/{file_name}/{y}.pt")

    except: 
        try:
            # Essayer de charger avec PIL (pour les formats standards)
            img = Image.open(image_file)
            data = np.array(img)
            data = torch.tensor(data)

            file_name =  Path(image_file).stem
            os.makedirs(f"{output_dir}/{file_name}", exist_ok=False)
            if verbose:
                print(f"Collecting data from {file_name}. 2D image shape: {data.shape}")
            torch.save(data.clone(), f"{output_dir}/{file_name}.pt")
            
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

