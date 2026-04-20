import os
import shutil
import torch
import mrcfile
import numpy as np
from PIL import Image

def collect_data(image_file, tomo_dir):
    try:
        with mrcfile.open(image_file, permissive=True) as mrc:
            try:
                data = torch.tensor(mrc.data)
            except TypeError:
                data = torch.tensor(mrc.data.astype(float))
            print(data.shape())

            #for y in range(data.shape[1]):
                #torch.save(data[:, y, :], f"{tomo_dir}/{os.path.basename(image_file)}_({y}).pt")

    except: 
        try:
            # Essayer de charger avec PIL (pour les formats standards)
            img = Image.open(image_file)
            data = np.array(img)
            data = torch.tensor(data)

            #torch.save(data, f"{tomo_dir}/{os.path.basename(image_file)}.pt")
            
        except Exception as e:
            raise ValueError(f"Error: {e}")
        

def setup_tomo_dir(project_dir, tomo_dir = None, overwrite = True, verbose =True):
    """
    Sets up and manages directories for storing tomogram and subtomogram data
    """
    if tomo_dir is None:
        if project_dir is not None:
            tomo_dir = f"{project_dir}/tomos"
        else:
            raise ValueError(
                "tomo_dir must be provided if project_dir is not provided"
            )
    if verbose:
        print(f"Saving all tomogram tensors to '{tomo_dir}'.")
    if os.path.exists(tomo_dir):
        if overwrite == True:
            if verbose:
                print(f"Removing existing tomogram directory '{tomo_dir}'.")
            shutil.rmtree(tomo_dir)
        else:
            raise ValueError(
                f"subtomo_dir '{tomo_dir}' already exists. Set 'overwrite' to 'True' to remove it."
            )

    os.makedirs(f"{tomo_dir}/tomo0/", exist_ok=False)
    os.makedirs(f"{tomo_dir}/tomo1/", exist_ok=False)

    return tomo_dir


tomo0_files = ["/home/nathan/Desktop/Ange-Louis/Dataset/DDW_tutorial/tomo_all_frames.rec"]
tomo_dir = setup_tomo_dir(project_dir= "testing")

for tomo0_file in tomo0_files:
	
    collect_data(tomo0_file, tomo_dir= tomo_dir)

