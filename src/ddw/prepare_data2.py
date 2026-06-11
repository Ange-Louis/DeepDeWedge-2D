# %%
import math
import os
import random
import shutil
from pathlib import Path
from typing import List, Optional

import torch
import typer
from typer_config import conf_callback_factory
from typing_extensions import Annotated

from src.ddw.utils.load_function_args_from_yaml_config import (
    load_function_args_from_yaml_config,
)
from src.ddw.utils.mrctools2 import load_data, collect_data
from src.ddw.utils.subtomos2 import extract_subtomos

loader = lambda yaml_config_file: load_function_args_from_yaml_config(
    function=prepare_data, yaml_config_file=yaml_config_file
)
callback = conf_callback_factory(loader)


def prepare_data(
    tomo0_files: Annotated[
        List[Path],
        typer.Option(
            help="List of paths to tomograms (mrc files) reconstructed from one half of the tilt series or movie frames."
        ),
    ],
    tomo1_files: Annotated[
        List[Path],
        typer.Option(
            help="List of paths to tomograms (mrc files) reconstructed from the other half of the tilt series or movie frames."
        ),
    ],
    subtomo_size: Annotated[
        int,
        typer.Option(
            help="Size of the square subtomograms to extract for model fitting. This value must be divisible by 2^{num_downsample_layers}, where {num_downsample_layers} is the number of downsampling layers used in the U-Net."
        ),
    ],
    val_fraction: Annotated[
        float,
        typer.Option(
            help="Fraction of subtomograms to use for validation. Increasing this fraction will decrease the number of subtomograms used for model fitting."
        ),
    ] = 0.1,
    mask_files: Annotated[
        List[Path],
        typer.Option(
            help="The masks aren't working at the moment\
                \nList of paths to binary masks (mrc files) that outline the region of interest in the tomograms to guide subtomogram extraction. The DeepDeWedge reconstruction of areas outside the mask may be less accurate. If no mask_files are provided, the entire tomogram is used for subtomogram extraction."
        ),
    ] = [],
    min_nonzero_mask_fraction_in_subtomo: Annotated[
        Optional[float],
        typer.Option(
            help="The masks aren't working at the moment\
                \nMinimum fraction of pixels in a subtomogram that correspond to nonzero pixels in the mask. If mask_files are provided, this parameter has to be provided as well. If no mask_files are provided, this parameter is ignored."
        ),
    ] = 0.3,
    subtomo_extraction_strides: Annotated[
        Optional[List[int]],
        typer.Option(
            help="List of 2 integers specifying the 2D Strides used for subtomogram extraction. If set to None, stride 'subtomo_size' is used in all 2 directions. Smaller strides result in more sub-tomograms being extracted."
        ),
    ] = None,
    pad_before_subtomo_extraction: Annotated[
        bool,
        typer.Option(
            help="Whether to pad the tomograms before extracting subtomograms."
        ),
    ] = False,
    extract_larger_subtomos_for_rotating: Annotated[
        bool,
        typer.Option(
            help="If True, larger subtomograms with a size of 'subtomo_size*sqrt(2)' will be extracted in order to avoid boundary effects when rotating the subtomograms."
        ),
    ] = True,
    standardize_full_tomos: Annotated[
        bool,
        typer.Option(
            help="If 'True', the tomo0 and tomo1 tomograms will be standardized (mean=0, std=1) before extracting the subtomograms. This is useful for tomograms with low pixel itensities, and DDW can fail when processing such tomograms wihtout standardization."
        ),
    ] = False,
    subtomo_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Where to save the subtomograms. If not provided, the subtomograms will be saved to '{project_dir}/subtomos'."
        ),
    ] = None,
    data_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Where to save the initial tomograms. If not provided, the tomograms will be saved to '{project_dir}/tomos'."
        ),
    ] = None,
    project_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="If 'subtomo_dir' is not provided, the subtomogram directory will saved to '{project_dir}/subtomos'."
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(
            help="Whether to overwrite the existing subtomo_dir if it already exists. If False, the function will raise an error if the directory already exists."
        ),
    ] = False,
    seed: Annotated[
        Optional[int],
        typer.Option(help="Controls the randomness of the validation data selection."),
    ] = None,
    verbose: Annotated[bool, typer.Option()] = True,
    config: Annotated[
        Optional[Path],
        typer.Option(
            callback=callback,
            is_eager=True,
            help="Path to a yaml file containing the argumens for this function. Comand line arguments will overwrite the ones in the yaml file.",
        ),
    ] = None,
):
    """
    Extract square sub-tomograms that are used to generate inputs and targets for model fitting. Typically the first command to run.
    """
    # check if mask_files are provided properly
    if len(mask_files) == 0:
        mask_files = [None] * len(tomo0_files)
        min_nonzero_mask_fraction_in_subtomo = 0.0
    else:
        if min_nonzero_mask_fraction_in_subtomo is None:
            raise ValueError(
                "min_nonzero_mask_fraction_in_subtomo must be provided if mask_files are provided"
            )
        
    if verbose:
        print(f"Starting subtomogram extraction from {len(tomo0_files)} tomogram(s).")

    if data_dir is None:
        if project_dir is not None:
            data_dir = f"{project_dir}/tomos"
        else:
            raise ValueError("data_dir must be provided if project_dir is not provided")

    if subtomo_dir is None:
        if project_dir is not None:
            subtomo_dir = f"{project_dir}/subtomos"
        else:
            raise ValueError("subtomo_dir must be provided if project_dir is not provided")

    for d in [data_dir, subtomo_dir]:
        if os.path.exists(d):
            if overwrite:
                if verbose:
                    print(f"Removing existing directory '{d}'.")
                shutil.rmtree(d)
            else:
                raise ValueError(
                    f"Directory '{d}' already exists. Set 'overwrite=True' to remove it."
                )
        os.makedirs(d, exist_ok=True)

    total_fitting_counter, total_val_counter = 0, 0

    for num_tomos, (tomo0_file, tomo1_file, mask_file) in enumerate(zip(tomo0_files, tomo1_files, mask_files)):
        fitting_counter, val_counter = 0, 0
 
        tomo0_name = Path(tomo0_file).stem
        tomo1_name = Path(tomo1_file).stem
        
        # create output directories specifically for these tomograms
        tomo0_dir, fitting_subtomo0_dir, val_subtomo0_dir,\
        tomo1_dir, fitting_subtomo1_dir, val_subtomo1_dir = setup_tomo_dir(
            data_dir=data_dir,
            subtomo_dir=subtomo_dir,
            tomo0_name=tomo0_name,
            tomo1_name=tomo1_name
        )

        if mask_file is not None:
            mask_name = Path(mask_file).stem
            mask_dir = Path(data_dir) / "masks" / mask_name
            os.makedirs(mask_dir, exist_ok=True)

        # Collect data from tomogram files
        if verbose:
            print(f"Processing tomogram pairs: '{tomo0_name}' & '{tomo1_name}'")

        collect_data(image_file=tomo0_file, output_dir=tomo0_dir)
        collect_data(image_file=tomo1_file, output_dir=tomo1_dir)

        if mask_file is not None:
            collect_data(image_file=mask_file, output_dir=mask_dir)
    
        # actual subtomogram extraction
        tomo0_tensorfiles = sorted(Path(tomo0_dir).glob("*.pt"), key=lambda x: int(x.stem))
        tomo1_tensorfiles = sorted(Path(tomo1_dir).glob("*.pt"), key=lambda x: int(x.stem))

        for (tomo0_tensorfile, tomo1_tensorfile) in zip(tomo0_tensorfiles, tomo1_tensorfiles):
            tomo0 = load_data(tomo0_tensorfile).float()
            tomo1 = load_data(tomo1_tensorfile).float()
            
            if standardize_full_tomos:
                print(
                    f"Standardizing tomogram '{Path(tomo0_tensorfile).stem}' & '{Path(tomo1_tensorfile).stem}' before extracting sub-tomograms."
                )
                tomo0 -= tomo0.mean()
                tomo1 -= tomo1.mean()
                tomo0 /= tomo0.std()
                tomo1 /= tomo1.std()
            else:
                std = tomo0.std()
                if std < 1e-3:
                    print(f"\
                        WARNING: Standard deviation of '{Path(tomo0_tensorfile).stem}' is low ({std}), which may lead to issues during model fitting!\
                        \nConsider setting 'standardize_full_tomos=True'.\
                        \nIf you do so, you must also set 'standardize_full_tomos=True' for 'ddw refine-tomogram'.\
                ")
          
  
            subtomos0, start_coords = extract_subtomos(
                tomo=tomo0,
                subtomo_size=subtomo_size,
                subtomo_extraction_strides=subtomo_extraction_strides,
                enlarge_subtomos_for_rotating=extract_larger_subtomos_for_rotating,
                pad_before_subtomo_extraction=pad_before_subtomo_extraction,
            )
            subtomos1, _ = extract_subtomos(
                tomo=tomo1,
                subtomo_size=subtomo_size,
                subtomo_extraction_strides=subtomo_extraction_strides,
                enlarge_subtomos_for_rotating=extract_larger_subtomos_for_rotating,
                pad_before_subtomo_extraction=pad_before_subtomo_extraction,
            )

            for idx in range(len(subtomos0)):
                torch.save(
                    subtomos0[idx].clone(), f"{fitting_subtomo0_dir}/{fitting_counter}.pt"
                )
                torch.save(
                    subtomos1[idx].clone(), f"{fitting_subtomo1_dir}/{fitting_counter}.pt"
                )
                fitting_counter += 1


        fitting_subtomo0_tensorfiles = sorted(Path(fitting_subtomo0_dir).glob("*.pt"), key=lambda x: int(x.stem))

        num_val_subtomos = math.ceil(len(fitting_subtomo0_tensorfiles) * val_fraction)
        val_ids = (
            random.Random(seed).sample(range(len(fitting_subtomo0_tensorfiles)), num_val_subtomos)
            if num_val_subtomos > 0
            else []
        )
        for idx in sorted(val_ids):
            shutil.move(f"{fitting_subtomo0_dir}/{idx}.pt", f"{val_subtomo0_dir}/{idx}.pt")
            shutil.move(f"{fitting_subtomo1_dir}/{idx}.pt", f"{val_subtomo1_dir}/{idx}.pt")

            val_counter += 1

        fitting_counter -= val_counter

        total_fitting_counter += fitting_counter
        total_val_counter += val_counter

        if verbose:
            print(f"Done with {num_tomos+1}th sub-tomogram extraction.")
            print(
                f"Saved {fitting_counter} sub-tomograms for model fitting from {num_tomos} tomogram."
            )
            print(
                f"Saved {val_counter} sub-tomograms for validation from {num_tomos} tomogram."
            )

    if verbose:
        print(f"Done with all sub-tomogram extraction.")
        print(
            f"Saved a total of {total_fitting_counter} sub-tomograms for model fitting."
        )
        print(
            f"Saved a total of {total_val_counter} sub-tomograms for validation."
        )

def setup_tomo_dir(data_dir, subtomo_dir, tomo0_name, tomo1_name):
    """
    Sets up sub-directories specific to the current tomogram pair.
    The root logic (overwrite/creation) is now handled outside of the loop.
    """
    tomo0_dir = f"{data_dir}/tomo0/{tomo0_name}"
    tomo1_dir = f"{data_dir}/tomo1/{tomo1_name}"
    
    fitting_subtomo0_dir = f"{subtomo_dir}/fitting_subtomos/subtomo0/{tomo0_name}"
    fitting_subtomo1_dir = f"{subtomo_dir}/fitting_subtomos/subtomo1/{tomo1_name}"
    
    val_subtomo0_dir = f"{subtomo_dir}/val_subtomos/subtomo0/{tomo0_name}"
    val_subtomo1_dir = f"{subtomo_dir}/val_subtomos/subtomo1/{tomo1_name}"

    os.makedirs(tomo0_dir, exist_ok=True)
    os.makedirs(tomo1_dir, exist_ok=True)
    os.makedirs(fitting_subtomo0_dir, exist_ok=True)
    os.makedirs(fitting_subtomo1_dir, exist_ok=True)
    os.makedirs(val_subtomo0_dir, exist_ok=True)
    os.makedirs(val_subtomo1_dir, exist_ok=True)
    
    return (tomo0_dir, fitting_subtomo0_dir, val_subtomo0_dir,
            tomo1_dir, fitting_subtomo1_dir, val_subtomo1_dir)

if __name__ == "__main__":
    prepare_data(
    tomo0_files=["/home/nathan/Desktop/Ange-Louis/Dataset/DDW_tutorial/tomo_even_frames.rec"],
    tomo1_files=["/home/nathan/Desktop/Ange-Louis/Dataset/DDW_tutorial/tomo_odd_frames.rec"],
    subtomo_size=128,
    project_dir="testing",
    overwrite=True,
    subtomo_extraction_strides=[80,80],
    extract_larger_subtomos_for_rotating=False,
    )
