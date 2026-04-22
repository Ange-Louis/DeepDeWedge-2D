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

from ddw.utils.load_function_args_from_yaml_config import (
    load_function_args_from_yaml_config,
)
from ddw.utils.mrctools2 import load_data, collect_data
from ddw.utils.subtomos2 import extract_subtomos

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
    # create output directories
    data_dir, fitting_subtomo_dir, val_subtomo_dir = setup_tomo_dir(
        data_dir= data_dir,
        subtomo_dir=subtomo_dir,
        project_dir=project_dir,
        overwrite=overwrite,
        verbose=verbose,
    )
    # check if mask_files are provided properly
    if len(mask_files) == 0:
        mask_files = [None] * len(tomo0_files)
        min_nonzero_mask_fraction_in_subtomo = 0.0
    else:
        if min_nonzero_mask_fraction_in_subtomo is None:
            raise ValueError(
                "min_nonzero_mask_fraction_in_subtomo must be provided if mask_files are provided"
            )
        
    fitting_counter, val_counter = 0, 0
    
    for k, (tomo0_file, tomo1_file, mask_file) in enumerate(zip(tomo0_files, tomo1_files, mask_files)):
        tomo0_name = Path(tomo0_file).stem ; tomo0_dir = f"{data_dir}/tomo0/{tomo0_name}" ; os.makedirs(tomo0_dir, exist_ok= False)
        tomo1_name = Path(tomo1_file).stem ; tomo1_dir = f"{data_dir}/tomo1/{tomo1_name}" ; os.makedirs(tomo1_dir, exist_ok= False)
        if mask_file is not None:
            mask_name = Path(mask_file).stem ; mask_dir = f"{data_dir}/masks/{mask_name}" ; os.makedirs(mask_dir, exist_ok= False)

        # Collect data from tomogram files
        if verbose:
            print(f"Processing tomogram pairs: '{tomo0_name}' & '{tomo1_name}'")

        collect_data(
            image_file= tomo0_file,
            output_dir= tomo0_dir
        )

        collect_data(
            image_file= tomo1_file,
            output_dir= tomo1_dir
        )

        if mask_file is not None:
            collect_data(
                image_file= mask_file,
                output_dir= mask_dir
        )
    
        # actual subtomogram extraction
        os.makedirs()
        tomo0_tensorfiles = sorted(Path(f"{data_dir}/tomo0").rglob("*.pt"))
        tomo1_tensorfiles = sorted(Path(f"{data_dir}/tomo1").rglob("*.pt"))
    if verbose:
        print(f"Starting subtomogram extraction from {len(tomo0_files)} tomogram(s).")
    fitting_counter, val_counter = 0, 0
    # sans masque car je ne sais pas le faire
    for k, (tomo0_tensorfile, tomo1_tensorfile) in enumerate(zip(tomo0_tensorfiles, tomo1_tensorfiles)):
        tomo0 = load_data(tomo0_tensorfile).float()
        tomo1 = load_data(tomo1_tensorfile).float()
        if standardize_full_tomos:
            print(
                f"Standardizing tomogram '{Path(tomo0_tensorfile).stem}' & '{Path(tomo1_tensorfile).stem}' before extracting sub-tomograms."
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
    #     if mask_file is not None:
    #         mask = load_data(mask_file).float()
    #     else:
    #         mask = torch.ones_like(tomo0)
    #     if not (mask == 0).logical_or(mask == 1).all():
    #         raise ValueError("Mask entries must be either 0 or 1")
    #     subtomos_mask, _ = extract_subtomos(
    #         tomo=mask,
    #         subtomo_size=subtomo_size,
    #         subtomo_extraction_strides=subtomo_extraction_strides,
    #         enlarge_subtomos_for_rotating=extract_larger_subtomos_for_rotating,
    #         pad_before_subtomo_extraction=pad_before_subtomo_extraction,
    #     )
    #     selected_subtomo_ids = [
    #         k
    #         for k, submask in enumerate(subtomos_mask)
    #         if (submask.sum() / submask.numel()) >= min_nonzero_mask_fraction_in_subtomo
    #     ]
    #     if mask_file is not None and verbose:
    #         print(
    #             f"Masking selected {len(selected_subtomo_ids)}/{len(subtomos0)} subtomos extracted from tomogram {k}"
    #         )
    #     subtomos0 = [
    #         subtomo for k, subtomo in enumerate(subtomos0) if k in selected_subtomo_ids
    #     ]
    #     subtomos1 = [
    #         subtomo for k, subtomo in enumerate(subtomos1) if k in selected_subtomo_ids
    #     ]
    #     start_coords = [
    #         coords for k, coords in enumerate(start_coords) if k in selected_subtomo_ids
    #     ]

    #     # val_ids = try_to_sample_non_overlapping_subtomo_ids(
    #     #     subtomo_start_coords=start_coords,
    #     #     subtomo_size=subtomo_size,
    #     #     target_sample_size=math.floor(len(subtomos0) * val_fraction),
    #     #     max_tries=3,
    #     #     verbose=False,
    #     # )
    #     # if len(val_ids) < math.floor(len(subtomos0) * val_fraction):
    #     #     print(
    #     #         f"WARNING: Could not sample {round(val_fraction*100, 2)}% of all subtomos for validation due to overlap with the fitting data. "
    #     #         f"Continuing with {round((len(val_ids)/len(subtomos0))*100, 2)}% (i.e. {len(val_ids)}) validation subtomos."
    #     #     )
        num_val_subtomos = math.ceil(len(subtomos0) * val_fraction)
        val_ids = (
            random.Random(seed).sample(range(len(subtomos0)), num_val_subtomos)
            if num_val_subtomos > 0
            else []
        )
        fitting_ids = [k for k in range(len(subtomos0)) if k not in val_ids]

        for idx in sorted(fitting_ids):
            torch.save(
                subtomos0[idx].clone(), f"{fitting_subtomo_dir}/subtomo0//{fitting_counter}.pt"
            )
            torch.save(
                subtomos1[idx].clone(), f"{fitting_subtomo_dir}/subtomo1//{fitting_counter}.pt"
            )
            fitting_counter += 1

        for idx in sorted(val_ids):
            torch.save(
                subtomos0[idx].clone(), f"{val_subtomo_dir}/subtomo0/{val_counter}.pt"
            )
            torch.save(
                subtomos1[idx].clone(), f"{val_subtomo_dir}/subtomo1/{val_counter}.pt"
            )
            val_counter += 1

    if verbose:
        print(f"Done with sub-tomogram extraction.")
        print(
            f"Saved a total of {fitting_counter} sub-tomograms for model fitting to '{fitting_subtomo_dir}'."
        )
        print(
            f"Saved a total of {val_counter} sub-tomograms for validation to '{val_subtomo_dir}'."
        )


def setup_tomo_dir(data_dir, subtomo_dir, project_dir, overwrite, verbose):
    """
    Sets up and manages directories for storing tomogram and subtomogram data
    """
    if data_dir is None:
        if project_dir is not None:
            data_dir = f"{project_dir}/tomos"
        else:
            raise ValueError(
                "tomo_dir must be provided if project_dir is not provided"
            )
    if verbose:
        print(f"Saving all tomogram tensors to '{data_dir}'.")
    if os.path.exists(data_dir):
        if overwrite == True:
            if verbose:
                print(f"Removing existing tomogram directory '{data_dir}'.")
            shutil.rmtree(data_dir)
        else:
            raise ValueError(
                f"subtomo_dir '{data_dir}' already exists. Set 'overwrite' to 'True' to remove it."
            )
    if subtomo_dir is None:
        if project_dir is not None:
            subtomo_dir = f"{project_dir}/subtomos"
        else:
            raise ValueError(
                "subtomo_dir must be provided if project_dir is not provided"
            )
    if verbose:
        print(f"Saving all subtomograms to '{subtomo_dir}'.")
    if os.path.exists(subtomo_dir):
        if overwrite == True:
            if verbose:
                print(f"Removing existing subtomogram directory '{subtomo_dir}'.")
            shutil.rmtree(subtomo_dir)
        else:
            raise ValueError(
                f"subtomo_dir '{subtomo_dir}' already exists. Set 'overwrite' to 'True' to remove it."
            )

    fitting_subtomo_dir = f"{subtomo_dir}/fitting_subtomos"
    val_subtomo_dir = f"{subtomo_dir}/val_subtomos"

    os.makedirs(f"{data_dir}/tomo0/", exist_ok=False)
    os.makedirs(f"{data_dir}/tomo1/", exist_ok=False)
    os.makedirs(f"{fitting_subtomo_dir}/subtomo0/", exist_ok=False)
    os.makedirs(f"{fitting_subtomo_dir}/subtomo1/", exist_ok=False)
    os.makedirs(f"{val_subtomo_dir}/subtomo0/", exist_ok=False)
    os.makedirs(f"{val_subtomo_dir}/subtomo1/", exist_ok=False)
    return data_dir, fitting_subtomo_dir, val_subtomo_dir

# Exemple d'utilisation :
data = prepare_data(tomo0_files=["/home/nathan/Desktop/Ange-Louis/Dataset/DDW_tutorial/tomo_all_frames.rec"],
                    tomo1_files= ["/home/nathan/Desktop/Ange-Louis/Dataset/DDW_tutorial/tomo_all_frames.rec"],
                    subtomo_size=100,
                    project_dir= "testing",
                    overwrite= True)


