import tempfile

import torch
import tqdm

from src.ddw.prepare_data2 import prepare_data

from src.ddw.utils.subtomo_dataset2 import SubtomoDataset
from src.ddw.utils.fourier2 import apply_fourier_mask_to_tomo
from src.ddw.utils.rotation2 import rotate_area


def get_avg_model_input_mean_and_std(tomo_file, subtomo_size, subtomo_extraction_strides, standardize, mw_angle, batch_size, num_workers, batches=None, verbose=False):
    """
    Computes the average mean and standard deviation of model-input-type sub-tomograms (with two missing wedges). These values are used to normalize sub-tomograms during model fitting and to normalize full tomograms in the final refinement step. 
    """
    with tempfile.TemporaryDirectory() as subtomo_dir:
        prepare_data(
            tomo0_files=[tomo_file],
            tomo1_files=[tomo_file],
            mask_files=[],
            subtomo_size=subtomo_size,
            extract_larger_subtomos_for_rotating=True,
            subtomo_extraction_strides=subtomo_extraction_strides,  
            val_fraction=0.0,
            subtomo_dir=subtomo_dir,
            standardize_full_tomos=standardize,
            overwrite=True,
            verbose=False,
        )
        dataset = SubtomoDataset(
            subtomo_dir=f"{subtomo_dir}/fitting_subtomos",
            crop_subtomos_to_size=subtomo_size,
            mw_angle=mw_angle,
            rotate_subtomos=True,
            deterministic_rotations=False,
        )
        fitting_dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        mean, std = get_avg_model_input_mean_and_std_from_dataloader(fitting_dataloader, batches=batches, verbose=verbose)
    return mean, std


def get_avg_model_input_mean_and_std_from_dataloader(dataloader, batches=None, verbose=False):
    """
    See above. 
    """
    if batches is None:
        batches = 1 * len(dataloader)
    means, vars = [], []
    bar = (
        tqdm.tqdm(range(batches), desc="Computing model-input normalization statistics")
        if verbose
        else range(batches)
    )
    iter_loader = iter(dataloader)
    for _ in bar:
        try:
            batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(dataloader)
            batch = next(iter_loader)

        if "model_input" in batch:
            model_input = batch["model_input"]
        else:
            subtomo0_rotated = rotate_area(
                batch["subtomo0_original"],
                rot_angle= batch["rot_angle"],
                output_shape= 2*batch["crop_subtomos_to_size"],
            )

            model_input = apply_fourier_mask_to_tomo(subtomo0_rotated, batch["mw_mask"])
        means.append(model_input.mean(dim=(-1, -2)))
        vars.append(model_input.var(dim=(-1, -2)))
    mean = torch.concat(means, 0).mean().cpu().item()
    std = torch.concat(vars, 0).mean().sqrt().cpu().item()
    return mean, std
    