# %%
import math
import os
from pathlib import Path
from typing import List, Optional

import torch
import tqdm
import typer
from torch.utils.data import DataLoader, TensorDataset
from typer_config import conf_callback_factory
from typing_extensions import Annotated

from src.ddw.fit_model2 import LitUnet2D
from src.ddw.utils.fourier2 import apply_fourier_mask_to_tomo
from src.ddw.utils.load_function_args_from_yaml_config import \
    load_function_args_from_yaml_config
from src.ddw.utils.missing_wedge2 import get_missing_wedge_mask
from src.ddw.utils.mrctools2 import load_data, save_mrc_data, load_2d_data
from src.ddw.utils.normalization2 import get_avg_model_input_mean_and_std
from src.ddw.utils.subtomos2 import extract_subtomos, reassemble_subtomos

loader = lambda yaml_config_file: load_function_args_from_yaml_config(
    function=refine_tomogram, yaml_config_file=yaml_config_file
)
callback = conf_callback_factory(loader)


def refine_tomogram(
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
    model_checkpoint_file: Annotated[
        Path,
        typer.Option(
            help="Path to a model checkpoint file (.ckpt extension). Checkpoints saved during model fitting can be found in the 'logdir' directory specifed for the 'fit-model' command."
        ),
    ],
    subtomo_size: Annotated[
        int,
        typer.Option(
            help="Size of the cubic subtomograms to extract. This should be the same as the subtomo_size used during model fitting."
        ),
    ],
    mw_angle: Annotated[
        int, typer.Option(help="Width of the missing wedge in degrees.")
    ],
    subtomo_overlap: Annotated[
        Optional[int],
        typer.Option(
            help="Overlap between subtomograms. This determines the stride of the sliding window used to extract subtomograms. If 'None', this is set to '1/3 * subtomo_size'."
        ),
    ] = None,
    standardize_full_tomos: Annotated[
        bool,
        typer.Option(
            help="Set to 'True' if and only if 'standardize_full_tomos' was 'True' for 'ddw fit-model'."
        ),
    ] = False,
    recompute_normalization: Annotated[
        bool,
        typer.Option(
            help="Whether to recompute the mean and variance used to normalize the tomo0s and tomo1s (see Appendix B in the paper). If `False`, the mean and variance of model inputs calculated during model fitting will be used. If `True`, the average model input mean and variance will be computed for each tomogram individually. We recommend setting this to to `True`. If you apply a model to a tomogram that was not used for model fitting or if the means and variances of the tomograms during model fitting are considerably different, recomputing the normalization is expected to be very beneficial for tomogram refinement."
        ),
    ] = True,
    batch_size: Annotated[
        int, typer.Option(help="Batch size for processing subtomograms.")
    ] = 1,
    return_tomos: Annotated[
        bool,
        typer.Option(
            help="Whether to return the refined tomograms as a list of tensors. If 'False', the refined tomograms will only be saved to 'output_dir', and the function returns nothing."
        ),
    ] = False,
    data_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Where to save the initial tomogram slices. If not provided, the tomograms will be saved to '{project_dir}/tomos'."
        ),
    ] = None,
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Where to save the refined tomograms. If not provided, either 'project_dir' has to be provided or 'return_subtomos' must be 'True'."
        ),
    ] = None,
    project_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Path to the project directory. If not provided the refined tomograms will be saved to {project_dir}/refined_tomograms. If 'return_subtomos' is False, and 'output_dir' is not provided, this has to be provided."
        ),
    ] = None,
    num_workers: Annotated[
        int,
        typer.Option(
            help="Number of CPU workers to use during the recomputation of the normalization statistics and dataloading for refining the tomograms."
        ),
    ] = 0,
    gpu: Annotated[
        Optional[List[int]],
        typer.Option(
            help="GPU id on which to run the model. If None, the model will be run on the CPU. Currently, only a single GPU is supported. Providing multiple GPUs will result in a warning and only the first GPU will be used."
        ),
    ] = None,
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
    Use a fitted U-Net to denoise tomograms and to fill in their missing wedge. Typically run after `fit-model`.
    """
    if data_dir is None:
        if project_dir is not None:
            data_dir = f"{project_dir}/tomos"
        else:
            raise ValueError("data_dir must be provided if project_dir is not provided")
    if output_dir is None:
        if project_dir is not None:
            output_dir = f"{project_dir}/refined_tomograms"
        elif project_dir is None and return_tomos is False:
            raise ValueError(
                "If return_tomos is False, output_dir or project_dir must be provided, otherwise the refined tomograms will be lost."
            )
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    if return_tomos is False and output_dir is None:
        raise ValueError(
            "If return_tomos is False, output_dir or project_dir must be provided, otherwise the refined tomograms will be lost."
        )
    if return_tomos:
        tomo_ref = []

    if subtomo_overlap is None:
        subtomo_overlap = int(math.ceil(subtomo_size / 3))

    # if hasattr(gpu, "__len__"):
    #     if len(gpu) > 1:
    #         print(f"WARNING: Currently, only a single GPU is supported in 'ddw refine-tomogram'. You passed gpu={gpu}. Continuing with gpu={gpu[0]}.")
        
    device = "cpu" if gpu is None else f"cuda:{gpu[0]}"
    lightning_model = (
        LitUnet2D.load_from_checkpoint(model_checkpoint_file).to(device).eval()
    )

    with torch.no_grad():
        for k, (t0_file, t1_file) in enumerate(zip(tomo0_files, tomo1_files)):
            t0_name = Path(t0_file).stem
            t1_name = Path(t1_file).stem

            tomo_to_refine = load_2d_data(t0_file)

            print(f"Refining {k}th tomogram")

            if len(tomo_to_refine.shape) == 3:
                refined = refine_3d(
                    tomo_to_refine= tomo_to_refine,
                    t0_file=t0_file,
                    t0_name=t0_name,
                    t1_name= t1_name,
                    data_dir=data_dir,
                    lightning_model= lightning_model,
                    subtomo_size=subtomo_size,
                    subtomo_overlap=subtomo_overlap,
                    mw_angle=mw_angle,
                    num_workers=num_workers,
                    batch_size=batch_size,
                    standardize_full_tomos= standardize_full_tomos,
                    recompute_normalization= recompute_normalization
                )
            else:
                refined = refine_2d(
                    t0_name= t0_name,
                    t1_name= t1_name,
                    t0_file= t0_file,
                    t1_file= t1_file,
                    lightning_model= lightning_model,
                    subtomo_size= subtomo_size,
                    subtomo_overlap= subtomo_overlap,
                    mw_angle= mw_angle,
                    num_workers= num_workers,
                    batch_size= batch_size,
                    standardize_full_tomos= standardize_full_tomos,
                    recompute_normalization= recompute_normalization
                )

            if return_tomos:
                tomo_ref.append(refined)
                
            if output_dir is not None:
                basename = f"{t0_name}+{t1_name}"
                outfile = f"{output_dir}/{basename}_refined.mrc"
                print(f"Saving refined tomogram to {outfile}")
                save_mrc_data(refined.cpu(), f"{outfile}", save=True)
    if return_tomos:
        return tomo_ref


def _refine_single_tomogram(
    tomo,
    lightning_model,
    subtomo_size,
    subtomo_overlap,
    mw_angle,
    normalization_loc,
    normalization_scale,
    num_workers=0,
    batch_size=1,
    pbar_desc="Refining tomogram",
):
    # apply missing wedge mask here to be more consistent with data during model fitting
    mw_mask = get_missing_wedge_mask(tomo.shape, mw_angle, device=tomo.device)
    tomo = apply_fourier_mask_to_tomo(tomo, mw_mask)

    tomo = (tomo / tomo.std()) * torch.tensor(normalization_scale).to(tomo.device)
    tomo = tomo - tomo.mean() + torch.tensor(normalization_loc).to(tomo.device)

    subtomos, subtomo_start_coords = extract_subtomos(
        tomo=tomo.cpu(),
        subtomo_size=subtomo_size,
        subtomo_extraction_strides=2 * [subtomo_size - subtomo_overlap],
        enlarge_subtomos_for_rotating=False,
        pad_before_subtomo_extraction=True,
    )
    
    subtomos = TensorDataset(torch.stack(subtomos))
    subtomo_loader = DataLoader(
        subtomos,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory= True
    )
    model_outputs = []
    with torch.no_grad():
        for batch in tqdm.tqdm(subtomo_loader, desc=pbar_desc):
            batch_subtomos = batch[0].to(lightning_model.device)
            model_output = lightning_model(batch_subtomos)
            model_output = model_output.detach().cpu()
            model_outputs.append(model_output)
    model_outputs = list(torch.concat(model_outputs, 0))

    tomo_ref = reassemble_subtomos(
        subtomos=model_outputs,
        subtomo_start_coords=subtomo_start_coords,
        subtomo_overlap=subtomo_overlap,
        crop_to_size=tomo.shape,
    )
    return tomo_ref

def refine_3d(
        tomo_to_refine,
        t0_file,
		t0_name,
		t1_name,
		data_dir,
		lightning_model,
		subtomo_size,
		subtomo_overlap,
		mw_angle,
		num_workers,
		batch_size,
        standardize_full_tomos,
        recompute_normalization,
):
    if recompute_normalization:
        loc, scale = get_avg_model_input_mean_and_std(
            tomo_file=t0_file,
            subtomo_size=subtomo_size,
            subtomo_extraction_strides=2 * [subtomo_size - subtomo_overlap],
            mw_angle=mw_angle,
            batch_size=batch_size,
            standardize=standardize_full_tomos,
            num_workers=num_workers,
            verbose=True,
        )
    else:
        loc, scale = (
            lightning_model.unet.normalization_loc.clone().detach().item(),
            lightning_model.unet.normalization_scale.clone().detach().item(),
        )

    t0_dir = f"{data_dir}/tomo0/{t0_name}"
    t1_dir = f"{data_dir}/tomo1/{t1_name}"
    t0_tensorfiles = sorted(Path(t0_dir).glob("*.pt"), key=lambda x: int(x.stem))
    t1_tensorfiles = sorted(Path(t1_dir).glob("*.pt"), key=lambda x: int(x.stem))

    refined = torch.empty(tomo_to_refine.shape[0], 0, tomo_to_refine.shape[2])


    for k, (t0_tensorfile, t1_tomotensorfile) in enumerate(zip(t0_tensorfiles, t1_tensorfiles)):
        t0 = load_data(t0_tensorfile).float()
        t1 = load_data(t1_tomotensorfile).float()

        print(t0_tensorfile.stem)

        t_ref = _refine_single_tomogram(
            tomo=t0,
            lightning_model=lightning_model,
            subtomo_size=subtomo_size,
            subtomo_overlap=subtomo_overlap,
            mw_angle=mw_angle,
            normalization_loc=loc,
            normalization_scale=scale,
            num_workers=num_workers,
            batch_size=batch_size,
            pbar_desc=f"EVN: {t0_name} slice {k}",
        )
        t_ref += _refine_single_tomogram(
            tomo=t1,
            lightning_model=lightning_model,
            subtomo_size=subtomo_size,
            subtomo_overlap=subtomo_overlap,
            mw_angle=mw_angle,
            normalization_loc=loc,
            normalization_scale=scale,
            num_workers=num_workers,
            batch_size=batch_size,
            pbar_desc=f"ODD: {t1_name} slice {k}",
        )
        t_ref /= 2

        t_ref = t_ref.unsqueeze(1)
        refined = torch.cat([refined, t_ref], dim=1)
    return refined


def refine_2d(
        t0_name,
        t1_name,
        t0_file,
        t1_file,
		lightning_model,
		subtomo_size,
		subtomo_overlap,
		mw_angle,
		num_workers,
		batch_size,
        standardize_full_tomos,
        recompute_normalization
):
    if recompute_normalization:
        loc, scale = get_avg_model_input_mean_and_std(
            tomo_file=t0_file,
            subtomo_size= subtomo_size,
            subtomo_extraction_strides= 2 * [subtomo_size-subtomo_overlap],
            mw_angle= mw_angle,
            batch_size= batch_size,
            standardize= standardize_full_tomos,
            num_workers= num_workers,
            verbose=True
        )

    else:
        loc, scale = (
            lightning_model.unet.normalization_loc.clone().detach().item(),
            lightning_model.unet.normalization_scale.clone().detach().item()
        )

    t0 = load_2d_data(t0_file).float()
    t1 = load_2d_data(t1_file).float()
    t_ref = _refine_single_tomogram(
        tomo=t0,
        lightning_model= lightning_model,
        subtomo_size= subtomo_size,
        subtomo_overlap= subtomo_overlap,
        mw_angle= mw_angle,
        normalization_loc= loc,
        normalization_scale= scale,
        num_workers= num_workers,
        batch_size= batch_size,
        pbar_desc=f"Refining {t0_name}"
    )
    t_ref += _refine_single_tomogram(
        tomo=t1,
        lightning_model= lightning_model,
        subtomo_size= subtomo_size,
        subtomo_overlap= subtomo_overlap,
        mw_angle= mw_angle,
        normalization_loc= loc,
        normalization_scale= scale,
        num_workers= num_workers,
        batch_size= batch_size,
        pbar_desc=f"Refining {t1_name}"
    )
    t_ref /=2
    return t_ref


# Exemple d'utilisation
if __name__ == "__main__":
    refine_tomogram(
        tomo0_files=["/home/nathan/Desktop/Ange-Louis/Dataset/DDW_tutorial/tomo_even_frames.rec"],
        tomo1_files= ["/home/nathan/Desktop/Ange-Louis/Dataset/DDW_tutorial/tomo_odd_frames.rec"],
        model_checkpoint_file= "testing2/logs/version_0/checkpoints/val_loss/epoch=362-val_loss=1.94520.ckpt",
        subtomo_size=128,
        mw_angle=50,
        project_dir= "testing",
        num_workers=0,
        recompute_normalization=False,
        batch_size= 10,
        gpu= [0,1]
    )
