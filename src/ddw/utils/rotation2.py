import math

import torch
from scipy import ndimage, spatial


def rotate_area(area, rot_angle, output_shape=None, order=3):
    """
    Rotates the 2D tensor 'area' by 'rot_angle' degrees. The rotated tensor, which is typically larger than the original one, is center-cropped such that it has dimensions 'output_shape'. If 'output_shape' is None, the rotated tensor is cropped to the dimensions of 'area'.
    """
    area_shape = torch.tensor(area.shape[-2:])
    if output_shape is None:
        output_shape = area_shape
    # need later for cropping
    crop_offset = [math.floor((vs - cs) / 2) for vs, cs in zip(area_shape, output_shape)]
    if rot_angle != 0:
        if not torch.is_tensor(rot_angle):
            rot_angle = torch.tensor(rot_angle)
        # apply the rotation using rotate
        area = torch.tensor(
            ndimage.rotate(area, angle=rot_angle, reshape= False, mode='reflect', order=1),
            device=area.device,
            dtype=area.dtype,
        )
    area = area[
        crop_offset[0] : crop_offset[0] + output_shape[0],
        crop_offset[1] : crop_offset[1] + output_shape[1],
    ]
    return area
