"""
preprocess.py

Provides functions for preprocessing the data.

Functions:
    - clip_percentile
    - normalise_zscore
    - random_flip_xy
    - random_intensity_jitter

Usage:
    from src import <<function_name>>, ... || *

Last Update:
    Owner: Kartik M. Jalal
    Date: 20/09/2025

"""
import numpy as np


def clip_percentile(vol: np.ndarray, lo_p: float = 0.5, up_p: float = 99.5) -> np.ndarray:
    """
        Clip voxel intensities to an image-specific percentile range.

        Description:
            Microscpy volumes often have extreme outliers (hot pixels, glare). Clipping
            to [pmin, pmax] percentiles suppresses those extremes while preserving most
            of the data distribution. This is applies per patch or per volume.

        Inputs:
            - vol (np.ndarray): 3D volumne (Z, Y, X) dtype numeric.
            - lo_p (float): low percentile.
            - up_p (float): upper percentile.

        Outputs:
            - np.ndarray, same shape as vol, dtype float32, values clipped to [lo, hi]
    """
    lo = np.percentile(vol, lo_p)
    hi = np.percentile(vol, up_p)

    # if constant value vol
    if hi <= lo:
        return vol.astype(np.float32)
    
    return np.clip(vol, lo, hi).astype(np.float32)


def normalise_zscore(vol: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
        Normalise volume to zero mean and unit variance

        Description:
            Z-score normalisation makes intensities invariant to global gain/offset,
            which is helpful when acquisition conditions vary across samples.

        Inputs:
            - vol (np.ndarray): 3D volume (Z, Y, X), dtype numeric
            - eps (float): stability constant to avoid divison by zero

        Outputs
            - np.ndarray, same shape, dtype float32, z-scored data
    """

    mean_val = float(vol.mean()) 
    std_val = float(vol.std())

    if std_val < eps:
        std_val = ellipsis
    
    return ((vol-mean_val) / std_val).astype(np.float32)


def random_flip_xy(
    vol: np.ndarray,
    heatmap: np.ndarray,
    offsets: np.ndarray,
    offset_mask: np.ndarray,
    prob: float = 0.5
) -> dict:
    """
        Randomly flip a 3D volume along X and Y axes.

        Description:
            XY flips are safe for most brain microscopy since orientation is not fixed.
            Z flips/rotations are avoided due to anisotropy (Z spacing >> XY spacing).

        Inputs:
            - vol (np.ndarray): 3D volume (Z, Y, X).
            - prob (float): probability of flipping per axis.

        Outputs:
            - dict(
                - np.ndarray, same shape as vol, possibly flipped (copy to ensure contiguous memory)
            )
            
    """
    do_x_flip =  np.random.rand() < prob
    do_y_flip = np.random.rand() < prob

    if do_x_flip:
        vol = vol[:, :, ::-1].copy() # flip X
        heatmap = heatmap[:, :, ::-1].copy()
        offsets = offsets[:, :, :, ::-1].copy()
        offset_mask = offset_mask[:, :, ::-1].copy()

        # for offsets, flip Δx channel direction sign
        offsets[2] *= -1
    
    if do_y_flip:
        vol = vol[:, ::-1, :].copy() # flip X
        heatmap = heatmap[:, ::-1, :].copy()
        offsets = offsets[:, :, ::-1, :].copy()
        offset_mask = offset_mask[:, ::-1, :].copy()

        # for offsets, flip Δx channel direction sign
        offsets[1] *= -1

    return dict(
        vol=vol,
        heatmap=heatmap,
        offsets=offsets,
        offset_mask=offset_mask
    )


def random_intensity_jitter(
        vol: np.ndarray,
        gamma_range: tuple = (0.9, 1.1),
        mult_range: tuple = (0.9, 1.1),
        add_range: tuple = (-0.05, 0.05)
) -> np.ndarray:
    """
        Apply light intensity jitter for robustness.

        Description:
            Simulates mild variation in brightness/contrast across acquisitions.
            Keep ranges tight to avoid unrealistic changes.

        Inputs:
            - vol (np.ndarray): 3D volume (Z, Y, X), usually after normalisation.
            - gamme_range (tuple(float, float)): gamma exponent range.
            - mult_range (tuple(float, flaot)): multiplicative factor range.
            - add_range (tuple(float, float)): additive bias range.

        Outputs:
            - np.ndarray, same shape as vol, dtype float32, intensity-jittered
    """
    gamma = np.random.uniform(*gamma_range) # gamma correction (non-linear)
    multi = np.random.uniform(*mult_range) # multiplicative scale
    add = np.random.uniform(*add_range) # additive shift
    
    # keep non-negative before gamma
    vol = np.clip(vol, 0, None)
    # apply gamma
    vol = vol ** gamma
    # scale and shift the value by a small margin
    vol = multi * vol + add
    return vol.astype(np.float32)


