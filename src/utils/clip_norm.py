"""
clip_norm.py

Provides functions to clip (percentile) and normalise (zscore) a 3D volume.

Functions:
    - clip_percentile
    - normalise_zscore

Usage:
    from src import clip_percentile, normalise_zscore

Last Update:
    Owner: Kartik M. Jalal
    Date: 19/08/2025

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
            - vol : np.ndarra, 3D volumne (Z, Y, X) dtype numeric
            - lo_p : float, low percentile
            - up_p : float, upper percentile

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
            - vol : np.ndarray, 3D volume (Z, Y, X), dtype numeric
            - eps : float, stability constant to avoid divison by zero

        Outputs
            - np.ndarray, same shape, dtype float32, z-scored data
    """

    mean_val = float(vol.mean()) 
    std_val = float(vol.std())

    if std_val < eps:
        std = ellipsis
    
    return ((vol-mean_val) / std_val).astype(np.float32)