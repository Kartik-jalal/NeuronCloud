"""
data.py

Provides functions to read, preprocess and augument the data.

Functions:
    - get_tiff_img
    - get_data_info
    - train_val_test_split
    - crop3d
    - choose_positive_start
    - choose_background_start

Usage:
    from src import <<function_name>>, ... || *

Last Update:
    Owner: Kartik M. Jalal
    Date: 20/09/2025

"""
import numpy as np
import pandas as pd
import os
from . import load_marker
from sklearn.model_selection import train_test_split
import tifffile


def get_tiff_img(img_path: str) -> np.ndarray:
    """
        Return the tiff image.

        Description:
            Reads and returns the tiff image at the given path 
            in numpy format.

        Inputs:
            - img_path (str): The path to the tiff image file.

        Outputs:
            - np.ndarray, (Z, Y, X) numpy format of the tiff image.
            
    """
    return tifffile.imread(img_path)


def get_data_info(data_path: str, tiff_suffixes: list, marker_suffix: str) -> pd.DataFrame:
    """
        Creates info about the data present.

        Description:
            This create a pandas dataframe which creates info about the data present at the given
            path (data_path). For every data the info includes:
                - "data_path", path to the data.
                - "img_name", tiff image file name.
                - "marker_name", corresponding marker file name.
                - "n_neurons", the number of neurons present in the marker file.
        
        Inputs:
            - data_path (str): The path where all the data is located.
            - tiff_suffixes (list[str,...]): List containing all the valid tiff files suffix.
            - marker_suffix (str): The marker file suffix.

        Outputs:
            - pd.DataFrame, containing info about the data.

    """
    # list to store all the info
    data_info = []

    # every file present at the data_path
    for file_name in os.listdir(data_path):
        # get the file suffix
        suffix = "." + file_name.split(".")[-1]
        if not suffix in tiff_suffixes:  # if the suffix is not the valid tiff file suffix
            continue # skip
        
        # add the marker suffix at the end of the file name
        # to create its corresponding marker file name,
        # e.g., img.tiff -> img.tiff.marker.
        marker_name = file_name + marker_suffix
        # check if its marker file exist
        marker_path = os.path.join(data_path, marker_name)
        if not os.path.exists(marker_path):
            print(f"Skipping img {file_name} because there is no corresponding {marker_suffix} file exist in the given path {data_path}.")
            continue # skip to the next file name
        
        # read the number of neurons present in the marker file
        n_neuron = len(load_marker(marker_path))

        # add all the info together
        data_info.append(
            {
                "data_path": data_path,
                "tiff_img_name": file_name,
                "marker_name": marker_name,
                "n_neurons": n_neuron
            }
        )
    
    # convert to pandas DataFrame
    df = pd.DataFrame(data_info)
    return df


def split_fraction(desired_fraction: float, already_taken_fraction: float = 0.0) -> float:
    """
        Returns the right split fraction based on how much data is left.

        Description:
            This calculates the right split fraction to use for the split given
            the desired fraction for the split and how much fraction of the data
            has already been used taken in a previous split.

        Inputs:
            - desired_fraction (float): The desired fraction for the split.
            - already_taken_fraction (float): Fraction used in the previous split.

        Outputs:
            - float, the right split fraction for the split.
    """
    return desired_fraction/(1-already_taken_fraction)


def train_val_test_split(
    df: pd.DataFrame, 
    val_size: float = 0.2, 
    test_size: float = 0.1, 
    straify: bool = True, 
    n_neurons_bins: int = 3
) -> tuple:
    """
        Splits the given data in to train, validation and test dataframes.

        Description:
            Ensures the data is split with the right split fraction and if stratification
            is specified (recommended) for the split - it first divides the data into
            multiple bins (n_neurons_bins tells how many bins to consider) which is
            based on the "n_neurons" (number of neurons present in the marker file) 
            column and then, stratifies the splits based on the distributions of the 
            bins.
        
        Inputs:
            - df (pd.DataFrame): The main DataFrame that needs to be split.
            - val_size (float): The fraction of data for the test DataFrame over the main DataFrame.
            - test_size (float): The fraction of validation for the test DataFrame over the main DataFrame.
            - straify (bool): Flag to specify the use of stratification.
            - n_neurons_bins (int): Number of bins to divide the data in based in the "n_neurons" column (if straify is True).

        outputs:
            - tuple(pd.DataFrame, pd.DataFrame, pd.DataFrame), the train, validation, test DataFrame
    """
    # if stratification is specified
    if straify:
        # divide the data into 'n_neurons_bins' bins (which is based on the "n_neurons" column),
        # and create a new column which contains the bin info for that data
        df["n_neurons_bins"] = pd.cut(df["n_neurons"], bins=n_neurons_bins, labels=False)

    # initial "train with val" and "test" split
    train_val, test = train_test_split(
        df, # main DataFrame
        test_size=split_fraction(desired_fraction=test_size, already_taken_fraction=0.0), # split fraction for the test DataFrame
        random_state=42, # random state (usful to create the same split later in the line)
        stratify=df["n_neurons_bins"] if straify else None # use stratification or not
    )

    # final "train" and "val" split
    train, val = train_test_split(
        train_val, # train with val DataFrame
        test_size=split_fraction(desired_fraction=val_size, already_taken_fraction=0.0), # split fraction for val DataFrame
        random_state=42,
        stratify=train_val["n_neurons_bins"] if straify else None
    )

    return train, val, test
    


def clamp_start(start: int, size: int, max_size: int) -> int:
    """
        Clamp a patch start index to keep the crop within bounds.

        Description:
            Ensures the crop [start: start+size] lies inside the valid axis range [0, max_size].
        
        Inputs:
            - start (int): proposed start index for the crop.
            - size (int): crop sizr along this axis.
            - max_size (int): total size of the axis.

        Outputs:
            - int, clamped start index.
    """
    return max(0, min(start, max_size - size))


def crop3d(vol : np.ndarray, start: tuple, size: tuple) -> np.ndarray:
    """
        Extract a 3D patch from a volume.

        Description:
            Takes a (Z, Y, X) volume and returns vol[z0:z0+dz, y0:y0+dy, x0:x0+dx].

        Inputs:
            - vol (np.ndarray): 3D volume (Z, Y, X).
            - start (tuple(int, int, int)): top-left-frony indicies (z0, y0, x0).
            - size (tuple(int, int, int)): patch size (dz, dy, dx).
.
        Outputs:
            - np.ndarray, 3D patch (dz, dy, dx).
    """
    z0, y0, x0 = start
    dz, dy, dx = size

    return vol[z0:z0+dz, y0:y0+dy, x0:x0+dx]


def choose_positive_start(
        neurons: np.ndarray,
        vol_shape: type,
        patch: tuple = (96, 96, 96),
        jitter_frac: float = 1/6
) -> tuple:
    """
        Pick a patch start near a random centrid (positive sampling).

        Description:
            Ensures the sampled patch contains ≥1 neuron. Adds small random spatial jitter
            so the neuron is not always perfectly centered, improving robustness.

        Inputs:
            - neurons (np.ndarray): shape(N, 3) of floar voxel coords (z, y, x).
            - vol_shape (tuple(int, int, int)): full volume shape (Z, Y, X).
            - patch (tuple(int, int, int)): desired patch size (dz, dy, dx).
            - jitter_frac (float): fraction of patch size for random jitter.

        Outputs:
            - tuple(int, int, int): start indices (z0, y0, x0) for cropping.
    """
    Z, Y, X = vol_shape
    dz, dy, dx = patch

    # choose a neuron at random
    nz, ny, nx = neurons[np.random.randint(len(neurons))]

    # small jitter to the neuron location proportional to patch size
    rnd_shift = lambda axis_size: int(np.round(np.random.uniform(-axis_size * jitter_frac, axis_size * jitter_frac)))
    # add the jitter
    nz_j = int(round(nz)) + rnd_shift(dz)
    ny_j = int(round(ny)) + rnd_shift(dy)
    nx_j = int(round(nx)) + rnd_shift(dx)

    z0 = clamp_start(
        start=(nz_j - dz // 2), size=dz, max_size=Z
    )
    y0 = clamp_start(
        start=(ny_j - dy // 2), size=dy, max_size=Y
    )
    x0 = clamp_start(
        start=(nx_j - dx // 2), size=dx, max_size=X
    )

    return (z0, y0, x0)


def choose_background_start(
    neurons: np.ndarray,
    vol_shape: tuple,
    patch: tuple = (96, 96, 96),
    min_dist: int = 8,
    trails: int = 100
) -> tuple:
    """
        Pick a patch start away from all centriods (background sampling).

        Description:
            Samples patches where there are no neurons within a small radius. This teaches the model
            the appearance of "no neuron" regions and reduces false positives.

        Inputs:
            - neurons (np.ndarray): shape(N, 3) or empty list if no neurrons.
            - vol_shape (tuple(int, int, int)): full volume shape (Z, Y, X).
            - patch (tuplee(int, int, int)): desired patch size (dz, dy, dx).
            - min_dist (int): minimum center-to-centroid distance (vox) for backgroud sampling/
            - trails (int): max random attempts before falling back

        Outputs:
            - tuple(int, int, int): start indices (z0, y0, x0) for cropping. 
    """
    Z, Y, X = vol_shape
    dz, dy, dx = patch
    n = np.asarray(neurons, dtype=np.float32) if (neurons is not None and len(neurons)) else None

    for _ in range(trails):
        # random patch center within valid range
        z_rand = np.random.randint(dz // 2, max(dz // 2 + 1, Z - dz // 2))
        y_rand = np.random.randint(dy // 2, max(dy // 2 + 1, Y - dy // 2))
        x_rand = np.random.randint(dx // 2, max(dx // 2 + 1, X - dx // 2))
        

        # if we have neurons, enforce min_dist from all
        if n is not None and len(n):
            d2 = ((n - np.array([z_rand, y_rand, x_rand], np.float32)) ** 2).sum(axis=1)
            if (d2 < (min_dist * min_dist)).any():
                continue # too close to a neuron, resample

        # convert center to valid start
        z0 = clamp_start(z_rand - dz // 2, dz, Z)
        y0 = clamp_start(y_rand - dy // 2, dy, Y)
        x0 = clamp_start(x_rand - dx // 2, dx, X)
        return (z0, y0, x0)
    
    # Fallback (rare), return a corner crop
    z0 = clamp_start(0, dz, Z)
    y0 = clamp_start(0, dy, Y)
    x0 = clamp_start(0, dx, X)
    return (z0, y0, x0)






