"""
ground_truth.py

Provides functions to generate ground truth from the given .marker* file

Functions:
    generate_ground_truth

Usage:
    from src import generate_ground_truth

Last Update:
    Owner: Kartik M. Jalal
    Date: 12/08/2025

"""

import numpy as np


def load_marker(
    marker_path,
    default_max_radius,
    use_marker_radius
):
    """
        Reads a '.marker*' file where each row has:
            x, y, z, radius, shape, name, comment

        Only the first 3 values (x, y, z) are important for finding neurron positions.

        This function:
            1. Skips the header row (lines starting with #')
            2. Reads coordinates from file.
            3. Reorders them from (X, Y, Z) to (Z, Y, X) format.
            4. Also collects the "radius" column from the file (if it exists),
               otherwise replace the corresponding radius with the default max radius.

        Returns:
            neurons: (N, 4) array with neurons coordinates in Z,Y,X order and thier max radius.
    """
    # coords and radii
    zs, ys, xs, max_radii = [], [], [], []

    with open(marker_path, "r") as f:
        for line in f:
            # Skip empty lines and header/comments 
            if not line.strip() or line.startswith("#"):
                continue

            neuron_info = [info.strip() for info in line.split(",")]
            # skips neurons that don't even have all x, y, z coords
            if len(neuron_info) < 3:
                continue

            # neuron coords (floats)
            x, y, z = map(float, neuron_info[:3])
            # append the coords
            zs.append(z)
            ys.append(y)
            xs.append(x)
                
            # radius for this neuron    
            radius = default_max_radius
            # given use .marker* file radius, if present - 4th column.
            if use_marker_radius and len(neuron_info) >= 4 and neuron_info[3] != "":
               radius = float(neuron_info[3])
            
            max_radii.append(radius)

        neurons = np.stack([zs, ys, xs, max_radii], axis=1).astype(np.float32)

        return neurons




def generate_ground_truth(
    marker_path,
    ground_truth_shape,
    dim_resolution, # Size of each voxel (Z, Y, X)
    use_marker_radius,
    default_max_radius=3.5,
    radius_safe_factor=3.5,
    radius_min=1.0,
    downscale_factor=None, # Scale coordinates if image is downsampled
    knn_k=1,
    knn_scale=0.33,
):
    """
        This function:
            1. Reads neuron centroids from .marker file.
            2. Filters them to keep only those inside the volumne.
            3. Decides the size each Gaussian blob should be.
            4. Creates a binary "peaks" map with 1s at neuron locations.
            5. Groups points with similar Blob sizes to save computation.
            6. Creates a smooth "heatmap" ground truth by blurring each group.

        Returns:
            - heatmap: final smooth ground truth
            - meta: details about points, radii, sigmas
    """

    # load neuron coords and their max radii
    neurons = load_marker(
        marker_path=marker_path,
        default_max_radius=default_max_radius,
        use_marker_radius=use_marker_radius
    )


