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
from scipy.spatial.distance import cdist # used to measure distances between points
from scipy.ndimage import gaussian_filter  # used to blur images to create smooth blobs


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


def filter_neurons(neurons, ground_truth_shape):
    """
        Remove any neurons that lie outside the ground truth volume.

        - neurons:  an array of (z, y, x, max_radius).
        - ground_truth_shape: size of the ground truth in each dimension: (Z, Y, X).

        A neuron is valid if its coordinates:
            0 < z < Z
            0 < y < Y
            0 < x < X
    """

    z_n, y_n, x_n = neurons[:, 0], neurons[:, 1], neurons[:, 2]
    z_gt, y_gt, x_gt = ground_truth_shape

    filtration = (z_n > 0) & (y_n > 0) & (x_n > 0) & (z_n > z_gt) & (y_n > y_gt) & (x_n > x_gt)

    return neurons[filtration]


def knn_spacing(neurons_coords, k=1):
    """
        For each neurons, find the distance to its k-th nearest neighbor. It's 
        yseful to adjust neuron blob size based on how crowded the region is.
        Here, 'distance' means the stright-line distance in 3D voxel space 

        Returns:
            Array of k-th nearest neighbor distances for each neuron.
    """

    # Calculate the euclidean distance between every pair of neurons
    distances = cdist(
        neurons_coords,
        neurons_coords,
        metric="euclidean"
    )

    # Ignore distance to itself (set to infinity so it's not picked as nearest)
    np.fill_diagonal(distances, np.inf)

    # Sort each row; nearest neighbor is at index 0
    distances_sorted = np.sort(distances, axis=1)

    # k-th nearest neighbor distances for each neuron
    kth_distances = distances_sorted[:, k-1] if distances_sorted.shape[1] >= k else distances_sorted[:, -1]
    return kth_distances.astype(np.float32)


def compute_radii(
    neurons,
    radius_safe_factor, # Used for overlap avoidance
    knn_k,              # k-th neighbor to consider in "knn"
    knn_radius_scale,          # Fraction of neighbor distance to use for radius in "knn"
    radius_min          # Minimum allowed radius
):
    """
        Decide how big each Gaussian blob should be for each neuron.

        Strategies:
            adapt neuron blob radius to local density
    """
    # the k-th nearest neighbor distances for each neuron
    kth_distances = knn_spacing(
        neurons_coords=neurons[:, :-1],
        k=knn_k
    )

    # Take a fraction of that kth distance for neuron blob radius
    adapted_radii = knn_radius_scale * kth_distances
    # Make sure it's not too small
    adapted_radii = np.maximum(adapted_radii, radius_min)
    # make sure it's not to big from the default max radius allowed
    deafult_max_radius = neurons[:, -1]
    adapted_radii = np.minimum(adapted_radii, deafult_max_radius)

    return adapted_radii.astype(np.float32)


def radius_to_sigma_per_axis(radius, dim_resolution):
    """
        Convert a radius in voxel units into Gaussian blur sigmas 
        for each axis (Z, Y, X), as, we want each neuron target to be
        a roughly spherical blob in real-world space (µm - dim_resolution),
        not in the raw voxel grid, and  if voxels are anisotropic 
        (e.g., Z slices are thicker than Y/X pixels), the Gaussian sigma must
        be scaled differently along each axis to avoid stretched blobs.

        Returns:
            - dim_sigma : np.ndarray, shape (3,) 
                The Gaussian sigma for each axis (Z, Y, X) in voxel units, 
                adjusted so that the physical blob size is spherical.
    """
    dim_resolution = np.asarray(dim_resolution, dtype=np.float32) #for safe math

    # calculate a base sigma to ensures scaling is consistent with smallest dimension.
    # max(1.0, ...) ensures sigma is at least 1 voxel in the smallest axis,
    # which avoids a Gaussian narrower than a single voxel.
    base_sigma = max(1.0, float(radius)/float(np.min(dim_resolution)))

    # For each axis, we scale the base sigma so that physically it's the same size, i.e,
    # If Z voxels are bigger (e.g., 2.0 vs. 0.65 µm), this will produce a smaller
    # sigma in voxel units for Z to keep the real-world blob radius equal.
    dim_sigma = base_sigma / (dim_resolution / np.min(dim_resolution))
    return dim_sigma.astype(np.float32) # (sigma_z, sigma_y, sigma_x)


def generate_ground_truth(
    marker_path,
    ground_truth_shape,
    dim_resolution, # Size of each voxel (Z, Y, X)
    use_marker_radius,
    default_max_radius=3.5,
    radius_safe_factor=3.5,
    radius_min=1.0,
    downscale_factors=None, # Scale coordinates if image is downsampled
    knn_k=1,
    knn_radius_scale=0.33,
    quantize_sigma=0.1 # Round sigma values to reduce unique cases
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
            Dictionary containing:
                - heatmap: final smooth ground truth
                - peaks: ground truth with only 1.0 spikes at neurons coords
                - meta: details about neurons, sigmas
    """

    # load neuron coords and their max radii
    neurons = load_marker(
        marker_path=marker_path,
        default_max_radius=default_max_radius,
        use_marker_radius=use_marker_radius
    )

    # if no neurons were present, return an empty ground truth heatmap
    if len(neurons) == 0:
        return dict(
            heatmap=np.zeros(ground_truth_shape, np.float32), # empty heat map
            peaks=np.zeros(ground_truth_shape, np.float32), # empty peaks map
            meta=dict(
                neurons=neurons,
                sigmas=np.array([])
            )
        )

    # apply downscling if needed
    if downscale_factors is not None:
        # multiplying downscale_factors to the neuron coordinates
        neurons[:,:-1] *=  np.asarray(downscale_factors, dtype=np.float32)
    
    # filter to keep only in-bound neurons
    neurons = filter_neurons(
        neurons=neurons,
        ground_truth_shape=ground_truth_shape
    )

    # if all the neurons are filtered out, return an empty ground truth heatmap
    if len(neurons) == 0:
        return dict(
            heatmap=np.zeros(ground_truth_shape, np.float32), # empty heat map
            peaks=np.zeros(ground_truth_shape, np.float32), # empty peaks map
            meta=dict(
                neurons=neurons,
                sigmas=np.array([])
            )
        )
    
    # compute per neuron adaptive radii to mitigate overlaps
    neurons[:,-1] = compute_radii(
        neurons,
        radius_safe_factor=radius_safe_factor,
        knn_k=knn_k,
        knn_radius_scale=knn_radius_scale,
        radius_min=radius_min
    )

    # convert radii to sigma per axis
    sigmas = np.stack(
        arrays=[radius_to_sigma_per_axis(radius, dim_resolution) for radius in neurons[:,-1]],
        axis=0 # sigma per axis per radius
    )


    # Group points by similar sigma values to avoid blurring individually
    sigma_keys = [
            tuple(
                map(
                    lambda sigma: float(np.round(sigma / quantize_sigma) * quantize_sigma),
                    per_axis_sigmas
                )
            ) 
        for per_axis_sigmas in sigmas
    ]
    sigma_groups = {}
    for idx, sigma_key in enumerate(sigma_keys):
        sigma_groups.setdefault(sigma_key, []).append(idx)

    # rounding up the neurons coords from float to int
    iz = np.minimum(np.round(neurons[:, 0]).astype(int), ground_truth_shape[0]-1)
    iy = np.minimum(np.round(neurons[:, 1]).astype(int), ground_truth_shape[1]-1)
    ix = np.minimum(np.round(neurons[:, 2]).astype(int), ground_truth_shape[2]-1)

    # build a binary peak map, i.e., only with neuron coords
    peaks = np.zeros(ground_truth_shape, dtype=np.float32)
    peaks[iz, iy, ix] = 1.0

    # Create heatmap by adding each group's blurred spikes
    heatmap = np.zeros(ground_truth_shape, dtype=np.float32)
    # for every sigma key
    for sigma_key, idxs in sigma_groups.items():
        # temporary space
        temp = np.zeros(ground_truth_shape, dtype=np.float32)
        # spike neurons centroid
        temp[iz[idxs], iy[idxs], iz[idxs]] = 1.0
        # apply gaussian burring with sigma key
        temp = gaussian_filter(
            input=temp,
            sigma=sigma_key,
            turncate=4.0,
            mode="constant"
        )
        # normalise the temp space
        if temp.max() > 0:
            temp /= temp.max()

        # add the info to the heatmap
        heatmap += temp

    # normalise the heatmap
    hm_max = float(heatmap.max())
    if hm_max > 0:
        heatmap /= hm_max

    return dict(
        heatmap=heatmap,
        peaks=peaks,
        meta=dict(
            neurons=neurons,
            sigmas=sigmas
        )
    )




