"""
dataset.py

Provides a dataset class for loading and serving to the model 

Classes:
    - NeuronPatchDataset

Usage:
    from src.classes import <<class_name>>, ... || *

Last Update:
    Owner: Kartik M. Jalal
    Date: 20/09/2025

"""

import torch
from torch.utils.data import Dataset 
import pandas as pd
from ..utils import (
    get_tiff_img,
    generate_ground_truth,
    clip_percentile,
    normalise_zscore,
    random_flip_xy,
    random_intensity_jitter,
    choose_positive_start,
    choose_background_start,
    crop3d
)
import numpy as np
import os

class NeuronPatchDataset(Dataset):
    """
        A Pytorch Dataset that samples 3D patches for training a two-head model
        (heatmap classification + offset regression).

        Description:
            Each __getitem__ returns  one 3D patch sampled from a full volume:
                - Positive patches: contain at least one centriod most of the time.
                - Background patches: contain no centroid some of the time.

            The same crop is applied to image and all ground truths. Gentle augmentations
            (XY flips, light intensity jittere) are applied consistently.

        Inputs:
            - df (pd.DataFrame): Dataframe which contains info about the data.
            - patch tuple(int, int, int): patch size (z, y, x), e.g. (96, 96, 96).
            - posi_patch_prob (float): Probability of sampling a positive patch, in [0, 1].
            - pclip (tuple(float, float)): voxel intensities percentile clip range for input image.

        Outputs (per __getitem__):
            - dict(
                image (torch.FloatTensor): (1, Z, Y, X), input image.
                heatmap (torch.FloatTensor): (1, Z, Y, X), neurons centriod gaussian heatmap.
                offsets_gt (torch.FloatTensor): (3, Z, Y, X), 3 channels where each voxel stores
                    a vector (Δz, Δy, Δx) pointing from voxel -> neuron centroid (only when 
                    in-within neuron sphere), i.e., how far to walk to reach the centroid.
                offset_mask (torch.FlaotTensor): (1, X, Y, X), binary mask with 1's where offsets are
                    defined (inside a neuron sphere), 0's elsewhere.
            )   
    """
    def __init__(
            self,
            df,
            ground_truth_config,
            pre_processing_config
    ):
        self.df = df
        self.ground_truth_config = ground_truth_config
        self.pre_processing_config = pre_processing_config

    
    def __len__(self) -> int:
        """
            The length of the DataFrame

            Description:
                Return the number of volumes (not patches) info present in the DataFrame.

            Inputs:
                - None
            
            Outputs:
                - int, number of base volumnes present in this DataFrame.
        """
        return len(self.df)
    

    def _get_tiff_img(self, index: int) -> np.ndarray:
        """
            Return the tiff image.

            Description:
                Returns the tiff image at the given path in numpy format.

            Inputs:
                - index (int), which base volume to read.
            
            Outputs:
                - np.ndarray, (Z, Y, X) numpy format of the tiff image.
        """
        # the path to the data
        data_path = self.df["data_path"][index]
        # the tiff image file name
        img_name = self.df["img_name"][index]
        
        img_path=os.path.join(data_path, img_name)
        return get_tiff_img(img_path=img_path)
    
    
    def _generate_ground_truth(self, index: int) -> dict:
        """
        """
        # Path to the marker
        data_path = self.df["data_path"][index] # the path to the data
        marker_name = self.df["marker_name"][index] # the marker file name

        marker_path=os.path.join(data_path, marker_name)

        gt = generate_ground_truth(
            marker_path=marker_path,
            **self.ground_truth_config
        )
        return gt
    

    def _normalise(self, vol: np.ndarray) -> np.ndarray:
        """
        """
        pclip = tuple(self.pre_processing_config['pclip'])
        clipped_vol = clip_percentile(vol, *pclip)

        return normalise_zscore(clipped_vol)
    

    def _apply_aug(
        self,
        volumes: dict,
    ) -> dict:
        """
        """
        # flip spatial dims
        volumes = random_flip_xy(**volumes)
        # light intensity jitter
        volumes["vol"] = random_intensity_jitter(vol=volumes["vol"])

        return volumes


    def __getitem__(self, index: int) -> dict:
        """
            Sample one 3D patch and return image and all ground truths as tensors.

            Description:
                Chooses a positive patch (if centroids exist and coin-flip passes) or a
                a background patch (otherwise). Crops image and all ground truths consistently.
                Applies gentle augmentations (XY flips, light intensity jitter (only the image))
                consistently.

            Inputs:
                - index (int): which base volume to draw a patch from.
            
            Outputs:
                - dict(
                    image (torch.FloatTensor): (1, Z, Y, X), input image.
                    heatmap (torch.FloatTensor): (1, Z, Y, X), neurons centriod gaussian heatmap.
                    offsets (torch.FloatTensor): (3, Z, Y, X), 3 channels where each voxel stores
                        a vector (Δz, Δy, Δx) pointing from voxel -> neuron centroid (only when 
                        in-within neuron sphere), i.e., how far to walk to reach the centroid.
                    offset_mask (torch.FlaotTensor): (1, X, Y, X), binary mask with 1's where offsets are
                        defined (inside a neuron sphere), 0's elsewhere.
                )
        """
        img_full = self._get_tiff_img(index)
        gt = self._generate_ground_truth(index)
        hm_full = gt["heatmap"]
        offsets_full = gt["offsets"]
        offset_mask_full = gt["offset_mask"]
        neurons = gt["meta"]["neurons"]

        # decide sampling type
        take_pos = (np.random.rand() < self.pre_processing_config['positive_patch_prob']) and (neurons is not None) and (len(neurons) > 0)
        if take_pos:
            start = choose_positive_start(
                neurons=neurons,
                vol_shape=img_full.shape,
                patch=self.pre_processing_config['patch']
            )
        else:
            start = choose_background_start(
                neurons=neurons,
                vol_shape=img_full.shape,
                patch=self.pre_processing_config['patch']
            )

        # performing cropping
        img = crop3d(
            vol=img_full,
            start=start,
            patch=self.pre_processing_config['patch']
        )
        hm = crop3d(
            vol=hm_full,
            start=start,
            patch=self.pre_processing_config['patch']
        )
        offset_mask = crop3d(
            vol=offset_mask_full,
            start=start,
            patch=self.pre_processing_config['patch']
        )

        offsets = np.stack([
            crop3d(
                vol=offsets_full[i],
                start=start,
                patch=self.pre_processing_config['patch']
            ) for i in range(offsets_full.shape[0])
        ], axis=0) # (3, dz, dy, dx)

        # normalise image
        img = self._normalise(vol=img)

        if self.pre_processing_config.get('perform_aug', False):
            # apply augmentations
            aug_vols = self._apply_aug(
                volumes=dict(
                    vol=img,
                    heatmap=hm,
                    offsets=offsets,
                    offset_mask=offset_mask
                )
            )
            img = aug_vols["vol"]
            hm = aug_vols["heatmap"]
            offsets = aug_vols["offsets"]
            offset_mask = aug_vols["offset_mask"]

        # convert to torch tensors (channel-first)
        img_tensor = torch.from_numpy(img[None, ...]).float() # (1, dz, dy, dx)
        hm_tensor = torch.from_numpy(hm[None, ...]).float() # (1, dz, dy, dx)
        offsets_tensor = torch.from_numpy(offsets).float() # (3, dz, dy, dx)
        offset_mask_tensor = torch.from_numpy(offset_mask[None, ...]).float() # (1, dz, dy, dx)

        return dict(
            image=img_tensor,
            heatmap=hm_tensor,
            offsets=offsets_tensor,
            offset_mask=offset_mask_tensor
        )



    