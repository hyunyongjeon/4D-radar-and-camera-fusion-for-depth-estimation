# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed
import cv2
import pickle

import torch
import torch.utils.data as data
from torchvision import transforms

from .mono_dataset import MonoDataset


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


# class MonoDataset(data.Dataset):
class mscrad4r(MonoDataset):    
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self, *args, **kwargs):
        # super(MonoDataset, self).__init__()
        super(mscrad4r, self).__init__(*args, **kwargs)

        self.full_res_shape = (720, 540)
        
        stereo_calib_path = './data/mscrad4r_pkl/stereo_calib/230402_cam_calib/stereo_calib.pkl'
        with open(stereo_calib_path, 'rb') as f:
            self.stereo_calib = pickle.load(f)
        
        mtx_left = self.stereo_calib['mtx_left']
        dist_left = self.stereo_calib['dist_left']
        mtx_right = self.stereo_calib['mtx_right']
        dist_right = self.stereo_calib['dist_right']
        R = self.stereo_calib['R']
        T = self.stereo_calib['T']
        E = self.stereo_calib['E']
        F = self.stereo_calib['F']
        # camera matrices ('mtx_left','mtx_right')
        # distortion coefficients ('dist_left','dist_right')
        # rotation matrix between the 1st and the 2nd cameras ('R')
        # translation vector between the coordinate systems of the cameras ('T')
        # essential matrix ('E')
        # fundamental matrix ('F')

        # stereo rectification
        R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
            mtx_left, dist_left, mtx_right, dist_right, self.full_res_shape, R, T)
        # Rectification matrices ('R1','R2')
        # Projection matrices ('P1','P2')
        # Disparity-to-depth mapping matrix ('Q')
        # Region of interest for the rectified images ('roi_left','roi_right')
        self.map_left1, self.map_left2 = cv2.initUndistortRectifyMap(
            mtx_left, dist_left, R1, P1, self.full_res_shape, cv2.CV_16SC2)
        # Generates retification maps ('map_left1','map_left2')
        
        K = np.eye(4, dtype=np.float32)
        K[:3, :3] = P1[:3, :3]
        K[0, 0] /= self.full_res_shape[0]
        K[1, 1] /= self.full_res_shape[1]
        K[0, 2] /= self.full_res_shape[0]
        K[1, 2] /= self.full_res_shape[1]
        self.K = K
        
    def stereo_rectify(self, img_left):
        img_left_rectified = cv2.remap(
            img_left, self.map_left1, self.map_left2, cv2.INTER_LINEAR)
        return img_left_rectified

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
                
    def radar_preprocess(self, inputs, radar_points):
        """
        Transform the numpy array to dictionary type to add the key name.

        Args:
            radar_points(numpy array):
        
        Returns:
            1. radar_x                       for x coordinate
            2. radar_y                       for y coordinate
            3. radar_z                       for z coordinate
            4. adar_s_0000                   not used
            5. adar_s_0001                   not used
            6. adar_s_0002                   not used
            7. adar_s_0003                   not used
            8. radar_alpha                   for azimuzh angle
            9. radar_beta                    for elevation angle
            10. radar_range                  for the distance(m)
            11. radar_doppler                for the relative velocity(m/s)
                                             The static object can be easily classified through this value.
            12. radar_power                  for the signal-to-ratio value of the transmitted power(dB)
            13. radar_recoveredSpeed         for the absolute velocity in the radial direction of target object(m/s)
            14. radar_dotFlages              not used  
            15. radar_denoiseFlag            not used
                                             as uint16 data types
                                             The static object can be easily classified through this flag,
                                             but, to classify static object, we can use doppler value instead.[MSC-RAD4R]
            16. radar_historyFrmaeFlag       not used 
            17. radar_dopplerCorrectionFlag  not used
        """
        inputs["radar_x"] = torch.from_numpy(radar_points[:,0])
        inputs["radar_y"] = torch.from_numpy(radar_points[:,1])
        inputs["radar_z"] = torch.from_numpy(radar_points[:,2])
        # inputs["radar_s_0000"] = torch.from_numpy(radar_points[:,3])
        # inputs["radar_s_0001"] = torch.from_numpy(radar_points[:,4])
        # inputs["radar_s_0002"] = torch.from_numpy(radar_points[:,5])
        # inputs["radar_s_0003"] = torch.from_numpy(radar_points[:,6])
        inputs["radar_alpha"] = torch.from_numpy(radar_points[:,7])
        inputs["radar_beta"] = torch.from_numpy(radar_points[:,8])
        inputs["radar_range"] = torch.from_numpy(radar_points[:,9])
        inputs["radar_doppler"] = torch.from_numpy(radar_points[:,10])
        inputs["radar_power"] = torch.from_numpy(radar_points[:,11])
        inputs["radar_recoveredSpeed"] = torch.from_numpy(radar_points[:,12])
        # inputs["radar_dotFlages"] = torch.from_numpy(radar_points[:,13])
        # inputs["radar_denoiseFlag"] = torch.from_numpy(radar_points[:,14])
        # inputs["radar_historyFrmaeFlag"] = torch.from_numpy(radar_points[:,15])
        # inputs["radar_dopplerCorrectionFlag"] = torch.from_numpy(radar_points[:,16])
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        inputs = {}
        mono_inputs = self.mono_getitem(index)
        radar_inputs = self.radar_getitem(index)
        inputs.update(mono_inputs)
        inputs.update(radar_inputs)
        return inputs
        
    def radar_getitem(self, index):
        """
        Adding radar point-cloud information to the 'inputs' variable.
        """
        inputs = {}
        folder = self.radar_filenames[index].split()[0]
        # the index is out of range of the radar data. How to solve this problem?
        radar_path = os.path.join(self.data_path, folder,"{:010d}.bin".format(int(self.radar_filenames[index].split()[1])))
        radar_points = np.fromfile(radar_path, dtype=np.float32).reshape(-1, 17)

        self.radar_preprocess(inputs,radar_points)
        
        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
    
    
# class mscrad4r_mono(mscrad4r):
#     def __init__(self, *args, **kwargs):
#         super(mscrad4r_mono, self).__init__(*args, **kwargs)
        
#     def __getitem__(self, index):
#         inputs = self.mono_getitem(index)
#         return inputs