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

import torch
import torch.utils.data as data
from torchvision import transforms


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


# class MonoDataset(data.Dataset):
class mscrad4r(data.Dataset):    
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
    def __init__(self,
                 data_path,
                 filenames,
                 radar_filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg'):
        # super(MonoDataset, self).__init__()
        super(mscrad4r, self).__init__()
        
        self.data_path = data_path
        self.filenames = filenames
        self.radar_filenames = radar_filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = self.check_depth()

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
                
    def radar_preprocess(self, radar_points):
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
        radar_x = torch.from_numpy(radar_points[:,0])
        radar_y = torch.from_numpy(radar_points[:,1])
        radar_z = torch.from_numpy(radar_points[:,2])
        # radar_s_0000 = torch.from_numpy(radar_points[:,3])
        # radar_s_0001 = torch.from_numpy(radar_points[:,4])
        # radar_s_0002 = torch.from_numpy(radar_points[:,5])
        # radar_s_0003 = torch.from_numpy(radar_points[:,6])
        radar_alpha = torch.from_numpy(radar_points[:,7])
        radar_beta = torch.from_numpy(radar_points[:,8])
        radar_range = torch.from_numpy(radar_points[:,9])
        radar_doppler = torch.from_numpy(radar_points[:,10])
        radar_power = torch.from_numpy(radar_points[:,11])
        radar_recoveredSpeed = torch.from_numpy(radar_points[:,12])
        # radar_dotFlages = torch.from_numpy(radar_points[:,13])
        # radar_denoiseFlag = torch.from_numpy(radar_points[:,14])
        # radar_historyFrmaeFlag = torch.from_numpy(radar_points[:,15])
        # radar_dopplerCorrectionFlag = torch.from_numpy(radar_points[:,16])
        
        return radar_x, radar_y, radar_z, radar_alpha, radar_beta, radar_range, radar_doppler, radar_power, radar_recoveredSpeed

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            # color_aug = transforms.ColorJitter.get_params(
            #     self.brightness, self.contrast, self.saturation, self.hue)
            color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)
            
        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)
            
        """
        Adding radar point-cloud information to the 'inputs' variable.
        """
        folder = self.radar_filenames[index].split()[0]
        radar_path = os.path.join(self.data_path, folder,"{:010d}.bin".format(int(index)))
        radar_points = np.fromfile(radar_path, dtype=np.float32).reshape(-1, 17)

        radar_x, radar_y, radar_z, radar_alpha, radar_beta, radar_range, \
            radar_doppler, radar_power, radar_recoveredSpeed = self.radar_preprocess(radar_points)
        
        # radar_features = []
        # radar_features.append(radar_info)
        
        inputs[("radar_x")] = radar_x
        inputs[("radar_y")] = radar_y
        inputs[("radar_z")] = radar_z
        inputs[("radar_alpha")] = radar_alpha
        inputs[("radar_beta")] = radar_beta
        inputs[("radar_range")] = radar_range
        inputs[("radar_doppler")] = radar_doppler
        inputs[("radar_power")] = radar_power
        inputs[("radar_recoveredSpeed")] = radar_recoveredSpeed
        
        # import pdb; pdb.set_trace()
        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError