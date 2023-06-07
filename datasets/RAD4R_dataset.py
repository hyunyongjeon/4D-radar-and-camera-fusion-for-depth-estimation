# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset


class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color
    

class MonoRadarDataset(KITTIDataset):    
# class MonoRadarDataset(MonoDataset):
    def __init__(self,*args,**kwargs):
        super(MonoRadarDataset, self).__init__(*args,**kwargs)   
        
    def preprocess(self, radar_points):
        radar_info = {}
        
        radar_info['x'] = radar_points[:,0]
        radar_info['y'] = radar_points[:,1]
        radar_info['z'] = radar_points[:,2]
        radar_info['s_0000'] = radar_points[:,3]
        radar_info['s_0001'] = radar_points[:,4]
        radar_info['s_0002'] = radar_points[:,5]
        radar_info['s_0003'] = radar_points[:,6]
        radar_info['alpha'] = radar_points[:,7]
        radar_info['beta'] = radar_points[:,8]
        radar_info['range'] = radar_points[:,9]
        radar_info['doppler'] = radar_points[:,10]
        radar_info['power'] = radar_points[:,11]
        radar_info['recoveredSpeed'] = radar_points[:,12]
        radar_info['dotFlages'] = radar_points[:,13]
        radar_info['denoiseFlag'] = radar_points[:,14]
        radar_info['historyFrmaeFlag'] = radar_points[:,15]
        radar_info['dopplerCorrectionFlag'] = radar_points[:,16]
        
        return radar_info

    def __getitem__(self,index):
        folder = self.radar_filenames[index].split()[0]
        radar_path = os.path.join(self.data_path, folder,"{:010d}.bin".format(int(index)))
        radar_points = np.fromfile(radar_path, dtype=np.float32).reshape(-1, 17)
        
        # x,y,z,alpha,beta,range,doppler,power,recoveredSpeed = self.preprocess(radar_points)
        radar_info = self.preprocess(radar_points)
        
        # import pdb; pdb.set_trace()
        radar_features = {}
        radar_features['radar_point'] = radar_info
        
        return radar_features
        
    
class RAD4RRAWDataset(MonoRadarDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(RAD4RRAWDataset, self).__init__(*args, **kwargs)

    # def get_image_path(self, folder, frame_index, side):
    def get_image_path(self, folder, frame_index, side):        
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            # self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
            self.data_path, folder, f_str)
        return image_path