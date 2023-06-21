# python -m datasets.mscrad4r.create_dataset

import pickle
import numpy as np
import cv2
import glob
import argparse
import os

from tqdm import tqdm
from pathlib import Path

import pdb


class DataBuilder:
    def __init__(self, training_data_paths, testing_data_paths, extrinsic_calib_path, stereo_calib_path, save_path):
        self.training_data_paths = training_data_paths
        self.testing_data_paths = testing_data_paths
        self.extrinsic_calib_path = extrinsic_calib_path
        self.stereo_calib_path = stereo_calib_path

        # read stereo calibration results
        with open(stereo_calib_path, 'rb') as f:
            self.stereo_calib = pickle.load(f)
        # # read extrinsic calibration results
        # self.extrinsic_calib = {}
        # with open(os.path.join(extrinsic_calib_path, 'stereo_lidar_extrinsics.pkl'), 'rb') as f:
        #     self.extrinsic_calib['stereo_lidar_extrinsics'] = pickle.load(f)
        # with open(os.path.join(extrinsic_calib_path, 'radar_lidar_extrinsics.pkl'), 'rb') as f:
        #     self.extrinsic_calib['radar_lidar_extrinsics'] = pickle.load(f)
        # # compute stereo camera to radar extrinsics from stereo camera to lidar extrinsics and lidar to radar extrinsics
        # self.extrinsic_calib['stereo_radar_extrinsics'] = np.matmul(
        #     self.extrinsic_calib['radar_lidar_extrinsics'],
        #     np.linalg.inv(self.extrinsic_calib['stereo_lidar_extrinsics']))
        
        self.save_path = save_path
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        
        self.training_save_path = os.path.join(self.save_path, 'training/')
        self.testing_save_path = os.path.join(self.save_path, 'testing/')
        Path(self.training_save_path).mkdir(parents=True, exist_ok=True)
        Path(self.testing_save_path).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.training_save_path, 'images')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.training_save_path, 'radar')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.testing_save_path, 'images')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.testing_save_path, 'radar')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.testing_save_path, 'lidar')).mkdir(parents=True, exist_ok=True)

        self.splits_save_path = os.path.join(self.save_path, 'splits/')
        Path(self.splits_save_path).mkdir(parents=True, exist_ok=True)

    def build_train(self):
        idx_sr = 0
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        # make a split file
        with open(os.path.join(self.splits_save_path, 'train_files.txt'), 'w') as f:
            f.write('')
        for training_data_path in tqdm(self.training_data_paths):
            seq_name = training_data_path.split('/')[-1]
            Path(os.path.join(self.training_save_path, 'images', seq_name)).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(self.training_save_path, 'radar', seq_name)).mkdir(parents=True, exist_ok=True)

            # read left camera pickle files
            with open(os.path.join(training_data_path, 'cam_left.pkl'), 'rb') as f:
                cam_left_msgs = pickle.load(f)
            with open(os.path.join(training_data_path, 'radar.pkl'), 'rb') as f:
                radar_msgs = pickle.load(f)

            # extract synced data and save them
            cam_left_msgs_time = cam_left_msgs['time']
            radar_msgs_time = radar_msgs['time']

            # left camera and radar synced data
            cam_left_msg_time_prev = None
            for i in range(len(cam_left_msgs_time)):
                cam_left_msg_time = cam_left_msgs_time[i]
                if cam_left_msg_time_prev is None:
                    cam_left_msg_time_prev = cam_left_msg_time
                    continue

                delta_time = cam_left_msg_time - cam_left_msg_time_prev
                if delta_time < 0.1:
                    continue

                # find the closest msgs in radar_msgs_time
                radar_msg_id = np.argmin(np.abs(radar_msgs_time - cam_left_msg_time))

                # check if the time difference is within 1ms
                timegap_radar = np.abs(cam_left_msg_time - radar_msgs_time[radar_msg_id]) / 1e9
                if timegap_radar > 0.01:
                    continue

                print(f'timegap_radar: {timegap_radar}')
                image_left = cam_left_msgs['msg'][i]
                image_left_rect = self.stereo_rectify(image_left)
                # save left images
                cv2.imwrite(os.path.join(self.training_save_path, 'images', seq_name, f'{idx_sr}.png'), image_left_rect)
                # save radar data
                radar_msg = radar_msgs['msg'][radar_msg_id]
                self.save_pcl_bin(radar_msg, os.path.join(self.training_save_path, 'radar', f'{seq_name}_{idx_sr}.bin'))

                # write the name of the saved files to the split file
                with open(os.path.join(self.splits_save_path, 'train_files.txt'), 'a') as f:
                    f.write(f'{seq_name} {idx_sr} l\n')

                idx_sr += 1
                cam_left_msg_time_prev = cam_left_msg_time

    def read_pcl(self, pcl_path):
        with open(pcl_path, 'rb') as f:
            pcl = pickle.load(f)
        return pcl
    
    def save_pcl_bin(self, pcl, pcl_path):
        stacked_points = np.transpose(np.vstack((\
            pcl['x'], pcl['y'], pcl['z'], \
            pcl['alpha'], pcl['beta'], \
            pcl['range'], pcl['doppler'], pcl['power'], pcl['recoveredSpeed'], \
            pcl['dotFlags'], pcl['denoiseFlag'], pcl['historyFrameFlag'], pcl['dopplerCorrectionFlag']
            )))
        stacked_points.tofile(pcl_path)
  

    def stereo_rectify(self, img_left):
        mtx_left = self.stereo_calib['mtx_left']
        dist_left = self.stereo_calib['dist_left']
        mtx_right = self.stereo_calib['mtx_right']
        dist_right = self.stereo_calib['dist_right']
        R = self.stereo_calib['R']
        T = self.stereo_calib['T']
        E = self.stereo_calib['E']
        F = self.stereo_calib['F']

        img_shape = img_left.shape[:2][::-1]
        # stereo rectification
        R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
            mtx_left, dist_left, mtx_right, dist_right, img_shape, R, T)
        map_left1, map_left2 = cv2.initUndistortRectifyMap(
            mtx_left, dist_left, R1, P1, img_shape, cv2.CV_16SC2)

        img_left_rectified = cv2.remap(img_left, map_left1, map_left2, cv2.INTER_LINEAR)

        return img_left_rectified

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data/mscrad4r_pkl/')
# accept multiple data names
parser.add_argument('--training_dirs', type=str, nargs='+',
                    default=['URBAN_A0', 'URBAN_B0', 'URBAN_D0', 'URBAN_E0', 'URBAN_F0', 'URBAN_G0',])  # only day
parser.add_argument('--testing_dirs', type=str, nargs='+',
                    default=['URBAN_H0'])
parser.add_argument('--extrinsic_calib', type=str, default='./data/mscrad4r_pkl/extrinsic_calib/data/')
parser.add_argument('--stereo_calib', type=str, default='./data/mscrad4r_pkl/stereo_calib/230402_cam_calib/stereo_calib.pkl')
parser.add_argument('--save_dir', type=str, default='./data/mscrad4r_dataset/')
args = parser.parse_args()


if __name__ == '__main__':
    training_data_paths = [os.path.join(args.data_dir, training_dir) for training_dir in args.training_dirs]
    testing_data_paths = [os.path.join(args.data_dir, testing_dir) for testing_dir in args.testing_dirs]
    extrinsic_calib_path = args.extrinsic_calib
    stereo_calib_path = args.stereo_calib
    save_path = args.save_dir
    databuilder = DataBuilder(training_data_paths, testing_data_paths, extrinsic_calib_path, stereo_calib_path, save_path)
    databuilder.build_train()
    databuilder.build_test()
