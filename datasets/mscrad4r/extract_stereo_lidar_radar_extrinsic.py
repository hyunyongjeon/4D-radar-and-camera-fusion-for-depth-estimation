# read radar, lidar, stereo camera pickle files
# extract synced files and save the radar and lidar data as pickle files
# and save the stereo images as png files for later use
# use previously computed stereo calibration results to rectify the stereo images
# use deep learning model to compute disparity map and unproject the disparity map to point cloud
# use the point cloud to compute the extrinsic parameters between the stereo camera, lidar, and radar
# plot stereo, lidar, and radar point cloud in the same coordinate system to validate the extrinsic parameters
# save the extrinsic parameters as pickle files for later use
# python -m datasets.mscrad4r.extract_stereo_lidar_radar_extrinsic

import pickle
import numpy as np
import cv2
import glob
import argparse
import os

from tqdm import tqdm
from pathlib import Path
from ruamel.yaml import YAML
from multiprocessing import Process, Queue

import kornia
import kornia.feature as KF

from .icp import icp, multi_icp, best_fit_transform, nearest_neighbor
from .vis3d import o3dvis, pcdBuffer

import torch
os.environ["CUDA_VISIBLE_DEVICES"]="0"

torch.set_grad_enabled(False)

import pdb


class RadarLidarStereo:
    def __init__(self, data_paths, save_path, stereo_calib_path):
        self.data_paths = data_paths
        self.save_path = save_path

        # read stereo calibration results
        with open(stereo_calib_path, 'rb') as f:
            self.stereo_calib = pickle.load(f)
        

        # keypoint matchers
        # contrastThreshold=0.04  # default=0.04
        contrastThreshold=0.001
        self.sift = cv2.SIFT_create(contrastThreshold=contrastThreshold)
        self.bf = cv2.BFMatcher()
        # loftr
        loftr_cfg = KF.loftr.loftr.default_cfg
        # loftr_cfg['match_coarse']['thr'] = 0.001
        self.loftr = KF.LoFTR(pretrained="outdoor", config=loftr_cfg).eval().cuda()
        print(f"\x1b[1;30;103m model: LoFTR outdoor\x1b[0m")
        
        self.pcdBuffer = pcdBuffer()
        self.o3dvis = None

    def read(self):
        idx_sl, idx_lr, idx_sr, idx_slr = 0, 0, 0, 0
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        for data_path in tqdm(self.data_paths):
            # read stereo camera pickle files
            with open(os.path.join(data_path, 'cam_left.pkl'), 'rb') as f:
                cam_left_msgs = pickle.load(f)
            with open(os.path.join(data_path, 'cam_right.pkl'), 'rb') as f:
                cam_right_msgs = pickle.load(f)
            with open(os.path.join(data_path, 'lidar.pkl'), 'rb') as f:
                lidar_msgs = pickle.load(f)
            with open(os.path.join(data_path, 'radar.pkl'), 'rb') as f:
                radar_msgs = pickle.load(f)

            # extract synced data and save them
            cam_left_msgs_time, cam_right_msgs_time = cam_left_msgs['time'], cam_right_msgs['time']
            lidar_msgs_time, radar_msgs_time = lidar_msgs['time'], radar_msgs['time']

            # stereo camera and lidar first
            for i in range(len(cam_left_msgs_time)):
                cam_left_msg_time = cam_left_msgs_time[i]

                # find the closest msgs in cam_right_msgs_time, lidar_msgs_time, radar_msgs_time
                cam_right_msg_id = np.argmin(np.abs(cam_right_msgs_time - cam_left_msg_time))
                lidar_msg_id = np.argmin(np.abs(lidar_msgs_time - cam_left_msg_time))

                # check if the time difference is within 1ms
                timegap_right = np.abs(cam_left_msg_time - cam_right_msgs_time[cam_right_msg_id]) / 1e9
                timegap_lidar = np.abs(cam_left_msg_time - lidar_msgs_time[lidar_msg_id]) / 1e9
                if timegap_right < 0.0005 and timegap_lidar < 0.0005:
                    print(f'timegap_right: {timegap_right}, timegap_lidar: {timegap_lidar}')
                    # save stereo images
                    cv2.imwrite(os.path.join(self.save_path, 'cam_left_' + str(idx_sl) + '_sl.png'), cam_left_msgs['msg'][i])
                    cv2.imwrite(os.path.join(self.save_path, 'cam_right_' + str(idx_sl) + '_sl.png'), cam_right_msgs['msg'][cam_right_msg_id])
                    # save lidar data
                    with open(os.path.join(self.save_path, 'lidar_' + str(idx_sl) + '_sl.pkl'), 'wb') as f:
                        pickle.dump(lidar_msgs['msg'][lidar_msg_id], f)
                
                    idx_sl += 1

            # then lidar and radar
            for i in range(len(lidar_msgs_time)):
                lidar_msg_time = lidar_msgs_time[i]

                # find the closest msgs in radar_msgs_time
                radar_msg_id = np.argmin(np.abs(radar_msgs_time - lidar_msg_time))

                # check if the time difference is within 1ms
                timegap_radar = np.abs(lidar_msg_time - radar_msgs_time[radar_msg_id]) / 1e9
                if timegap_radar < 0.001:
                    print(f'timegap_radar: {timegap_radar}')
                    # save radar data
                    with open(os.path.join(self.save_path, 'radar_' + str(idx_lr) + '_lr.pkl'), 'wb') as f:
                        pickle.dump(radar_msgs['msg'][radar_msg_id], f)
                    # save lidar data
                    with open(os.path.join(self.save_path, 'lidar_' + str(idx_lr) + '_lr.pkl'), 'wb') as f:
                        pickle.dump(lidar_msgs['msg'][i], f)

                    idx_lr += 1

            # then stereo camera and radar
            for i in range(len(cam_left_msgs_time)):
                cam_left_msg_time = cam_left_msgs_time[i]

                # find the closest msgs in radar_msgs_time
                radar_msg_id = np.argmin(np.abs(radar_msgs_time - cam_left_msg_time))
                cam_right_msg_id = np.argmin(np.abs(cam_right_msgs_time - cam_left_msg_time))

                # check if the time difference is within 1ms
                timegap_radar = np.abs(cam_left_msg_time - radar_msgs_time[radar_msg_id]) / 1e9
                timegap_right = np.abs(cam_left_msg_time - cam_right_msgs_time[cam_right_msg_id]) / 1e9
                if timegap_radar < 0.0001 and timegap_right < 0.0001:
                    print(f'timegap_radar: {timegap_radar}')
                    # save stereo images
                    cv2.imwrite(os.path.join(self.save_path, 'cam_left_' + str(idx_sr) + '_sr.png'), cam_left_msgs['msg'][i])
                    cv2.imwrite(os.path.join(self.save_path, 'cam_right_' + str(idx_sr) + '_sr.png'), cam_right_msgs['msg'][cam_right_msg_id])
                    # save radar data
                    with open(os.path.join(self.save_path, 'radar_' + str(idx_sr) + '_sr.pkl'), 'wb') as f:
                        pickle.dump(radar_msgs['msg'][radar_msg_id], f)

                    idx_sr += 1


    def read_pcl(self, pcl_path):
        with open(pcl_path, 'rb') as f:
            pcl = pickle.load(f)
        return pcl

    def compute_extrinsics(self):
        print('computing extrinsics between stereo camera, lidar, and radar...')
        self.stereo_lidar_extrinsics()
        self.lidar_radar_extrinsics()

    def verify_extrinsics(self):
        print('verifying extrinsics between stereo camera, lidar, and radar...')
        # self.verify_stereo_lidar_extrinsics()
        # self.verify_lidar_radar_extrinsics()
        self.verify_stereo_radar_extrinsics()

    # def stereo_lidar_extrinsics(self):
    #     print(' first is between stereo camera and lidar...')
    #     # read data
    #     img_left_paths = glob.glob(os.path.join(self.save_path, 'cam_left_*_sl.png'))
    #     img_right_paths = glob.glob(os.path.join(self.save_path, 'cam_right_*_sl.png'))
    #     lidar_paths = glob.glob(os.path.join(self.save_path, 'lidar_*_sl.pkl'))
    #     img_left_paths.sort()
    #     img_right_paths.sort()
    #     lidar_paths.sort()

    #     gathered_source_pts, gathered_target_pts = [], []
    #     print(f'  {len(img_left_paths)} pairs of stereo images and lidar data are found')
    #     T_l2c = None
    #     for i, img_left_path in tqdm(enumerate(img_left_paths)):
    #         img_left = cv2.imread(img_left_path)
    #         img_right = cv2.imread(img_right_paths[i])
    #         stereo_pcl, stereo_clr, _, _ = self.stereo_forward(img_left, img_right)
    #         # stereo_pcl, stereo_clr = self.voxelize(stereo_pcl, stereo_clr)
            
    #         lidar = self.read_pcl(lidar_paths[i])
    #         lidar_pcl = np.stack([lidar['x'], lidar['y'], lidar['z']], axis=-1).reshape(-1, 3)
    #         lidar_pcl = lidar_pcl[lidar_pcl[:, 0] > 0]  # only points in front of the lidar are used

    #         # perform icp
    #         # initial transformation between stereo camera and lidar 
    #         if T_l2c is None:
    #             T_l2c = np.array([[0, -1, 0, 0.25],
    #                             [0, 0, -1, -0.36],
    #                             [1, 0, 0, -0.63],
    #                             [0, 0, 0, 1]])
            
    #         # transform lidar points to stereo camera frame
    #         lidar_stereo_pcl = np.hstack([lidar_pcl, np.ones((lidar_pcl.shape[0], 1))])
    #         lidar_stereo_pcl = np.matmul(T_l2c, lidar_stereo_pcl.T).T[:, :3]

    #         p3d = np.concatenate([stereo_pcl, lidar_stereo_pcl], axis=0)
    #         clr = np.concatenate([stereo_clr, np.ones((lidar_stereo_pcl.shape[0], 3)) / 10], axis=0)
    #         pcd = [p3d, clr]
    #         self.pcdBuffer.pcd.put(pcd)
    #         if self.o3dvis is None:
    #             self.o3dvis = Process(target=o3dvis, args=(self.pcdBuffer,))
    #             self.o3dvis.start()
    #         # input("Before icp, press Enter to continue...")

    #         # T, target_pts, num_inliers = icp(
    #         #     lidar_pcl, stereo_pcl, init_pose=T_l2c,
    #         #     match_all=False, dist_thresh=1, max_iterations=20)
    #         T, target_pts, num_inliers = icp(
    #             stereo_pcl, lidar_pcl, init_pose=np.linalg.inv(T_l2c),
    #             match_all=False, dist_thresh=.1, max_iterations=100)
    #         T = np.linalg.inv(T)
    #         # T_l2c = T

    #         print('T: ', T)
            
    #         # transform lidar points to stereo camera frame
    #         lidar_stereo_pcl = np.hstack([lidar_pcl, np.ones((lidar_pcl.shape[0], 1))])
    #         lidar_stereo_pcl = np.matmul(T, lidar_stereo_pcl.T).T[:, :3]

    #         p3d = np.concatenate([stereo_pcl, lidar_stereo_pcl], axis=0)
    #         clr = np.concatenate([stereo_clr, np.ones((lidar_stereo_pcl.shape[0], 3)) / 10], axis=0)
    #         pcd = [p3d, clr]
    #         self.pcdBuffer.pcd.put(pcd)
    #         # input("After icp, press Enter to continue...")

    #         if num_inliers > 10:
    #             # gathered_source_pts.append(lidar_pcl)
    #             gathered_source_pts.append(stereo_pcl)
    #             gathered_target_pts.append(target_pts)

    #     gathered_source_pts = np.concatenate(gathered_source_pts, axis=0)
    #     gathered_target_pts = np.concatenate(gathered_target_pts, axis=0)

    #     # T, _, _ = best_fit_transform(gathered_source_pts, gathered_target_pts)
    #     # # T_l2c = T
    #     # T_l2c = np.linalg.inv(T)
    #     print('T_l2c: ', T_l2c)

    #     # save extrinsics
    #     with open(self.save_path + 'stereo_lidar_extrinsics.pkl', 'wb') as f:
    #         pickle.dump(T_l2c, f)

    #     print(' saved stereo lidar extrinsics...')

    def stereo_lidar_extrinsics(self):
        print(' first is between stereo camera and lidar...')
        # read data
        img_left_paths = glob.glob(os.path.join(self.save_path, 'cam_left_*_sl.png'))
        img_right_paths = glob.glob(os.path.join(self.save_path, 'cam_right_*_sl.png'))
        lidar_paths = glob.glob(os.path.join(self.save_path, 'lidar_*_sl.pkl'))
        img_left_paths.sort()
        img_right_paths.sort()
        lidar_paths.sort()

        print(f'  {len(img_left_paths)} pairs of stereo images and lidar data are found')
        stereo_pcls, stereo_clrs = [], []
        lidar_pcls = []
        for i, img_left_path in tqdm(enumerate(img_left_paths)):
            img_left = cv2.imread(img_left_path)
            img_right = cv2.imread(img_right_paths[i])
            stereo_pcl, stereo_clr, _, _ = self.stereo_forward(img_left, img_right)
            # stereo_pcl, stereo_clr = self.voxelize(stereo_pcl, stereo_clr)
            
            lidar = self.read_pcl(lidar_paths[i])
            lidar_pcl = np.stack([lidar['x'], lidar['y'], lidar['z']], axis=-1).reshape(-1, 3)
            lidar_pcl = lidar_pcl[lidar_pcl[:, 0] > 0]  # only points in front of the lidar are used
            
            stereo_pcls.append(stereo_pcl)
            stereo_clrs.append(stereo_clr)
            lidar_pcls.append(lidar_pcl)

        T_l2c = np.array([[0, -1, 0, 0.3],
                          [0, 0, -1, -0.36],
                          [1, 0, 0, -0.63],
                          [0, 0, 0, 1]])
        # T, _, _ = multi_icp(
        #     lidar_pcls, stereo_pcls, init_pose=T_l2c,
        #     match_all=False, dist_thresh=1, max_iterations=20)
        T, _, _ = multi_icp(
            stereo_pcls, lidar_pcls, init_pose=np.linalg.inv(T_l2c),
            match_all=False, dist_thresh=1, max_iterations=20)
        T = np.linalg.inv(T)
        T_l2c = T

        print('T_l2c: ', T_l2c)

        # save extrinsics
        with open(self.save_path + 'stereo_lidar_extrinsics.pkl', 'wb') as f:
            pickle.dump(T_l2c, f)

        print(' saved stereo lidar extrinsics...')

    def verify_stereo_lidar_extrinsics(self):
        # read data
        img_left_paths = glob.glob(os.path.join(self.save_path, 'cam_left_*_sl.png'))
        img_right_paths = glob.glob(os.path.join(self.save_path, 'cam_right_*_sl.png'))
        lidar_paths = glob.glob(os.path.join(self.save_path, 'lidar_*_sl.pkl'))
        img_left_paths.sort()
        img_right_paths.sort()
        lidar_paths.sort()

        # load saved extrinsics
        with open(self.save_path + 'stereo_lidar_extrinsics.pkl', 'rb') as f:
            T_l2c = pickle.load(f)
        # T_l2c = np.array([[0, -1, 0, 0.3],
        #                   [0, 0, -1, -0.36],
        #                   [1, 0, 0, -0.63],
        #                   [0, 0, 0, 1]])

        gathered_source_pts, gathered_target_pts = [], []
        print(f'  {len(img_left_paths)} pairs of stereo images and lidar data are found')
        for i, img_left_path in tqdm(enumerate(img_left_paths)):
            img_left = cv2.imread(img_left_path)
            img_right = cv2.imread(img_right_paths[i])
            stereo_pcl, stereo_clr, img_left_rect, P1 = self.stereo_forward(img_left, img_right, use_mask=True)
            # stereo_pcl, stereo_clr = self.voxelize(stereo_pcl, stereo_clr)
            
            lidar = self.read_pcl(lidar_paths[i])
            lidar_pcl = np.stack([lidar['x'], lidar['y'], lidar['z']], axis=-1).reshape(-1, 3)
            lidar_pcl = lidar_pcl[lidar_pcl[:, 0] > 0]  # only points in front of the lidar are used
            
            # transform lidar points to stereo camera frame
            lidar_stereo_pcl = np.hstack([lidar_pcl, np.ones((lidar_pcl.shape[0], 1))])
            lidar_stereo_pcl = np.matmul(T_l2c, lidar_stereo_pcl.T).T[:, :3]

            p3d = np.concatenate([stereo_pcl, lidar_stereo_pcl], axis=0)
            clr = np.concatenate([stereo_clr, np.ones((lidar_stereo_pcl.shape[0], 3)) / 10], axis=0)
            pcd = [p3d, clr]
            self.pcdBuffer.pcd.put(pcd)

            if self.o3dvis is None:
                self.o3dvis = Process(target=o3dvis, args=(self.pcdBuffer,))
                self.o3dvis.start()

            # project lidar points to stereo image
            lidar_stereo_proj = np.hstack([lidar_stereo_pcl, np.ones((lidar_stereo_pcl.shape[0], 1))])
            lidar_stereo_proj = np.matmul(P1, lidar_stereo_proj.T).T
            lidar_stereo_proj = lidar_stereo_proj[:, :2] / lidar_stereo_proj[:, 2:]
            # colorize lidar points based on depth
            lidar_stereo_clr = cv2.applyColorMap((10 * lidar_stereo_pcl[:, 2]).astype(np.uint8), cv2.COLORMAP_JET)[:, 0]

            mask = np.logical_and(lidar_stereo_proj[:, 0] > 0, lidar_stereo_proj[:, 0] < img_left.shape[1])
            mask = np.logical_and(mask, lidar_stereo_proj[:, 1] > 0)
            mask = np.logical_and(mask, lidar_stereo_proj[:, 1] < img_left.shape[0])
            lidar_stereo_proj = lidar_stereo_proj[mask]
            lidar_stereo_clr = lidar_stereo_clr[mask]
            
            lidar_img = np.zeros((img_left.shape[0], img_left.shape[1], 3), dtype=np.uint8)
            lidar_img[lidar_stereo_proj[:, 1].astype(int), lidar_stereo_proj[:, 0].astype(int)] = lidar_stereo_clr
            # superimpose lidar points on stereo image
            img_left_lidar = cv2.addWeighted(img_left_rect, 0.5, lidar_img, 1, 0)
            cv2.imwrite('imgs/img_left_proj_lidar.png', img_left_lidar)

            # pause until the user presses a key
            input("Press Enter to continue...")

    def lidar_radar_extrinsics(self):
        print(' second is between lidar and radar...')
        # read data
        lidar_paths = glob.glob(os.path.join(self.save_path, 'lidar_*_lr.pkl'))
        radar_paths = glob.glob(os.path.join(self.save_path, 'radar_*_lr.pkl'))
        lidar_paths.sort()
        radar_paths.sort()

        T_r2l = np.array([[1, 0, 0, 2.63],
                          [0, 1, 0, 0.],
                          [0, 0, 1, -1.36],
                          [0, 0, 0, 1]])
        print(f'  {len(lidar_paths)} pairs of lidar data and radar data are found')
        lidar_pcls, radar_pcls = [], []
        for i, lidar_path in tqdm(enumerate(lidar_paths)):
            lidar = self.read_pcl(lidar_path)
            lidar_pcl = np.stack([lidar['x'], lidar['y'], lidar['z']], axis=-1).reshape(-1, 3)
            lidar_pcl = lidar_pcl[lidar_pcl[:, 0] > 0]
            lidar_pcl = lidar_pcl[lidar_pcl[:, 0] < 100]

            radar = self.read_pcl(radar_paths[i])
            radar_pcl = np.stack([radar['x'], radar['y'], radar['z']], axis=-1).reshape(-1, 3)
            radar_pcl = radar_pcl[radar_pcl[:, 0] > 0]
            radar_pcl = radar_pcl[radar_pcl[:, 0] < 100]

            lidar_pcls.append(lidar_pcl)
            radar_pcls.append(radar_pcl)

            # transform radar points to lidar frame
            radar_lidar_pcl = np.hstack([radar_pcl, np.ones((radar_pcl.shape[0], 1))])
            radar_lidar_pcl = np.matmul(T_r2l, radar_lidar_pcl.T).T[:, :3]

            # get two different colors for lidar and radar points
            lidar_clr = np.zeros((lidar_pcl.shape[0], 3), dtype=np.uint8)
            radar_clr = cv2.applyColorMap((10 * np.ones((radar_lidar_pcl.shape[0], 1))).astype(np.uint8), cv2.COLORMAP_JET)[:, 0]
            p3d = np.concatenate([lidar_pcl, radar_lidar_pcl], axis=0)
            clr = np.concatenate([lidar_clr, radar_clr], axis=0)
            # p3d = radar_lidar_pcl
            # clr = radar_clr
            pcd = [p3d, clr]
            self.pcdBuffer.pcd.put(pcd)
            if self.o3dvis is None:
                self.o3dvis = Process(target=o3dvis, args=(self.pcdBuffer,))
                self.o3dvis.start()
            # input("Before icp, press Enter to continue...")

        T, target_pts, num_inliers = multi_icp(
            lidar_pcls, radar_pcls, init_pose=T_r2l,
            match_all=False, dist_thresh=1, max_iterations=20)
        T = np.linalg.inv(T)
        T_r2l = T

        print('T_r2l: ', T_r2l)

        # save extrinsics
        with open(self.save_path + 'lidar_radar_extrinsics.pkl', 'wb') as f:
            pickle.dump(T_r2l, f)

        print(' saved lidar radar extrinsics...')

    def verify_lidar_radar_extrinsics(self):
        # read data
        lidar_paths = glob.glob(os.path.join(self.save_path, 'lidar_*_lr.pkl'))
        radar_paths = glob.glob(os.path.join(self.save_path, 'radar_*_lr.pkl'))
        lidar_paths.sort()
        radar_paths.sort()

        # load saved extrinsics
        # with open(self.save_path + 'lidar_radar_extrinsics.pkl', 'rb') as f:
        #     T_r2l = pickle.load(f)
        T_r2l = np.array([[1, 0, 0, 2.63],
                          [0, 1, 0, 0.],
                          [0, 0, 1, -1.36],
                          [0, 0, 0, 1]])

        gathered_source_pts, gathered_target_pts = [], []
        print(f'  {len(lidar_paths)} pairs of lidar data and radar data are found')
        for i, lidar_path in tqdm(enumerate(lidar_paths)):
            lidar = self.read_pcl(lidar_path)
            lidar_pcl = np.stack([lidar['x'], lidar['y'], lidar['z']], axis=-1).reshape(-1, 3)
            lidar_pcl = lidar_pcl[lidar_pcl[:, 0] > 0]
            lidar_pcl = lidar_pcl[lidar_pcl[:, 0] < 100]

            radar = self.read_pcl(radar_paths[i])
            radar_pcl = np.stack([radar['x'], radar['y'], radar['z']], axis=-1).reshape(-1, 3)
            radar_pcl = radar_pcl[radar_pcl[:, 0] > 0]
            radar_pcl = radar_pcl[radar_pcl[:, 0] < 100]

            # transform radar points to lidar frame
            radar_lidar_pcl = np.hstack([radar_pcl, np.ones((radar_pcl.shape[0], 1))])
            radar_lidar_pcl = np.matmul(T_r2l, radar_lidar_pcl.T).T[:, :3]

            # get two different colors for lidar and radar points
            lidar_clr = np.zeros((lidar_pcl.shape[0], 3), dtype=np.uint8)
            radar_clr = cv2.applyColorMap((10 * np.ones((radar_lidar_pcl.shape[0], 1))).astype(np.uint8), cv2.COLORMAP_JET)[:, 0]
            p3d = np.concatenate([lidar_pcl, radar_lidar_pcl], axis=0)
            clr = np.concatenate([lidar_clr, radar_clr], axis=0)
            # p3d = radar_lidar_pcl
            # clr = radar_clr
            pcd = [p3d, clr]
            self.pcdBuffer.pcd.put(pcd)

            if self.o3dvis is None:
                self.o3dvis = Process(target=o3dvis, args=(self.pcdBuffer,))
                self.o3dvis.start()

            # pause until the user presses a key
            input("Press Enter to continue...")

    def verify_stereo_radar_extrinsics(self):
        # read data
        img_left_paths = glob.glob(os.path.join(self.save_path, 'cam_left_*_sr.png'))
        img_right_paths = glob.glob(os.path.join(self.save_path, 'cam_right_*_sr.png'))
        radar_paths = glob.glob(os.path.join(self.save_path, 'radar_*_sr.pkl'))
        img_left_paths.sort()
        img_right_paths.sort()
        radar_paths.sort()

        # load saved extrinsics
        with open(self.save_path + 'stereo_lidar_extrinsics.pkl', 'rb') as f:
            T_l2c = pickle.load(f)
        # with open(self.save_path + 'lidar_radar_extrinsics.pkl', 'rb') as f:
        #     T_r2l = pickle.load(f)
        T_r2l = np.array([[1, 0, 0, 2.63],
                          [0, 1, 0, 0.],
                          [0, 0, 1, -1.36],
                          [0, 0, 0, 1]])
        # compute extrinsics between stereo camera and radar
        T_r2c = np.matmul(T_l2c, T_r2l)

        gathered_source_pts, gathered_target_pts = [], []
        print(f'  {len(img_left_paths)} pairs of stereo images and radar data are found')
        for i, img_left_path in tqdm(enumerate(img_left_paths)):
            img_left = cv2.imread(img_left_path)
            img_right = cv2.imread(img_right_paths[i])
            stereo_pcl, stereo_clr, img_left_rect, P1 = self.stereo_forward(img_left, img_right, use_mask=True)
            # stereo_pcl, stereo_clr = self.voxelize(stereo_pcl, stereo_clr)
            
            radar = self.read_pcl(radar_paths[i])
            radar_pcl = np.stack([radar['x'], radar['y'], radar['z']], axis=-1).reshape(-1, 3)
            radar_pcl = radar_pcl[radar_pcl[:, 0] > 0]
            radar_pcl = radar_pcl[radar_pcl[:, 0] < 100]

            # transform radar points to stereo camera frame
            radar_stereo_pcl = np.hstack([radar_pcl, np.ones((radar_pcl.shape[0], 1))])
            radar_stereo_pcl = np.matmul(T_r2c, radar_stereo_pcl.T).T[:, :3]

            p3d = np.concatenate([stereo_pcl, radar_stereo_pcl], axis=0)
            clr = np.concatenate([stereo_clr, np.ones((radar_stereo_pcl.shape[0], 3)) / 10], axis=0)
            pcd = [p3d, clr]
            self.pcdBuffer.pcd.put(pcd)

            if self.o3dvis is None:
                self.o3dvis = Process(target=o3dvis, args=(self.pcdBuffer,))
                self.o3dvis.start()

            # Project radar points to stereo image
            radar_stereo_proj = np.hstack([radar_stereo_pcl, np.ones((radar_stereo_pcl.shape[0], 1))])
            radar_stereo_proj = np.matmul(P1, radar_stereo_proj.T).T
            radar_stereo_proj = radar_stereo_proj[:, :2] / radar_stereo_proj[:, 2:]
            # colorize radar points based on depth
            radar_stereo_clr = cv2.applyColorMap((10 * radar_stereo_pcl[:, 2]).astype(np.uint8), cv2.COLORMAP_JET)[:, 0]

            mask = np.logical_and(radar_stereo_proj[:, 0] > 0, radar_stereo_proj[:, 0] < img_left.shape[1])
            mask = np.logical_and(mask, radar_stereo_proj[:, 1] > 0)
            mask = np.logical_and(mask, radar_stereo_proj[:, 1] < img_left.shape[0])
            radar_stereo_proj = radar_stereo_proj[mask]
            radar_stereo_clr = radar_stereo_clr[mask]

            radar_img = np.zeros((img_left.shape[0], img_left.shape[1], 3), dtype=np.uint8)
            radar_img[radar_stereo_proj[:, 1].astype(int), radar_stereo_proj[:, 0].astype(int)] = radar_stereo_clr
            # superimpose radar points on stereo image
            img_left_radar = cv2.addWeighted(img_left_rect, 0.5, radar_img, 1, 0)
            cv2.imwrite('imgs/img_left_proj_radar.png', img_left_radar)

            # pause until the user presses a key
            input("Press Enter to continue...")

    def stereo_rectify(self, img_left, img_right):
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
        map_right1, map_right2 = cv2.initUndistortRectifyMap(
            mtx_right, dist_right, R2, P2, img_shape, cv2.CV_16SC2)

        img_left_rectified = cv2.remap(img_left, map_left1, map_left2, cv2.INTER_LINEAR)
        img_right_rectified = cv2.remap(img_right, map_right1, map_right2, cv2.INTER_LINEAR)

        return img_left_rectified, img_right_rectified, R1, R2, P1, P2
    
    def stereo_forward(self, img_left, img_right, use_mask=True):
        unrectified = np.concatenate([img_left, img_right], axis=1)
        for i in range(0, unrectified.shape[0], 50):
            cv2.line(unrectified, (0, i), (unrectified.shape[1], i), (0, 255, 0), 1)
        cv2.imwrite('imgs/unrectified.png', unrectified)

        img_left_rect, img_right_rect, R1, R2, P1, P2 = self.stereo_rectify(img_left, img_right)
        rectified = np.concatenate([img_left_rect, img_right_rect], axis=1)
        for i in range(0, rectified.shape[0], 50):
            cv2.line(rectified, (0, i), (rectified.shape[1], i), (0, 255, 0), 1)
        cv2.imwrite('imgs/rectified.png', rectified)

        # XYZ, color = self.stereo_dense_forward(img_left_rect, img_right_rect, P1, P2, use_mask=use_mask)
        XYZ, color = self.stereo_sparse_forward(img_left_rect, img_right_rect, P1, P2)

        return XYZ, color, img_left_rect, P1

    def stereo_sparse_forward(self, img_left_rect, img_right_rect, P1, P2):
        baseline = -P2[0, 3] / P2[0, 0]
        focal, cx, cy = P1[0, 0], P1[0, 2], P1[1, 2]
        H, W = img_left_rect.shape[:2]

        # ############################
        # # find the keypoints and descriptors with SIFT
        # kp1, des1 = self.sift.detectAndCompute(img_left_rect, None)
        # kp2, des2 = self.sift.detectAndCompute(img_right_rect, None)

        # # Match descriptors.
        # matches = self.bf.knnMatch(des1, des2, k=2)

        # # Select good matches.
        # good = []
        # for m, n in matches:
        #     if m.distance < 0.75 * n.distance:
        #         good.append([m])

        # src_pts = np.float32([kp1[m[0].queryIdx].pt for m in good]).reshape(-1, 2)
        # dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in good]).reshape(-1, 2)

        ###########################
        img1 = torch.tensor(img_left_rect/255).permute(2, 0, 1).unsqueeze(0).float().cuda()
        img2 = torch.tensor(img_right_rect/255).permute(2, 0, 1).unsqueeze(0).float().cuda()
        # scale to loftr input size
        img1 = kornia.geometry.resize(img1, (480, 640), antialias=True)
        img2 = kornia.geometry.resize(img2, (480, 640), antialias=True)
        # # scale camera parameters
        # focal = focal * 640 / img_left_rect.shape[1]
        # cx = cx * 640 / img_left_rect.shape[1]
        # cy = cy * 480 / img_left_rect.shape[0]
        # baseline = baseline * 640 / img_left_rect.shape[1]

        input_dict = {
            "image0": kornia.color.rgb_to_grayscale(img1),
            "image1": kornia.color.rgb_to_grayscale(img2),
        }
        with torch.no_grad():
            loftr_correspondences = self.loftr(input_dict)
        src_pts = loftr_correspondences['keypoints0'].cpu().numpy()
        dst_pts = loftr_correspondences['keypoints1'].cpu().numpy()
        # scale the points back to original size
        x_ratio, y_ratio = W / 640, H / 480
        src_pts[:, 0] = (src_pts[:, 0] * x_ratio) + (x_ratio - 1) / 2
        src_pts[:, 1] = (src_pts[:, 1] * y_ratio) + (y_ratio - 1) / 2
        dst_pts[:, 0] = (dst_pts[:, 0] * x_ratio) + (x_ratio - 1) / 2
        dst_pts[:, 1] = (dst_pts[:, 1] * y_ratio) + (y_ratio - 1) / 2
        # repeat in reverse direction
        input_dict = {
            "image1": kornia.color.rgb_to_grayscale(img1),
            "image0": kornia.color.rgb_to_grayscale(img2),
        }
        with torch.no_grad():
            loftr_correspondences = self.loftr(input_dict)
        src_pts_reverse = loftr_correspondences['keypoints1'].cpu().numpy()
        dst_pts_reverse = loftr_correspondences['keypoints0'].cpu().numpy()
        # scale the points back to original size
        src_pts_reverse[:, 0] = (src_pts_reverse[:, 0] * x_ratio) + (x_ratio - 1) / 2
        src_pts_reverse[:, 1] = (src_pts_reverse[:, 1] * y_ratio) + (y_ratio - 1) / 2
        dst_pts_reverse[:, 0] = (dst_pts_reverse[:, 0] * x_ratio) + (x_ratio - 1) / 2
        dst_pts_reverse[:, 1] = (dst_pts_reverse[:, 1] * y_ratio) + (y_ratio - 1) / 2
        src_pts = np.concatenate([src_pts, src_pts_reverse], axis=0)
        dst_pts = np.concatenate([dst_pts, dst_pts_reverse], axis=0)

        
        disp_np = dst_pts - src_pts
        # filter out points with large vertical disparity and positive horizontal disparity
        mask = np.logical_and(np.abs(disp_np[:, 1]) < 1, disp_np[:, 0] < 0)
        src_pts = src_pts[mask]
        dst_pts = dst_pts[mask]
        disp_np = disp_np[mask][:, 0]
        
        # Draw matches with different colors.
        colors = np.random.randint(0, 255, (src_pts.shape[0], 3))
        img_match = np.concatenate([img_left_rect, img_right_rect], 1)
        img_match = np.ascontiguousarray((255*img_match).astype(np.uint8))
        for j in range(src_pts.shape[0]):
            cv2.circle(img_match, tuple(src_pts[j].astype(int)), 1, colors[j].tolist(), 1)
            cv2.circle(img_match, tuple(dst_pts[j].astype(int) + np.array([W, 0])), 1, colors[j].tolist(), 1)
            cv2.line(img_match, tuple(src_pts[j].astype(int)), tuple(dst_pts[j].astype(int) + np.array([W, 0])), colors[j].tolist(), 1)
        cv2.imwrite('imgs/matches.png', img_match)

        # compute point cloud
        # unproject disparity map to point cloud
        Z = focal * baseline / np.clip(-disp_np, 1e-3, None)
        X = (src_pts[:, 0] - cx) * Z / focal
        Y = (src_pts[:, 1] - cy) * Z / focal
        XYZ = np.stack([X, Y, Z], axis=-1)
        XYZ = XYZ.reshape(-1, 3)

        color = img_left_rect[src_pts[:, 1].astype(int), src_pts[:, 0].astype(int)]
        color = color[:, ::-1] / 255

        mask = XYZ[:, 2] > 0
        # mask = np.logical_and(mask, XYZ[:, 1] > -1)
        mask = np.logical_and(mask, XYZ[:, 2] < 100)
        # ############################
        # # also filter out points that are in the ground plane ??
        # mask = np.logical_and(mask, XYZ[:, 1] < 1)
        # ############################
        # # for validation, only include points in the middle
        # mask = np.logical_and(mask, XYZ[:, 0] > -1)
        # mask = np.logical_and(mask, XYZ[:, 0] < 1)
        # ############################
        # # mask out points around image border
        # mask = np.logical_and(mask, src_pts[:, 0] > 10)
        # mask = np.logical_and(mask, src_pts[:, 0] < img_left_rect.shape[1] - 10)
        # mask = np.logical_and(mask, src_pts[:, 1] > 10)
        # mask = np.logical_and(mask, src_pts[:, 1] < img_left_rect.shape[0] - 10)
        ############################

        XYZ = XYZ[mask]
        color = color[mask]

        return XYZ, color

    def voxelize(self, pcd, clr, voxel_size=0.01, min_points=0):
        # voxelize the point cloud
        pcl_round = np.round(pcd / voxel_size) * voxel_size
        # use np.unique to get voxels containing more than min_points and their indices
        # so that we can get the corresponding colors
        pcl_round, indices, counts = np.unique(pcl_round, axis=0, return_index=True, return_counts=True)
        pcl = pcl_round[counts > min_points]
        clr = clr[indices[counts > min_points]]
        return pcl, clr


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data/mscrad4r_pkl/')
# accept multiple data names
parser.add_argument('--data_name', type=str, nargs='+', default=['230402_radar_cam_calib'])
parser.add_argument('--save_dir', type=str, default='./data/mscrad4r_pkl/extrinsic_calib/')
parser.add_argument('--stereo_calib', type=str, default='./data/mscrad4r_pkl/stereo_calib/230402_cam_calib/stereo_calib.pkl')
args = parser.parse_args()


if __name__ == '__main__':
    data_paths = [os.path.join(args.data_dir, data_name) for data_name in args.data_name]
    save_path = os.path.join(args.save_dir, 'data')
    stereo_calib_path = args.stereo_calib
    radarlidarstereo = RadarLidarStereo(data_paths, save_path, stereo_calib_path)
    # radarlidarstereo.read()
    # radarlidarstereo.compute_extrinsics()
    radarlidarstereo.verify_extrinsics()
