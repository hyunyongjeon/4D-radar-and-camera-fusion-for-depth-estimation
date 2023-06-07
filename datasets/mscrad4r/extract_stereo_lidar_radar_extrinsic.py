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

from .icp import icp, best_fit_transform, nearest_neighbor
from .vis3d import o3dvis, pcdBuffer

import torch
from torchvision import transforms
from .coex.models.Stereo import Stereo
os.environ["CUDA_VISIBLE_DEVICES"]="0"

torch.set_grad_enabled(False)

import pdb


config = 'cfg_coextr.yaml'
version = 18  # 
train_data = 'KITTI-KITTI'

half_precision = True

def load_configs(current_path, path):
    cfg = YAML().load(open(current_path + '/' + path, 'r'))
    backbone_cfg = YAML().load(
        open(current_path + '/coex/' + cfg['model']['stereo']['backbone']['cfg_path'], 'r'))
    cfg['model']['stereo']['backbone'].update(backbone_cfg)
    return cfg


class RadarLidarStereo:
    def __init__(self, data_path, save_path, stereo_calib_path):
        self.data_path = data_path
        self.save_path = save_path

        # read stereo calibration results
        with open(stereo_calib_path, 'rb') as f:
            self.stereo_calib = pickle.load(f)
        
        # load coex model
        current_path = os.path.dirname(os.path.realpath(__file__))
        coex_cfg = load_configs(current_path, 'coex/configs/stereo/models_cfg/{}'.format(config))
        model_name = coex_cfg['model']['stereo']['name']
        print(f"\x1b[1;30;103m model: {model_name}\x1b[0m")

        coex_ckpt = '{}/coex/{}/{}/{}/version_{}/checkpoints/last.ckpt'.format(
            current_path, 'logs/stereo', model_name, train_data, version)
        coex_cfg['stereo_ckpt'] = coex_ckpt
        coex_cfg['divide_batch'] = 1
        self.coex = Stereo.load_from_checkpoint(coex_ckpt, strict=False, cfg=coex_cfg).cuda().eval()
        self.to_tensor = transforms.ToTensor()
        
        self.pcdBuffer = pcdBuffer()
        self.o3dvis = None

    def read(self):
        # read stereo camera pickle files
        with open(self.data_path + 'cam_left.pkl', 'rb') as f:
            cam_left_msgs = pickle.load(f)
        with open(self.data_path + 'cam_right.pkl', 'rb') as f:
            cam_right_msgs = pickle.load(f)
        with open(self.data_path + 'lidar.pkl', 'rb') as f:
            lidar_msgs = pickle.load(f)
        with open(self.data_path + 'radar.pkl', 'rb') as f:
            radar_msgs = pickle.load(f)

        # extract synced data and save them
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
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
            if timegap_right < 0.001 and timegap_lidar < 0.005:
                print(f'timegap_right: {timegap_right}, timegap_lidar: {timegap_lidar}')
                # save stereo images
                cv2.imwrite(self.save_path + 'cam_left_' + str(i) + '_sl.png', cam_left_msgs['msg'][i])
                cv2.imwrite(self.save_path + 'cam_right_' + str(i) + '_sl.png', cam_right_msgs['msg'][cam_right_msg_id])
                # save lidar data
                with open(self.save_path + 'lidar_' + str(i) + '_sl.pkl', 'wb') as f:
                    pickle.dump(lidar_msgs['msg'][lidar_msg_id], f)

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
                with open(self.save_path + 'radar_' + str(i) + '_lr.pkl', 'wb') as f:
                    pickle.dump(radar_msgs['msg'][radar_msg_id], f)

        # then stereo camera and radar
        for i in range(len(cam_left_msgs_time)):
            cam_left_msg_time = cam_left_msgs_time[i]

            # find the closest msgs in cam_right_msgs_time, radar_msgs_time
            cam_right_msg_id = np.argmin(np.abs(cam_right_msgs_time - cam_left_msg_time))
            radar_msg_id = np.argmin(np.abs(radar_msgs_time - cam_left_msg_time))

            # check if the time difference is within 1ms
            timegap_right = np.abs(cam_left_msg_time - cam_right_msgs_time[cam_right_msg_id]) / 1e9
            timegap_radar = np.abs(cam_left_msg_time - radar_msgs_time[radar_msg_id]) / 1e9
            if timegap_right < 0.001 and timegap_radar < 0.005:
                print(f'timegap_right: {timegap_right}, timegap_radar: {timegap_radar}')
                # save stereo images
                cv2.imwrite(self.save_path + 'cam_left_' + str(i) + '_sr.png', cam_left_msgs['msg'][i])
                cv2.imwrite(self.save_path + 'cam_right_' + str(i) + '_sr.png', cam_right_msgs['msg'][cam_right_msg_id])
                # save radar data
                with open(self.save_path + 'radar_' + str(i) + '_sr.pkl', 'wb') as f:
                    pickle.dump(radar_msgs['msg'][radar_msg_id], f)

        # then all three
        for i in range(len(cam_left_msgs_time)):
            cam_left_msg_time = cam_left_msgs_time[i]

            # find the closest msgs in cam_right_msgs_time, lidar_msgs_time, radar_msgs_time
            cam_right_msg_id = np.argmin(np.abs(cam_right_msgs_time - cam_left_msg_time))
            lidar_msg_id = np.argmin(np.abs(lidar_msgs_time - cam_left_msg_time))
            radar_msg_id = np.argmin(np.abs(radar_msgs_time - cam_left_msg_time))

            # check if the time difference is within 1ms
            timegap_right = np.abs(cam_left_msg_time - cam_right_msgs_time[cam_right_msg_id]) / 1e9
            timegap_lidar = np.abs(cam_left_msg_time - lidar_msgs_time[lidar_msg_id]) / 1e9
            timegap_radar = np.abs(cam_left_msg_time - radar_msgs_time[radar_msg_id]) / 1e9
            if timegap_right < 0.001 and timegap_lidar < 0.01 and timegap_radar < 0.005:
                print(f'timegap_right: {timegap_right}, timegap_lidar: {timegap_lidar}, timegap_radar: {timegap_radar}')
                # save stereo images
                cv2.imwrite(self.save_path + 'cam_left_' + str(i) + '_slr.png', cam_left_msgs['msg'][i])
                cv2.imwrite(self.save_path + 'cam_right_' + str(i) + '_slr.png', cam_right_msgs['msg'][cam_right_msg_id])
                # save lidar data
                with open(self.save_path + 'lidar_' + str(i) + '_slr.pkl', 'wb') as f:
                    pickle.dump(lidar_msgs['msg'][lidar_msg_id], f)
                # save radar data
                with open(self.save_path + 'radar_' + str(i) + '_slr.pkl', 'wb') as f:
                    pickle.dump(radar_msgs['msg'][radar_msg_id], f)

    def read_pcl(self, pcl_path):
        with open(pcl_path, 'rb') as f:
            pcl = pickle.load(f)
        return pcl

    def compute_extrinsics(self):
        print('computing extrinsics between stereo camera, lidar, and radar...')
        self.stereo_lidar_extrinsics()

    def verify_extrinsics(self):
        print('verifying extrinsics between stereo camera, lidar, and radar...')
        self.verify_stereo_lidar_extrinsics()

    def stereo_lidar_extrinsics(self):
        print(' first is between stereo camera and lidar...')
        # read data
        img_left_paths = glob.glob(self.save_path + 'cam_left_*_sl.png')
        img_right_paths = glob.glob(self.save_path + 'cam_right_*_sl.png')
        lidar_paths = glob.glob(self.save_path + 'lidar_*_sl.pkl')
        img_left_paths.sort()
        img_right_paths.sort()
        lidar_paths.sort()

        gathered_source_pts, gathered_target_pts = [], []
        print(f'  {len(img_left_paths)} pairs of stereo images and lidar data are found')
        for i, img_left_path in tqdm(enumerate(img_left_paths)):
            img_left = cv2.imread(img_left_path)
            img_right = cv2.imread(img_right_paths[i])
            stereo_pcl, stereo_clr, _, _ = self.stereo_forward(img_left, img_right)
            
            lidar = self.read_pcl(lidar_paths[i])
            lidar_pcl = np.stack([lidar['x'], lidar['y'], lidar['z']], axis=-1).reshape(-1, 3)
            lidar_pcl = lidar_pcl[lidar_pcl[:, 0] > 0]  # only points in front of the lidar are used

            # perform icp
            # initial transformation between stereo camera and lidar 
            T_l2c = np.array([[0, -1, 0, 0],
                              [0, 0, -1, -0.36],
                              [1, 0, 0, -0.63],
                              [0, 0, 0, 1]])
            
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

            # T, target_pts, num_inliers = icp(
            #     lidar_pcl, stereo_pcl, init_pose=T_l2c,
            #     match_all=False, dist_thresh=1, max_iterations=20)
            # T_l2c = T
            T, target_pts, num_inliers = icp(
                stereo_pcl, lidar_pcl, init_pose=np.linalg.inv(T_l2c),
                match_all=False, dist_thresh=1, max_iterations=20)
            T_l2c = np.linalg.inv(T)

            if num_inliers > 1000:
                # gathered_source_pts.append(lidar_pcl)
                gathered_source_pts.append(stereo_pcl)
                gathered_target_pts.append(target_pts)

        gathered_source_pts = np.concatenate(gathered_source_pts, axis=0)
        gathered_target_pts = np.concatenate(gathered_target_pts, axis=0)

        T, _, _ = best_fit_transform(gathered_source_pts, gathered_target_pts)
        T_l2c = np.linalg.inv(T)
        print('T_l2c: ', T_l2c)

        # save extrinsics
        with open(self.save_path + 'stereo_lidar_extrinsics.pkl', 'wb') as f:
            pickle.dump(T_l2c, f)

        print(' saved stereo lidar extrinsics...')

    def verify_stereo_lidar_extrinsics(self):
        # read data
        img_left_paths = glob.glob(self.save_path + 'cam_left_*_sl.png')
        img_right_paths = glob.glob(self.save_path + 'cam_right_*_sl.png')
        lidar_paths = glob.glob(self.save_path + 'lidar_*_sl.pkl')
        img_left_paths.sort()
        img_right_paths.sort()
        lidar_paths.sort()

        # load saved extrinsics
        with open(self.save_path + 'stereo_lidar_extrinsics.pkl', 'rb') as f:
            T_l2c = pickle.load(f)

        gathered_source_pts, gathered_target_pts = [], []
        print(f'  {len(img_left_paths)} pairs of stereo images and lidar data are found')
        for i, img_left_path in tqdm(enumerate(img_left_paths)):
            img_left = cv2.imread(img_left_path)
            img_right = cv2.imread(img_right_paths[i])
            stereo_pcl, stereo_clr, left_img, P1 = self.stereo_forward(img_left, img_right, use_mask=False)
            
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
            img_left = cv2.addWeighted(img_left, 0.5, lidar_img, 1, 0)
            cv2.imwrite('imgs/img_left_proj_lidar.png', img_left)

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

        imgL, imgR = self.to_tensor(img_left_rect)[None].cuda(), self.to_tensor(img_right_rect)[None].cuda()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=half_precision):
                disp_pred = self.coex(imgL, imgR, training=False)
                disp_np = disp_pred[0].cpu().numpy()

        # warp right image to left view using gridsample to validate disparity map
        H, W = disp_pred.shape[-2:]
        scale = torch.tensor([W, H], dtype=torch.float32, device=disp_pred.device)[None, None, :]
        gridy, gridx = torch.meshgrid(
            torch.arange(0, H, device=disp_pred.device),
            torch.arange(0, W, device=disp_pred.device))
        grid = torch.stack([gridx-disp_pred[0], gridy], dim=-1)
        grid = ((grid + 0.5) / scale) * 2 - 1
        grid = grid.clamp(min=-2, max=2)
        warpedR = torch.nn.functional.grid_sample(imgR, grid[None],
                              mode='bilinear', padding_mode='zeros', align_corners=False)
        warped_right = warpedR[0].cpu().numpy().transpose(1, 2, 0)
        warped_right = (255*np.clip(warped_right, 0, 1))
        cv2.imwrite('imgs/warped_right.png', warped_right.astype(np.uint8))
        cv2.imwrite('imgs/left_warped_right.png', ((warped_right + img_left_rect.astype(float)) / 2).astype(np.uint8))
        cv2.imwrite('imgs/left_right.png', ((img_right_rect.astype(float) + img_left_rect.astype(float)) / 2).astype(np.uint8))
        
        # save disparity map for visualization
        disp_vis = disp_np / 192.0 * 4
        disp_vis = cv2.applyColorMap((disp_vis * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)
        cv2.imwrite('imgs/disp_vis.png', disp_vis)

        # compute point cloud
        # unproject disparity map to point cloud
        baseline = -P2[0, 3] / P2[0, 0]
        focal, cx, cy = P1[0, 0], P1[0, 2], P1[1, 2]
        x, y = np.meshgrid(np.arange(disp_np.shape[1]), np.arange(disp_np.shape[0]))
        Z = focal * baseline / np.clip(disp_np, 1e-3, None)
        X = (x - cx) * Z / focal
        Y = (y - cy) * Z / focal
        XYZ = np.stack([X, Y, Z], axis=-1)
        XYZ = XYZ.reshape(-1, 3)
        
        mask = np.logical_and(XYZ[:, 1] > -1, XYZ[:, 2] > 0)
        mask = np.logical_and(mask, XYZ[:, 2] < 10)
        # # also filter out points that are in the ground plane ??
        # mask = np.logical_and(mask, XYZ[:, 1] < 1)
        # mask out points around image border
        mask = np.logical_and(mask, x.reshape(-1) > 10)
        mask = np.logical_and(mask, x.reshape(-1) < disp_np.shape[1] - 10)
        mask = np.logical_and(mask, y.reshape(-1) > 10)
        mask = np.logical_and(mask, y.reshape(-1) < disp_np.shape[0] - 10)
        # also filter out points around depth discontinuity
        # first compute depth gradient using sobel filter
        # gaussian blur to remove noise
        depth = XYZ[:, 2].reshape(disp_np.shape)
        depth = cv2.GaussianBlur(depth, (3, 3), 0)
        depth_grad = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
        depth_grad = np.abs(depth_grad)
        # mask out points with high gradient
        depth_grad_flat = depth_grad.reshape(-1)
        mask = np.logical_and(mask, depth_grad_flat < 0.5)

        # display disp map with the high gradient masked out
        depth_grad = depth_grad < 0.5
        disp_vis = disp_vis * depth_grad[..., None]
        cv2.imwrite('imgs/disp_vis_masked.png', disp_vis)

        color = img_left_rect.reshape(-1, 3) / 255
        if use_mask:
            XYZ = XYZ[mask]
            color = color[mask]
        return XYZ, color, img_left_rect, P1


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data/mscrad4r_pkl/')
parser.add_argument('--data_name', type=str, default='230402_radar_cam_calib')
parser.add_argument('--save_dir', type=str, default='./data/mscrad4r_pkl/extrinsic_calib/')
parser.add_argument('--stereo_calib', type=str, default='./data/mscrad4r_pkl/stereo_calib/230402_cam_calib/stereo_calib.pkl')
args = parser.parse_args()


if __name__ == '__main__':
    data_path = args.data_dir + args.data_name + '/'
    save_path = args.save_dir + args.data_name + '/'
    stereo_calib_path = args.stereo_calib
    radarlidarstereo = RadarLidarStereo(data_path, save_path, stereo_calib_path)
    # radarlidarstereo.read()
    # radarlidarstereo.compute_extrinsics()
    radarlidarstereo.verify_extrinsics()
