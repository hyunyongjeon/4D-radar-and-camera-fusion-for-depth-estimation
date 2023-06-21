# python -m datasets.mscrad4r.extract_stereo_calib 

import pickle
import numpy as np
import cv2
import glob
import argparse
import os
from pathlib import Path

import pdb


class Stereo:
    def __init__(self, data_path, save_path):
        self.data_path = data_path
        self.save_path = save_path

    def read(self):
        # read stereo camera pickle files
        with open(os.path.join(self.data_path, 'cam_left.pkl'), 'rb') as f:
            cam_left_msgs = pickle.load(f)
        with open(os.path.join(self.data_path, 'cam_right.pkl'), 'rb') as f:
            cam_right_msgs = pickle.load(f)

        # extract synced stereo images and save them as png files
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        for i in range(len(cam_left_msgs['time'])):
            timegap = np.abs(cam_left_msgs['time'][i] - cam_right_msgs['time'][i]) / 1e9
            
            if timegap < 0.001: #
                cv2.imwrite(os.path.join(self.save_path, 'cam_left_' + str(i) + '.png'), cam_left_msgs['msg'][i])
                cv2.imwrite(os.path.join(self.save_path, 'cam_right_' + str(i) + '.png'), cam_right_msgs['msg'][i])
    
    def calib(self):
        print('calibrating stereo camera...')
        # read stereo images
        img_left_paths = glob.glob(os.path.join(self.save_path, 'cam_left_*.png'))
        img_right_paths = glob.glob(os.path.join(self.save_path, 'cam_right_*.png'))
        img_left_paths.sort()
        img_right_paths.sort()

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        print(" find corners in chessboard images...")
        chessboard_save_path = Path(os.path.join(self.save_path, 'chessboard/'))
        chessboard_save_path.mkdir(parents=True, exist_ok=True)
        for i, img_left_path in enumerate(img_left_paths):
            img_left = cv2.imread(img_left_path)
            img_right = cv2.imread(img_right_paths[i])
            img_left_gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            img_right_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

            # stereo calibration
            if i == 0:
                img_shape = img_left_gray.shape[::-1]
                objp = np.zeros((7*10, 3), np.float32)
                objp[:, :2] = np.mgrid[0:10, 0:7].T.reshape(-1, 2) * 0.1
                objpoints = []
                imgpoints_left = []
                imgpoints_right = []

            ret_left, corners_left = cv2.findChessboardCorners(img_left_gray, (10, 7), None)
            ret_right, corners_right = cv2.findChessboardCorners(img_right_gray, (10, 7), None)

            if ret_left and ret_right:
                objpoints.append(objp)
                corners_left = cv2.cornerSubPix(img_left_gray, corners_left, (11, 11), (-1, -1), criteria)
                corners_right = cv2.cornerSubPix(img_right_gray, corners_right, (11, 11), (-1, -1), criteria)
                imgpoints_left.append(corners_left)
                imgpoints_right.append(corners_right)

                # draw and display the corners
                img_left = cv2.drawChessboardCorners(img_left, (10, 7), corners_left, ret_left)
                img_right = cv2.drawChessboardCorners(img_right, (10, 7), corners_right, ret_right)
                cv2.imwrite(str(chessboard_save_path / f'chessboard_left_{i}.png'), img_left)
                cv2.imwrite(str(chessboard_save_path / f'chessboard_right_{i}.png'), img_right)


        print(" stereo calibration...")
        # calibrate each camera
        ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
            objpoints, imgpoints_left, img_shape, None, None)
        ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
            objpoints, imgpoints_right, img_shape, None, None)

        # stereo calibration
        flags = 0
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        # flags |= cv2.CALIB_FIX_INTRINSIC
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        # flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        # flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5
        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        ret_stereo, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints_left, imgpoints_right, mtx_left, dist_left, mtx_right, dist_right, img_shape,
            criteria=stereocalib_criteria, flags=flags)

        # save calibration results as pickle files for later use
        stereo_calib = {'mtx_left': mtx_left, 'dist_left': dist_left, 'mtx_right': mtx_right, 'dist_right': dist_right, 'R': R, 'T': T, 'E': E, 'F': F}
        with open(os.path.join(self.save_path, 'stereo_calib.pkl'), 'wb') as f:
            pickle.dump(stereo_calib, f)
        print(' stereo calibration results saved as pickle files')

    def rectify(self):
        print('performing stereo rectification to validate...')
        rectify_save_path = Path(self.save_path + 'rectified/')
        rectify_save_path.mkdir(parents=True, exist_ok=True)
        # read stereo images
        img_left_paths = glob.glob(os.path.join(self.save_path, 'cam_left_*.png'))
        img_right_paths = glob.glob(os.path.join(self.save_path, 'cam_right_*.png'))
        img_left_paths.sort()
        img_right_paths.sort()

        # read stereo calibration results
        with open(os.path.join(self.save_path, 'stereo_calib.pkl'), 'rb') as f:
            stereo_calib = pickle.load(f)
        mtx_left = stereo_calib['mtx_left']
        dist_left = stereo_calib['dist_left']
        mtx_right = stereo_calib['mtx_right']
        dist_right = stereo_calib['dist_right']
        R = stereo_calib['R']
        T = stereo_calib['T']
        E = stereo_calib['E']
        F = stereo_calib['F']

        img_shape = cv2.imread(img_left_paths[0]).shape[:2][::-1]
        # stereo rectification
        R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
            mtx_left, dist_left, mtx_right, dist_right, img_shape, R, T)
        map_left1, map_left2 = cv2.initUndistortRectifyMap(
            mtx_left, dist_left, R1, P1, img_shape, cv2.CV_16SC2)
        map_right1, map_right2 = cv2.initUndistortRectifyMap(
            mtx_right, dist_right, R2, P2, img_shape, cv2.CV_16SC2)

        for i, img_left_path in enumerate(img_left_paths):
            img_left = cv2.imread(img_left_path)
            img_right = cv2.imread(img_right_paths[i])
            img_left_rectified = cv2.remap(img_left, map_left1, map_left2, cv2.INTER_LINEAR)
            img_right_rectified = cv2.remap(img_right, map_right1, map_right2, cv2.INTER_LINEAR)
            rectified_imgs = np.hstack((img_left_rectified, img_right_rectified))
            # draw lines
            for j in range(0, rectified_imgs.shape[0], 16):
                cv2.line(rectified_imgs, (0, j), (rectified_imgs.shape[1], j), (0, 255, 0), 1, 1)

            cv2.imwrite(str(rectify_save_path / f'rectified_{i}.png'), rectified_imgs)


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data/mscrad4r_pkl/')
parser.add_argument('--data_name', type=str, default='230402_cam_calib')
parser.add_argument('--save_dir', type=str, default='./data/mscrad4r_pkl/stereo_calib/')
args = parser.parse_args()


if __name__ == '__main__':
    data_path = os.path.join(args.data_dir, args.data_name)
    save_path = os.path.join(args.save_dir, args.data_name)
    stereo = Stereo(data_path, save_path)
    # stereo.read()
    # stereo.calib()
    stereo.rectify()
