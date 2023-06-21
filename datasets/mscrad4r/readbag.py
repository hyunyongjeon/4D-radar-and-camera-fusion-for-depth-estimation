# read rosbag and save to pickle file
# python -m datasets.mscrad4r.readbag 

import pickle
import numpy as np
import pandas as pd
import os
import struct
import argparse

import rospy
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.image import message_to_cvimage
import ros_numpy

from sensor_msgs.msg import Image, PointCloud2, PointField

# prefix to the names of dummy fields we add to get byte alignment correct. this needs to not
# clash with any actual field names
DUMMY_FIELD_PREFIX = '__'

# mappings between PointField types and numpy types
type_mappings = [(PointField.INT8, np.dtype('int8')), (PointField.UINT8, np.dtype('uint8')), (PointField.INT16, np.dtype('int16')),
                 (PointField.UINT16, np.dtype('uint16')), (PointField.INT32, np.dtype('int32')), (PointField.UINT32, np.dtype('uint32')),
                 (PointField.FLOAT32, np.dtype('float32')), (PointField.FLOAT64, np.dtype('float64'))]
pftype_to_nptype = dict(type_mappings)
nptype_to_pftype = dict((nptype, pftype) for pftype, nptype in type_mappings)

# sizes (in bytes) of PointField types
pftype_sizes = {PointField.INT8: 1, PointField.UINT8: 1, PointField.INT16: 2, PointField.UINT16: 2,
                PointField.INT32: 4, PointField.UINT32: 4, PointField.FLOAT32: 4, PointField.FLOAT64: 8}

from . import cv_bridge
# from cv_bridge import CvBridge

import pdb


class ReadBag:
    def __init__(self, bag_path, topic_names, save_path):
        self.bag_path = bag_path
        self.topic_names = topic_names
        self.save_path = save_path

        # self.bridge = CvBridge()

    
    def read(self):
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        
        cam_left_msgs = {'time': [], 'msg': []}
        cam_right_msgs = {'time': [], 'msg': []}
        radar_msgs = {'time': [], 'msg': []}
        lidar_msgs = {'time': [], 'msg': []}

        topics = [self.topic_names[n] for n in self.topic_names.keys()]
        with AnyReader([Path(self.bag_path)]) as reader:
            connections = [x for x in reader.connections if x.topic in topics]
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, connection.msgtype)
                if 'Image' in connection.msgtype:
                    msg = message_to_cvimage(msg)
                elif 'PointCloud2' in connection.msgtype:
                    msg = self.pointcloud2_to_array(msg)
                else:
                    continue
                
                if connection.topic == self.topic_names['cam_left']:
                    cam_left_msgs['time'].append(timestamp)
                    cam_left_msgs['msg'].append(msg)
                elif connection.topic == self.topic_names['cam_right']:
                    cam_right_msgs['time'].append(timestamp)
                    cam_right_msgs['msg'].append(msg)
                elif connection.topic == self.topic_names['radar']:
                    radar_msgs['time'].append(timestamp)
                    radar_msgs['msg'].append(msg)
                elif connection.topic == self.topic_names['lidar']:
                    lidar_msgs['time'].append(timestamp)
                    lidar_msgs['msg'].append(msg)
                else:
                    raise ValueError('Wrong topic name')

        cam_left_msgs['time'] = np.array(cam_left_msgs['time'])
        cam_left_msgs['msg'] = np.array(cam_left_msgs['msg'])
        cam_right_msgs['time'] = np.array(cam_right_msgs['time'])
        cam_right_msgs['msg'] = np.array(cam_right_msgs['msg'])
        radar_msgs['time'] = np.array(radar_msgs['time'])
        lidar_msgs['time'] = np.array(lidar_msgs['time'])
        lidar_msgs['msg'] = np.array(lidar_msgs['msg'])

        with open(os.path.join(self.save_path, 'cam_left.pkl'), 'wb') as f:
            pickle.dump(cam_left_msgs, f)
        with open(os.path.join(self.save_path, 'cam_right.pkl'), 'wb') as f:
            pickle.dump(cam_right_msgs, f)
        with open(os.path.join(self.save_path, 'radar.pkl'), 'wb') as f:
            pickle.dump(radar_msgs, f)
        with open(os.path.join(self.save_path, 'lidar.pkl'), 'wb') as f:
            pickle.dump(lidar_msgs, f)


    def fields_to_dtype(self, fields, point_step):
        '''Convert a list of PointFields to a numpy record datatype.
        '''
        offset = 0
        np_dtype_list = []
        for f in fields:
            while offset < f.offset:
                # might be extra padding between fields
                np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
                offset += 1

            dtype = pftype_to_nptype[f.datatype]
            if f.count != 1:
                dtype = np.dtype((dtype, f.count))

            np_dtype_list.append((f.name, dtype))
            offset += pftype_sizes[f.datatype] * f.count

        # might be extra padding between points
        while offset < point_step:
            np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
            offset += 1
            
        return np_dtype_list

    def pointcloud2_to_array(self, cloud_msg, squeeze=True):
        ''' Converts a rospy PointCloud2 message to a numpy recordarray 
        
        Reshapes the returned array to have shape (height, width), even if the height is 1.

        The reason for using np.frombuffer rather than struct.unpack is speed... especially
        for large point clouds, this will be <much> faster.
        '''
        # construct a numpy record type equivalent to the point type of this cloud
        dtype_list = self.fields_to_dtype(cloud_msg.fields, cloud_msg.point_step)

        # parse the cloud into an array
        cloud_arr = np.frombuffer(cloud_msg.data, dtype_list)

        # remove the dummy fields that were added
        cloud_arr = cloud_arr[
            [fname for fname, _type in dtype_list if not (fname[:len(DUMMY_FIELD_PREFIX)] == DUMMY_FIELD_PREFIX)]]
        
        if squeeze and cloud_msg.height == 1:
            return np.reshape(cloud_arr, (cloud_msg.width,))
        else:
            return np.reshape(cloud_arr, (cloud_msg.height, cloud_msg.width))

    

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data/mscrad4r/')
parser.add_argument('--bag_dir', type=str, default='URBAN_Radar_Camera_Calibration')
parser.add_argument('--bag_name', type=str, default='230402_radar_cam_calib.bag')
parser.add_argument('--cam_left_topic_name', type=str, default='/camera_array/left/image_raw')
parser.add_argument('--cam_right_topic_name', type=str, default='/camera_array/right/image_raw')
parser.add_argument('--radar_topic_name', type=str, default='/oculii_radar/point_cloud')
parser.add_argument('--lidar_topic_name', type=str, default='/ouster/points')
parser.add_argument('--save_dir', type=str, default='./data/mscrad4r_pkl/')
args = parser.parse_args()


if __name__ == '__main__':
    bag_path = os.path.join(args.data_dir, args.bag_dir, args.bag_name)
    save_path = os.path.join(args.save_dir, args.bag_name[:-4])

    topic_names = {
        'cam_left': args.cam_left_topic_name,
        'cam_right': args.cam_right_topic_name,
        'radar': args.radar_topic_name,
        'lidar': args.lidar_topic_name
    }

    readbag = ReadBag(bag_path, topic_names, save_path)
    readbag.read()
