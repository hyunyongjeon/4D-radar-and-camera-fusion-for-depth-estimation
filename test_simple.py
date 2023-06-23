# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
from multiprocessing import Process, Queue

import torch
from torchvision import transforms, datasets
import open3d as o3d

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
from evaluate_depth import STEREO_SCALE_FACTOR

from visualize import pcd_vis, depth_vis_for_test_simple
import cv2

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str, 
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--model_name', type=str, 
                        help='name of a pretrained model to use',
                        choices=[
                            "mono_640x192",
                            "M_640x192_",
                            "M_640x192_230413",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320",
                            "mscrad4r_320x256_230621/models/weights_19"])
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument("--pred_metric_depth",
                        help='if set, predicts metric depth instead of disparity. (This only makes sense for stereo-trained KITTI models).',
                        action='store_true')

    return parser.parse_args()


def create_point_actor(points, colors):
    """ open3d point cloud from numpy array """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud


def o3dvis(video):
    print("o3dvis process ...")
    o3dvis.points = []

    def animation_callback(vis):
        cam = vis.get_view_control().convert_to_pinhole_camera_parameters()

        points, colors = [], []
        while not video.pcd.empty():
            point, color = video.pcd.get()
            points.append(point)
            colors.append(color)
        for i in range(len(points)):
            pts = points[i]
            clr = colors[i]
            point_actor = create_point_actor(pts, clr)
            
            vis.clear_geometries()
            vis.add_geometry(point_actor)
            

            o3dvis.points.append(point_actor)

        cam = vis.get_view_control().convert_from_pinhole_camera_parameters(cam)

        vis.poll_events()
        vis.update_renderer()
        
        
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_animation_callback(animation_callback)

    vis.create_window(height=540, width=960)
    
    vis.run()
    vis.destroy_window()


class pcdBuffer():
    def __init__(self):
        self.pcd = Queue()
        

def test_simple(args):
    visualizer = None
    pcd_buff = pcdBuffer()
    
    """Function to predict for a single image or folder of images
    """
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.pred_metric_depth and "stereo" not in args.model_name:
        print("Warning: The --pred_metric_depth flag only makes sense for stereo-trained KITTI "
              "models. For mono-trained models, output depths will not in metric space.")
        
    download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("models", args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")
    
    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)
    depth_decoder.to(device)
    depth_decoder.eval()
    
    # import pdb; pdb.set_trace()

    # FINDING INPUT IMAGES
    if os.path.isfile(args.image_path):
        # Only testing on a single image
        paths = [args.image_path]
        output_directory = os.path.dirname(args.image_path)

    elif os.path.isdir(args.image_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
        # how to sort the paths in orderly
        paths.sort()
        output_directory = args.image_path
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths):
            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)
            
            """Prediction the depth map"""
            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features) #outputs.keys() = dict_keys(['disp',3),('disp',2),('disp',1),('disp',0)])
            
            disp = outputs[("disp", 0)] #disp.shape = torch.Size([1,1,192,640])
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False) 
                #disp_resized.shape = torch.Size([1,1,235,638])

            """Saving numpy file"""
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
            # # if args.pred_metric_depth:
            # #     name_dest_npy = os.path.join(output_directory,"{}_depth.npy".format(output_name))
            # #     metric_depth = STEREO_SCALE_FACTOR * depth.cpu().numpy()
            # #     np.save(name_dest_npy, metric_depth)
            # # else:
            # #     name_dest_npy = os.path.join(output_directory,"{}_disp.npy".format(output_name))
            # #     np.save(name_dest_npy, scaled_disp.cpu().numpy())
            
            """Saving colormapped depth image"""
            # disp_resized_np = disp_resized.squeeze().cpu().numpy()
            # vmax = np.percentile(disp_resized_np, 95)
            # normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            # mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            # # mapper = cm.ScalarMappable(norm=normalizer, cmap='plasma')
            # colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            # # # im = pil.fromarray(colormapped_im)
            # # # name_dest_im = os.path.join(output_directory,"disp_jpeg","{}_disp.jpeg".format(output_name))
            # # # im.save(name_dest_im)
            
            # print("   Processed {:d} of {:d} images - saved predictions to:".format(
            #     idx + 1, len(paths)))
            # print("   - {}".format(name_dest_im))
            # print("   - {}".format(name_dest_npy))
            
            """Visualize the colormapped depth image"""
            # import pdb; pdb.set_trace()
            depth_vis_for_test_simple(depth[0])
            # cv2.imshow("coloredmapped_im", colormapped_im)
            # cv2.waitKey(10)
            # cv2.destroyAllWindows()
            
            
            """visualize with o3d"""
            # # convert the depth tensor to a Numpy array.
            # # depth.shape = torch.Size([1, 1, 192, 640]). (batch_size, channel, height, width)
            # depth_np = depth.cpu().numpy()[0, 0]
            # # Extract the height and width of the depth image
            # h, w = depth_np.shape
            # # Camera intrinsic parameters from the depth image dimensions. 
            # # Assume that the depth image has the same aspect ratio as a 1440x1080 image
            # # (fx, fy, cx, cy) = (637.04, 253.86, 319.78, 95.91)
            # fx, fy, cx, cy = w * 1433.34 / 1440, h * 1427.96 / 1080, \
            #     w * 719.5 / 1440, h * 539.5 / 1080
            # # Create two 2D arrays of indices representing the row and column indices of each pixel in the depth image
            # y, x = np.mgrid[0:h, 0:w]
            # # Convert the pixel coordinates to world coordinates 
            # # by applying the camera intrinsic parameters to each pixel.
            # # The resulting 'X' and 'Y' arrays contain the world x and y coordinates of each pixel.
            # Y = (y - cy) * depth_np / fy
            # X = (x - cx) * depth_np / fx
            # # Concatenates the 'X', 'Y' and 'depth' arrays representin the 3D point cloud.
            # # pcd.shape = (num_points,3) = (122880, 3)
            # pcd = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1), depth_np.reshape(-1, 1)], axis=1)
            # # Extract the color information from the input image tensor.
            # # and, reshape from '(height,width,channels)' to '(num_points,3)'.
            # color = input_image[0].data.cpu().numpy().transpose(1, 2, 0)
            # color = color.reshape(-1, 3)
            # # Create a binary mask indicating which point in the point cloud should be included for visualization.
            # # The mask based on two conditions. 
            # # 1. the depth should be less than 1
            # # (i.e., within a certain distance from the camera).
            # # 2. the y must be greater than 30% of the image height
            # # (i.e., exclude points that are too close to the bottom of the image).
            # # mask = np.logical_and(depth_np < 1, y > 0.3 * h).reshape(-1)
            # mask = np.logical_and(depth_np < 5, y > 0.1 * h).reshape(-1)
            # # Put the filtered point cloud and color information into a multiprocessing queue,\
            # # which is used to communicate with the visualization process. 
            # pcd_buff.pcd.put([pcd[mask], color[mask]])
            # # Start to visualize the point cloud using Open3D
            # # The 'o3dvis' function takes a 'pcd_buff' object as input and continuously updates the visualization.
            # if visualizer is None:
            #     visualizer = Process(target=o3dvis, args=(pcd_buff,))
            #     visualizer.start()

    print('-> Done!')

if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
