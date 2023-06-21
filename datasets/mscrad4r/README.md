# Preparation

Read rosbag and save them as pkl files for later use.
Example usage:
to prepare the data for calibration, run the following command
`sh
python -m datasets.mscrad4r.readbag \
    --data_dir ./data/mscrad4r/ \
    --bag_dir URBAN_A0 \
    --bag_name URBAN_A0.bag \
    --save_dir ./data/mscrad4r_pkl \
`   

# Calibrations

Stereo camera calibration

`sh
python -m datasets.mscrad4r.readbag \
    --data_dir ./data/mscrad4r \
    --bag_dir ./data/mscrad4r/URBAN_Stereo_Camera_Calibration \
    --bag_name 230402_cam_calib.bag \
    --save_dir ./data/mscrad4r_pkl \
python -m datasets.mscrad4r.extract_stereo_calib
`

Radar, LiDAR, and camera calibration

`sh
python -m datasets.mscrad4r.readbag \
    --data_dir ./data/mscrad4r \
    --bag_dir ./data/mscrad4r/URBAN_LiDAR_Camera_Calibration \
    --bag_name 230402_lidar_cam_calib.bag \
    --save_dir ./data/mscrad4r_pkl \
python -m datasets.mscrad4r.extract_stereo_lidar_radar_extrinsic
`
