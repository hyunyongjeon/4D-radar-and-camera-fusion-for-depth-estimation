python -m datasets.mscrad4r.extract_stereo_lidar_radar_extrinsic \
    --data_dir ./data/mscrad4r_pkl \
    --data_name 230402_lidar_cam_calib 230402_radar_cam_calib URBAN_A0 URBAN_B0 URBAN_D0 URBAN_E0 URBAN_F0 URBAN_G0 URBAN_H0 \
    --save_dir ./data/mscrad4r_pkl/extrinsic_calib \
    --stereo_calib ./data/mscrad4r_pkl/stereo_calib/230402_cam_calib/stereo_calib.pkl