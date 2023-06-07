This folder is for 
1. containing the samples of the R4DAR dataset 
2. extracting from *.bag to *.jpg and *.bin
   a. from *.bag to *.jpg
      Using "bag2image.py" file
   b. from *.bag to *.bin
      We need to proceed this steps "*.bag -> *.pcd -> *.bin"
      (*.bag -> *.pcd) 
      $ roscore 
      $ rosrun pcl_ros bag_to_pcd (bag_file_path/bag_file_name.bag) (topic_name) (output_path/folder_name)
      Ex. rosrun pcl_ros bag_to_pcd URBAN_C0/URBAN_C0.bag /oculii_radar/point_cloud URBAN_C0/radar_pcd
      (*.pcd -> *.bin)
      Using "pcd2bin.py"
