import numpy as np
import os
import argparse
# import pypcd
from pypcd.pypcd import PointCloud
# import pypcd.pypcd.PointCloud
import csv
from tqdm import tqdm

def main():
    ## Add parser
    parser = argparse.ArgumentParser(description="Convert .pcd to .bin")
    parser.add_argument(
        "--pcd_path",
        help=".pcd file path.",
        type=str,
        default="/home/jeon/Desktop/RAD4R/URBAN_C0/radar_pcd"
    )
    parser.add_argument(
        "--bin_path",
        help=".bin file path.",
        type=str,
        default="/home/jeon/Desktop/RAD4R/URBAN_C0/radar"
    )
    parser.add_argument(
        "--file_name",
        help="File name.",
        type=str,
        default="oculii"
    )
    args = parser.parse_args()

    ## Find all pcd files
    pcd_files = []
    for (path, dir, files) in os.walk(args.pcd_path):
        for filename in files:
            # print(filename)
            ext = os.path.splitext(filename)[-1]
            if ext == '.pcd':
                pcd_files.append(path + "/" + filename)

    ## Sort pcd files by file name
    pcd_files.sort()   
    print("Finish to load point clouds!")

    ## Make bin_path directory
    try:
        if not (os.path.isdir(args.bin_path)):
            os.makedirs(os.path.join(args.bin_path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print ("Failed to create directory!!!!!")
            raise

    # ## Generate csv meta file
    # csv_file_path = os.path.join(args.bin_path, "meta.csv")
    # csv_file = open(csv_file_path, "w")
    # meta_file = csv.writer(
    #     csv_file, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
    # )
    # ## Write csv meta file header
    # meta_file.writerow(
    #     [
    #         "pcd file name",
    #         "bin file name",
    #     ]
    # )
    # print("Finish to generate csv meta file")


    ## Converting Process
    print("Converting Start!")
    seq = 0
    for pcd_file in tqdm(pcd_files):


        ## Get pcd file
        # pc = pypcd.PointCloud.from_path(pcd_file)
        pc = PointCloud.from_path(pcd_file)

        ## Generate bin file name
        # bin_file_name = "{}_{:05d}.bin".format(args.file_name, seq)
        bin_file_name = "{:010d}.bin".format(seq)
        bin_file_path = os.path.join(args.bin_path, bin_file_name)
        
        # ## Get data from pcd (x, y, z, intensity, ring, time)
        # np_x = (np.array(pc.pc_data['x'], dtype=np.float32)).astype(np.float32)
        # np_y = (np.array(pc.pc_data['y'], dtype=np.float32)).astype(np.float32)
        # np_z = (np.array(pc.pc_data['z'], dtype=np.float32)).astype(np.float32)
        # np_i = (np.array(pc.pc_data['intensity'], dtype=np.float32)).astype(np.float32)/256
        # # np_r = (np.array(pc.pc_data['ring'], dtype=np.float32)).astype(np.float32)
        # # np_t = (np.array(pc.pc_data['time'], dtype=np.float32)).astype(np.float32)

        """Get data from pcd (ours)"""
        np_x = (np.array(pc.pc_data['x'], dtype=np.float32)).astype(np.float32)
        np_y = (np.array(pc.pc_data['y'], dtype=np.float32)).astype(np.float32)
        np_z = (np.array(pc.pc_data['z'], dtype=np.float32)).astype(np.float32)
        np_s_0000 = (np.array(pc.pc_data['s_0000'], dtype=np.float32)).astype(np.float32)
        np_s_0001 = (np.array(pc.pc_data['s_0001'], dtype=np.float32)).astype(np.float32)
        np_s_0002 = (np.array(pc.pc_data['s_0002'], dtype=np.float32)).astype(np.float32)
        np_s_0003 = (np.array(pc.pc_data['s_0003'], dtype=np.float32)).astype(np.float32)
        np_alpha = (np.array(pc.pc_data['alpha'], dtype=np.float32)).astype(np.float32)
        np_beta = (np.array(pc.pc_data['beta'], dtype=np.float32)).astype(np.float32)
        np_range = (np.array(pc.pc_data['range'], dtype=np.float32)).astype(np.float32)
        np_doppler = (np.array(pc.pc_data['doppler'], dtype=np.float32)).astype(np.float32)
        np_power = (np.array(pc.pc_data['power'], dtype=np.float32)).astype(np.float32)
        np_recoveredSpeed = (np.array(pc.pc_data['recoveredSpeed'], dtype=np.float32)).astype(np.float32)
        np_dotFlags = (np.array(pc.pc_data['dotFlags'], dtype=np.float32)).astype(np.float32)
        np_denoiseFlag = (np.array(pc.pc_data['denoiseFlag'], dtype=np.float32)).astype(np.float32)
        np_historyFrameFlag = (np.array(pc.pc_data['historyFrameFlag'], dtype=np.float32)).astype(np.float32)
        np_dopplerCorrectionFlag = (np.array(pc.pc_data['dopplerCorrectionFlag'], dtype=np.float32)).astype(np.float32)


        ## Stack all data    
        # points_32 = np.transpose(np.vstack((np_x, np_y, np_z, np_i)))
        stacked_points = np.transpose(np.vstack((\
            np_x,np_y,np_z, \
            np_s_0000,np_s_0001,np_s_0002,np_s_0003, \
            np_alpha,np_beta, \
            np_range,np_doppler,np_power,np_recoveredSpeed, \
            np_dotFlags,np_denoiseFlag,np_historyFrameFlag,np_dopplerCorrectionFlag
            )))

        ## Save bin file                                    
        # points_32.tofile(bin_file_path)
        stacked_points.tofile(bin_file_path)

        # ## Write csv meta file
        # meta_file.writerow(
        #     [os.path.split(pcd_file)[-1], bin_file_name]
        # )

        seq = seq + 1
    
if __name__ == "__main__":
    main()
