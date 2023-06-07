# python -m utils.vowl.generate_adverse

import os
import glob
import numpy as np
import cv2
import tqdm
from PIL import Image
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic


import pdb

# date of data
_DATA_DATE = '2014-12-09-13-21-02'
# _DATA_DATE = '2014-12-16-18-44-24'
# image directory
_IMG_TYPE = 'left'
_IMG_DIR = '/media/antyanta/Extreme Pro1/public/RobotCarDataset-Scraper/robotcar_dataset/{}/stereo/{}/'.format(_DATA_DATE, _IMG_TYPE)


if not os.path.exists(_IMG_DIR.replace('left', 'left_rgb')):
		os.mkdir(_IMG_DIR.replace('left', 'left_rgb'))

image_paths = glob.glob(_IMG_DIR + '*')
with tqdm.tqdm(image_paths, ascii=True) as tq:
	for img_path in tq:

		img = Image.open(img_path)
		img = demosaic(img, 'gbrg')[:, :, ::-1].astype(np.uint8)
		cv2.imwrite(img_path.replace('left', 'left_rgb'), img)
