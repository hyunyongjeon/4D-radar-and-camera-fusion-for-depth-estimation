# python -m utils.vowl.generate_adverse

import os
import glob
import numpy as np
import cv2
import tqdm
from PIL import Image
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.core.lightning import LightningModule

from ruamel.yaml import YAML

from dataloaders.vowl import build_dataset, ROBOTCAR_ROOT
from utils.vowl import read_list_from_file

from models.cyclegan import create_model

import pdb

# date of data
_DATA_DATE = '2014-12-09-13-21-02'
# image directory
_IMG_TYPE = 'left'
_IMG_DIR = '/media/antyanta/Extreme Pro1/public/RobotCarDataset-Scraper/robotcar_dataset/{}/stereo/{}/'.format(_DATA_DATE, _IMG_TYPE)

class ImageLoader(torch.utils.data.Dataset):
    def __init__(self, ):
    	self.transform = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
			])

    	image_paths = read_list_from_file(
			_IMG_DIR.replace('/stereo/{}/'.format(_IMG_TYPE), '/train_split.txt'), 1)

    	self.image_paths = []
    	for image_path in image_paths:
    		if image_path.startswith('----'):
    			continue
    		self.image_paths.append(image_path)

    def __getitem__(self, index):
    	img_path = os.path.join(_IMG_DIR, self.image_paths[index]) + '.png'

    	img = Image.open(img_path)
    	img = demosaic(img, 'gbrg').astype(np.uint8)

    	img = self.transform(img)

    	return img, img_path

    def __len__(self):
    	return len(self.image_paths)

dataloader = torch.utils.data.DataLoader(
	ImageLoader(),
	batch_size=6, shuffle=False, num_workers=1, drop_last=False)

device = torch.device('cuda:1')

opt = YAML().load(open('./configs/vowl/vowl.yaml', 'r'))

cyclegan_types = [
		# ['oxford-day-night-256', 25],
		# ['oxford-day-night-512', 0],
		# ['oxford-day-snow', 25],
		# ['oxford-overcast-rain', 25],
		['oxford-overcast-night-rain', 50],
	]

for cyclegan_type, cyclegan_epoch in cyclegan_types:
	opt['model']['cyclegan']['epoch'] = cyclegan_epoch
	opt['model']['cyclegan']['name'] = cyclegan_type

	cyclegan = create_model(opt['model']['cyclegan'])
	cyclegan.setup(opt['model']['cyclegan'])

	target_ = cyclegan_type.split('-')[2:]
	target = 'left'
	for t in target_:
		target += '-'+t
	out_dir = _IMG_DIR.replace('left', target)
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)

	netG_A = cyclegan.netG_A.to(device)
	with tqdm.tqdm(dataloader, ascii=True) as tq:
		for img, img_path in tq:
			img_T = img.to(device)

			_, _, h, w = img_T.shape
			img_T = F.interpolate(img_T, 256)
			fake = netG_A(img_T)
			fake = (F.interpolate(fake, (h, w))*0.5)+0.5

			fake = fake.permute(0, 2, 3, 1).data.cpu().numpy()
			img_fakes = ((255*fake)[:,:,:,::-1]).astype(np.uint8)

			for i, img_fake in enumerate(img_fakes):
				cv2.imwrite(img_path[i].replace('left', target), img_fake)
