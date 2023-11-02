from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from .mono_dataset import MonoDataset

class NYUv2Dataset(MonoDataset):
	def __init__(self, *args, **kwargs):
		super(NYUv2Dataset, self).__init__(*args, **kwargs)

		self.K = np.array([[0.8107, 0, 0.5087, 0],
                           [0, 1.0822, 0.5286, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

		self.full_res_shape = (640, 480)

	def get_image_path(self, folder, frame_index, side):
		f_str = "img_{}{}".format(frame_index, self.img_ext)
		image_path = os.path.join(
			self.data_path,
			folder,
			f_str)
		return image_path

	def get_depth(self, folder, frame_index, side, do_flip):
		f_str = "gt_{}{}".format(frame_index, self.img_ext)
		depth_path = os.path.join(
			self.data_path,
			"depths/",
			f_str)
		
		depth_gt = pil.open(depth_path)
		depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
		depth_gt = np.array(depth_gt).astype(np.float32) / 256

		if do_flip:
		    depth_gt = np.fliplr(depth_gt)

		return depth_gt

	def get_color(self, folder, frame_index, side, do_flip):
		color = self.loader(self.get_image_path(folder, frame_index, side))

		if do_flip:
			color = color.transpose(pil.FLIP_LEFT_RIGHT)

		return color

	def check_depth(self):
		line = self.filenames[0].split()
		scene_name = line[0]
		frame_index = int(line[1])
		
		depth_filename = os.path.join(
			self.data_path,
			"depths/",
			"gt_{}.jpg".format(int(frame_index)))

