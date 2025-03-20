import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import json

class HS_multiscale_DSet(Dataset):
    def __init__(self, opt, split):
        self.opt = opt
        self.split = split
        # Read data paths from configuration
        self.scene_path = opt.scene_path
        self.gt_path = opt.gt_path
        with open(opt.json_file) as json_file:
            self.content = json.load(json_file)  # Load JSON data

        self.img_list = self.get_names(split)
        if split == 'train':
            random.shuffle(self.img_list)

    def get_names(self, split):
        if split == 'demo':
            # In demo mode, use alternative paths (can be extended via configuration)
            self.scene_path = '/amax/home/dzr/dzr/dzr/autoCalib/Mine/visual-paper/rebuttal_data/raw_scenes/'
            self.gt_path = '/amax/home/dzr/dzr/dzr/autoCalib/Mine/visual-paper/rebuttal_data/gt_files/'
            return os.listdir(self.scene_path)
        else:
            return self.content[split]

    def load_data(self, index):
        scene_name = self.img_list[index]
        # Remove file extension
        scene_name = scene_name[:-4]
        # Extract image ID (characters from index 3 onward)
        img_id = scene_name[3:]
        gt_name = 'gtRef_' + img_id
        img_A = np.load(os.path.join(self.scene_path, scene_name + '.npy'))
        img_A = img_A / 4096.0  # Normalize the image
        img_B = np.load(os.path.join(self.gt_path, gt_name + '.npy'))
        small_b = 1
        img_A_out = np.zeros((small_b, self.opt.crop_size, self.opt.crop_size, self.opt.in_channels))
        img_B_out = np.zeros((small_b, self.opt.crop_size, self.opt.crop_size, self.opt.in_channels))
        if self.split == 'train':
            for b in range(small_b):
                h, w = img_A.shape[:2]
                rand_h = random.randint(0, h - self.opt.crop_size)
                rand_w = random.randint(0, w - self.opt.crop_size)
                img_A_out[b] = img_A[rand_h:rand_h + self.opt.crop_size,
                                     rand_w:rand_w + self.opt.crop_size, :]
                img_B_out[b] = img_B[rand_h:rand_h + self.opt.crop_size,
                                     rand_w:rand_w + self.opt.crop_size, :]
            img_A_out = torch.from_numpy(img_A_out.astype(np.float32).transpose(0, 3, 1, 2)).contiguous()
            img_B_out = torch.from_numpy(img_B_out.astype(np.float32).transpose(0, 3, 1, 2)).contiguous()
            return img_A_out, img_B_out, scene_name
        img_A = torch.from_numpy(img_A.astype(np.float32).transpose(2, 0, 1)).contiguous()
        img_B = torch.from_numpy(img_B.astype(np.float32).transpose(2, 0, 1)).contiguous()
        return img_A, img_B, scene_name

    def __getitem__(self, index):
        return self.load_data(index)

    def __len__(self):
        return len(self.img_list)
