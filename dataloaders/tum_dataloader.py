import os
import sys
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image, ImageEnhance
from collections import namedtuple
from dataloaders.dataloader import collect_batch_data
import time 

class TUMMetaDataset:
    def __init__(self, root, mode='val', modality=['rgb', 'd', 'pose'], output_size=(224, 224), val_transform_type = "direct_resizing", split_start = 0, split_end = 1.0):
        self.root = root
        self.pose_root = os.path.join(root, 'pose_per_frame')
        self.modality = modality # (rgb, d, pose)
        self.output_size = output_size # (width, height)
        self.iheight, self.iwidth = 480, 640
        self.color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)

        self.matches = self._readDataset()
        if not (split_start == 0 and split_end == 1.0):
            if split_start > 1.0: # unit in frame
                split_start_idx = int(split_start)
            else:
                split_start_idx = min(round(split_start*len(self.matches)),len(self.matches)-1) # don't give out of range index 
            if split_end > 1.0: # unit in frame
                split_end_idx = int(split_end)
            else:
                split_end_idx = round(split_end*len(self.matches)-1)
            self.matches = self.matches[split_start_idx:split_end_idx:1]
        self.real_idx = list(range(0,len(self.matches))) # used for indexing pose, table
        print("Found {} images and depth pairs in {}.".format(len(self.matches), self.root))

        self.to_image = transforms.ToPILImage()

        if mode == 'train':
            self.transform = self.train_transform
        elif mode == 'val':
            self.transform = self.val_transform
        else:
            raise (RuntimeError("Invalid dataset type: " + mode + "\n"
                                "Supported dataset types are: train, val"))
        # val transforms 
        self.val_transform_type = val_transform_type # direct_resizing, resize_centercrop_resize, dinov2

    def __getraw__(self, index):
        # print("__getraw__ index: " + str(index) + ", self.real_idx[index]: " + str(self.real_idx[index])+ "\n")
        raw_items = {}
        timestamp = float(os.path.splitext(self.matches[index][0].split('/')[1])[0])
        raw_items['timestamp'] = timestamp
        raw_items['real_idx'] = self.real_idx[index] # real indices if adding sway to frames
        if 'rgb' in self.modality:
            rgb_file = os.path.join(self.root, self.matches[index][0])
            rgb = Image.open(rgb_file)
            raw_items['rgb'] = rgb
        if 'd' in self.modality:
            depth_file = os.path.join(self.root, self.matches[index][1])
            depth = Image.open(depth_file)
            raw_items['depth'] = depth
        if 'pose' in self.modality:
            trans_file = os.path.join(self.pose_root, f'{self.real_idx[index]}', 'translation.pt')
            rot_file = os.path.join(self.pose_root, f'{self.real_idx[index]}', 'rotation.pt')
            translation = torch.load(trans_file)
            rotation = torch.load(rot_file)
            raw_items['trans'] = translation
            raw_items['rot'] = rotation
        if rgb.size != depth.size:
            raise Exception("Color and depth image do not have the same resolution.")
        if rgb.mode != "RGB":
            raise Exception("Color image is not in RGB format")
        if depth.mode not in ["I", "I;16"]:
            raise Exception("Depth image is not in intensity format")
        return raw_items

    def __getitem__(self, index):
        raw_items = self.__getraw__(index)
        items = self.transform(raw_items)
        items['depth'] = items['depth'].type(torch.FloatTensor) / 5000.0
        return collect_batch_data(**items)

    def __len__(self):
        return len(self.matches)

    def _readDataset(self):
        with open(os.path.join(self.root, 'associations.txt'), 'r') as f:
            matches = [ (line.split()[3], line.split()[1]) if line.split()[3][:3] == 'rgb' else (line.split()[1], line.split()[3]) for line in f ]
        return matches

    def val_transform(self, items):
        rgb, depth = items['rgb'], items['depth']

        to_tensor = transforms.ToTensor()
        if self.val_transform_type == "direct_resizing":
            transform = transforms.Compose([
                transforms.Resize(self.output_size, interpolation=InterpolationMode.NEAREST),
            ])            
            rgb_t = transform(to_tensor(rgb))
            depth_t = transform(to_tensor(depth))
        elif self.val_transform_type == "dinov2":
            transform = transforms.Compose([
                transforms.Resize(self.output_size, interpolation=InterpolationMode.NEAREST),
            ])            
            rgb_t = transform(to_tensor(rgb))
            depth_t = transform(to_tensor(depth))
            # add additional step on RGB after resizing 
            transform_norm = transforms.Compose([
                        lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
                        transforms.Normalize(
                            mean=(123.675, 116.28, 103.53),
                            std=(58.395, 57.12, 57.375),
                        ),
                    ])
            rgb_t = transform_norm(rgb_t)
        elif self.val_transform_type == "resize_centercrop_resize":
            transform = transforms.Compose([
                transforms.Resize(250.0 / self.iheight), # this is for computational efficiency, since rotation can be slow (in training)
                transforms.CenterCrop((228, 304)),
                transforms.Resize(self.output_size, interpolation=InterpolationMode.NEAREST),
            ])
            rgb_np = transform(np.array(rgb))
            depth_np = transform(np.array(depth))
            rgb_t = torch.from_numpy(rgb_np).permute(2, 0, 1)
            depth_t = torch.from_numpy(depth_np)
        else:
            print("This val_transform_type not implemented!")
        items['rgb'] = rgb_t
        items['depth'] = depth_t
        return items