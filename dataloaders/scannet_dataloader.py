import os
import sys
import numpy as np
import torch
from torchvision import transforms as torch_transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image, ImageEnhance
from dataloaders.dataloader import collect_batch_data
from . import associate 
import torch.nn.functional as F

# Dataset details: http://www.scan-net.org/ScanNet/, https://github.com/ScanNet/ScanNet
class ScanNetMetaDataset:
    def __init__(self, root, mode='val', modality=['rgb', 'd', 'pose'], output_size=(224, 224), val_transform_type = "direct_resizing", split_start = 0, split_end = 1.0):
        self.root = root
        self.pose_root = os.path.join(root, 'pose_per_frame')
        self.modality = modality # (rgb, d, sd, pose)
        self.output_size = output_size # (width, height)
        self.iheight, self.iwidth = 968, 1296
        self.depth_iheight, self.depth_iwidth = 480, 640

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
        if mode == 'val':
            self.transform = self.val_transform
        else:
            raise (RuntimeError("Invalid dataset type: " + mode + "\n"
                                "Supported dataset types are: val"))
        # camera intrinsics
        self.color_intrinsics = torch.from_numpy(np.loadtxt(os.path.join(self.root, 'intrinsic/intrinsic_color.txt'))).float().view(4, 4)
        self.depth_intrinsics = torch.from_numpy(np.loadtxt(os.path.join(self.root, 'intrinsic/intrinsic_depth.txt'))).float().view(4, 4)
        self.intrinsics = {
            'fx': self.color_intrinsics[0, 0],
            'fy': self.color_intrinsics[1, 1],
            'cx': self.color_intrinsics[0, 2],
            'cy': self.color_intrinsics[1, 2],
        }
        # val transforms 
        self.val_transform_type = val_transform_type # direct_resizing, resize_centercrop_resize, dinov2

    def __getraw__(self, index):
        # print("__getraw__ index: " + str(index) + ", self.real_idx[index]: " + str(self.real_idx[index])+ "\n")
        raw_items = {}
        timestamp = float(os.path.splitext(self.matches[index][0].split('.')[0])[0])
        raw_items['timestamp'] = timestamp
        raw_items['real_idx'] = self.real_idx[index] # real indices if adding sway to frames 
        if 'rgb' in self.modality:
            rgb_file = os.path.join(os.path.join(self.root, 'color'), self.matches[index][0])
            rgb = Image.open(rgb_file)
            raw_items['rgb'] = rgb
        if 'd' in self.modality:
            depth_file = os.path.join(os.path.join(self.root, 'depth'), self.matches[index][1])
            depth = Image.open(depth_file)
            raw_items['depth'] = depth
        if 'pose' in self.modality:
            trans_file = os.path.join(self.pose_root, f'{self.real_idx[index]}', 'translation.pt')
            rot_file = os.path.join(self.pose_root, f'{self.real_idx[index]}', 'rotation.pt')
            translation = torch.load(trans_file)
            rotation = torch.load(rot_file)
            raw_items['trans'] = translation
            raw_items['rot'] = rotation

        if rgb.mode != "RGB":
            raise Exception("Color image is not in RGB format")
        if depth.mode not in ["I", "I;16"]:
            raise Exception("Depth image is not in intensity format")

        return raw_items

    def __getitem__(self, index):
        raw_items = self.__getraw__(index)
        items = self.transform(raw_items)
        return collect_batch_data(**items)

    def __len__(self):
        return len(self.matches)

    def _readDataset(self):
        # if associations file exists, read matches from associations.txt
        if os.path.isfile(self.root + "/associations.txt"):
            # read file 
            with open(os.path.join(self.root + "/associations.txt"), 'r') as f:
                matches_filenames = [(line.split()[0], line.split()[1]) for line in f ]
            f.close()
        else: # associations file does not exist, make associations 
            rgb_files = sorted(os.listdir(os.path.join(self.root, 'color')))
            depth_files = sorted(os.listdir(os.path.join(self.root, 'depth')))
            # strip timestamps from filenames 
            rgb_timestamps = [int(x[:-4]) for x in rgb_files]
            depth_timestamps = [int(x[:-4]) for x in depth_files]
            # make matches from closest timestamps 
            matches = associate.associate(rgb_timestamps, depth_timestamps, 0, 1)
            # find index of that timestamp in rgb_files 
            matches_filenames = []
            for i in range(len(matches)):
                # find timestamp of rgb and depth in rgb_files names
                # pad timestamp to correct number of digits 
                for j in rgb_files:
                    if j.find(str(matches[i][0]).zfill(4)) != -1:
                        rgb_file = j
                for j in depth_files:
                    if j.find(str(matches[i][1]).zfill(4)) != -1:
                        depth_file = j
                matches_filenames.append((rgb_file, depth_file))
                # save matches as associations file
                with open(self.root + "/associations.txt", 'a+') as f:
                    f.write(rgb_file + " " + depth_file + "\n")
            f.close()
        return matches_filenames

    def val_transform(self, items):
        rgb, depth = np.array(items['rgb']), items['depth']
        if self.val_transform_type == "direct_resizing":
            rgb_t = F.interpolate(torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float(), self.output_size, mode="bilinear", align_corners=True).squeeze(0) # Tensor 
            # depth_t = F.interpolate(torch.tensor(depth).unsqueeze(0).unsqueeze(0), size=self.output_size, mode='nearest').squeeze(0).float()
            depth_t = torch.tensor(np.array(depth.resize(size=self.output_size, resample=Image.NEAREST)).astype(np.int16)).unsqueeze(0)
            items['rgb'] = rgb_t / 255.0
        elif self.val_transform_type == "dinov2":
            rgb_t = F.interpolate(torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float(), self.output_size, mode="bilinear", align_corners=True).squeeze(0) # Tensor 
            # add additional step on RGB after resizing 
            transform_norm = torch_transforms.Compose([
                        # lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
                        torch_transforms.Normalize(
                            mean=(123.675, 116.28, 103.53),
                            std=(58.395, 57.12, 57.375),
                        ),
                    ])
            rgb_t = transform_norm(rgb_t)
            depth_t = torch.tensor(np.array(depth.resize(size=self.output_size, resample=Image.NEAREST)).astype(np.int16)).unsqueeze(0)
            items['rgb'] = rgb_t
        else:
            print("This val transform not implemented for ScanNet!")
        if rgb_t.shape[1] != depth_t.shape[1] or rgb_t.shape[2] != depth_t.shape[2]:
            raise Exception("Color and depth image do not have the same resolution.")
        items['depth'] = depth_t / 1000.0 # adjust scaling correctly [m]
        return items