import os
import cv2
import torch

import numpy as np


from torch.utils.data import Dataset


def read_keypoint(jpg_path: str) -> torch.FloatTensor:
    # read keypoint from dir
    # assume .pts file is in the same fir as .jpg
    pts_name = jpg_path[:-4] + '.pts'
    with open(pts_name) as f:
        lines = f.readlines()
        if lines[0].startswith('version'):  # to support different formats
            lines = lines[3:-1]
        mat = np.fromstring(''.join(lines), sep=' ')
        mat_tensor = mat.reshape(-1, 2)
        # visibility = torch.ones([68, 1], dtype=torch.float)
        # keypoint = torch.cat((mat_tensor, visibility), dim=1)
    return mat_tensor


class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dir_to_jpgs: str, transform=None):
        """
        Arguments:
            dir_to_folder (string): Path to folder with .jpg and .pts files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = dir_to_jpgs
        self.resize = 224
        self.transform = transform
        self.images = []
        for idx, fname in enumerate(os.listdir(self.root_dir)):
            cur_path = os.path.join(self.root_dir, fname)
            if cur_path.endswith('.jpg'):
                self.images.append(cur_path)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        # (np_array[68, 2]) format points : [x, y]
        keypoint = read_keypoint(img_path)
        img = img / 255  # normalize values
        img_height, img_width, _ = img.shape
        img = cv2.resize(img, (self.resize, self.resize))

        if self.transform is not None:
            # to tensor, from shape (H, W, C) -> (C, H, W)
            img = self.transform(img)

        keypoint = keypoint * [self.resize /
                               img_width, self.resize / img_height]

        return {'image': img.to(torch.float),
                'keypoints': torch.tensor(keypoint, dtype=torch.float),
                }

    def __len__(self):
        return len(self.images)
