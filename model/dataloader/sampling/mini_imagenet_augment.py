# MIT License

# Copyright (c) 2022 Yaqiang Sun

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import os.path as osp
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from PIL import Image

THIS_PATH = osp.dirname(__file__)
ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..', '..','..'))
ROOT_PATH2 = osp.abspath(osp.join(THIS_PATH, '..', '..','..'))
IMAGE_PATH1 = osp.join(ROOT_PATH2, 'data/miniimagenet/images')
SPLIT_PATH = osp.join(ROOT_PATH, 'data/miniimagenet/split')
CACHE_PATH = osp.join(ROOT_PATH, '.cache/')
import os

def identity(x):
    return x

class MiniImageNet(Dataset):

    def __init__(self, setname, args,augment=False):
        im_size = args.orig_imsize
        csv_path = osp.join(SPLIT_PATH, setname + '.csv')
        cache_path = osp.join( CACHE_PATH, "{}.{}.{}.pt".format(self.__class__.__name__, setname, im_size) )
        self.data, self.label = self.parse_csv(csv_path, setname)
        
        
        self.num_class = len(set(self.label))
        image_size = 84

        if 'num_patch' not in vars(args).keys():
            print ('do not assign num_patch parameter, set as default: 9')
            self.num_patch=9
        else:
            self.num_patch=args.num_patch


        if augment and setname == 'train':
            self.pre_transform = transforms.Compose(
                    [
                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2,hue=0.001),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(), 

            ])
            self.single_transform = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(image_size,scale=(0.7, 1.0), ratio=(3. / 4., 4. / 3.)),
                        transforms.ToTensor(),
                        transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                                        np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))
            ])
            transforms_list = [
                    transforms.RandomResizedCrop(image_size,scale=(0.08, 0.7)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
        else:
            self.pre_transform = transforms.Compose(
                    [
            ])
            self.single_transform = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(image_size,scale=(0.7, 1.0), ratio=(3. / 4., 4. / 3.)),
                        transforms.ToTensor(),
                        transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                                        np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))
            ])
            transforms_list = [
                    transforms.RandomResizedCrop(image_size,scale=(0.08, 0.7)),

                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
                
        self.transform = transforms.Compose(
                    transforms_list + [
                    transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                                        np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))
            ])


    def parse_csv(self, csv_path, setname):
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in tqdm(lines, ncols=64):
            name, wnid = l.split(',')
            path = osp.join(IMAGE_PATH1, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append( path )
            label.append(lb)

        return data, label
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        patch_list=[]
        pil_image = Image.open(path).convert('RGB')
        pil_image = self.pre_transform(pil_image)
        patch_list.append(self.single_transform(pil_image))
        for _ in range(self.num_patch-1):
            patch_list.append(self.transform(pil_image))
        patch_list=torch.stack(patch_list,dim=0)
        return patch_list, label

if __name__ == '__main__':
    pass