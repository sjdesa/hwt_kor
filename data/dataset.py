# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
#import lmdb
import torchvision.transforms as transforms
import six
import sys
from PIL import Image
import numpy as np
import os
import sys
import pickle
import numpy as np
from params import *
import glob, cv2
import torchvision.transforms as transforms

def crop_(input):
    image = Image.fromarray(input)
    image = image.convert('L')
    binary_image = image.point(lambda x: 0 if x > 127 else 255, '1')
    bbox = binary_image.getbbox()
    cropped_image = image.crop(bbox)
    return np.array(cropped_image)

def get_transform(grayscale=False, convert=True):

    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))

    if convert:
        transform_list += [transforms.Resize(32), transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)

# def load_itw_samples(folder_path, num_samples = 15):

#   paths = glob.glob(f'{folder_path}/*')
#   paths = np.random.choice(paths, num_samples, replace = len(paths)<=num_samples)

#   words = [os.path.basename(path_i)[:-4] for path_i in paths]

#   imgs = [np.array(Image.open(i).convert('L')) for i in paths]

#   imgs =  [crop_(im) for im in imgs]
#   imgs = [cv2.resize(imgs_i, (int(32*(imgs_i.shape[1]/imgs_i.shape[0])), 32)) for imgs_i in imgs]
#   max_width = 192

#   imgs_pad = []
#   imgs_wids = []

#   trans_fn = get_transform(grayscale=True)

#   for img in imgs:

#       img = 255 - img
#       img_height, img_width = img.shape[0], img.shape[1]
#       outImg = np.zeros(( img_height, max_width), dtype='float32')
#       outImg[:, :img_width] = img[:, :max_width]

#       img = 255 - outImg

#       imgs_pad.append(trans_fn((Image.fromarray(img))))
#       imgs_wids.append(img_width)

#   imgs_pad = torch.cat(imgs_pad, 0)

#   return imgs_pad.unsqueeze(0).cuda(), torch.Tensor(imgs_wids).unsqueeze(0).cuda()


class TextDataset():

    def __init__(self, base_path = DATASET_PATHS,  num_examples = NUM_EXAMPLES, target_transform=None):

        self.NUM_EXAMPLES = num_examples
  
        base_path = DATASET_PATHS
        with open(base_path, "rb") as f:
            data = pickle.load(f)

        self.IMG_DATA  = data['train'] #[:NUM_WRITERS])
        if 'None' in self.IMG_DATA.keys():
            del self.IMG_DATA['None']
                    
        self.author_id = list(self.IMG_DATA.keys())

        self.transform = get_transform(grayscale=True)
        self.target_transform = target_transform
        
        self.collate_fn = TextCollator()


    def __len__(self):
        return len(self.author_id)

    def __getitem__(self, index):

        

        NUM_SAMPLES = self.NUM_EXAMPLES


        author_id = self.author_id[index]

        self.IMG_DATA_AUTHOR = self.IMG_DATA[author_id]
        random_idxs = np.random.choice(len(self.IMG_DATA_AUTHOR), NUM_SAMPLES, replace = True)

        rand_id_real = np.random.choice(len(self.IMG_DATA_AUTHOR))
        real_img = self.transform(self.IMG_DATA_AUTHOR[rand_id_real]['img'].convert('L'))
        real_labels = self.IMG_DATA_AUTHOR[rand_id_real]['label'].encode()

        imgs = [np.array(self.IMG_DATA_AUTHOR[idx]['img'].convert('L')) for idx in random_idxs]
        labels = [self.IMG_DATA_AUTHOR[idx]['label'].encode() for idx in random_idxs]
       
        width = [img.shape[1] for img in imgs] 
        max_width = max(width)
        
        imgs_pad = []
        imgs_wids = []

        for img in imgs:

            img = 255 - img
            img_height, img_width = img.shape[0], img.shape[1]
            outImg = np.zeros(( img_height, max_width), dtype='float32')
            outImg[:, :img_width] = img[:, :max_width]

            img = 255 - outImg

            imgs_pad.append(self.transform((Image.fromarray(img))))
            imgs_wids.append(img_width)

        imgs_pad = torch.cat(imgs_pad, 0)
        

        item = {'simg': imgs_pad, 'swids':imgs_wids, 'img' : real_img, 'label':real_labels,'img_path':'img_path', 'idx':'indexes', 'wcl':index}
    


        return item




class TextDatasetval():

    def __init__(self, base_path = DATASET_PATHS, num_examples = NUM_EXAMPLES, target_transform=None):
        
        self.NUM_EXAMPLES = num_examples
        
        base_path = DATASET_PATHS
        
        with open(base_path, "rb") as f:
            data = pickle.load(f)
        

        self.IMG_DATA  = data['test']
        if 'None' in self.IMG_DATA.keys():
            del self.IMG_DATA['None']
        self.author_id = list(self.IMG_DATA.keys())

        self.transform = get_transform(grayscale=True)
        self.target_transform = target_transform
        
        self.collate_fn = TextCollator()
    

    def __len__(self):
        return len(self.author_id)

    def __getitem__(self, index):

        NUM_SAMPLES = self.NUM_EXAMPLES

        author_id = self.author_id[index]

        self.IMG_DATA_AUTHOR = self.IMG_DATA[author_id]
        random_idxs = np.random.choice(len(self.IMG_DATA_AUTHOR), NUM_SAMPLES, replace = True)

        rand_id_real = np.random.choice(len(self.IMG_DATA_AUTHOR))
        real_img = self.transform(self.IMG_DATA_AUTHOR[rand_id_real]['img'].convert('L'))
        real_labels = self.IMG_DATA_AUTHOR[rand_id_real]['label'].encode()


        imgs = [np.array(self.IMG_DATA_AUTHOR[idx]['img'].convert('L')) for idx in random_idxs]
        labels = [self.IMG_DATA_AUTHOR[idx]['label'].encode() for idx in random_idxs]
       
        width = [img.shape[1] for img in imgs] 
        max_width = max(width)
        
        imgs_pad = []
        imgs_wids = []

        for img in imgs:

            img = 255 - img
            img_height, img_width = img.shape[0], img.shape[1]
            outImg = np.zeros(( img_height, max_width), dtype='float32')
            outImg[:, :img_width] = img[:, :max_width]

            img = 255 - outImg

            imgs_pad.append(self.transform((Image.fromarray(img))))
            imgs_wids.append(img_width)

        imgs_pad = torch.cat(imgs_pad, 0)
        

        item = {'simg': imgs_pad, 'swids':imgs_wids, 'img' : real_img, 'label':real_labels,'img_path':'img_path', 'idx':'indexes', 'wcl':index}
    


        return item




class TextCollator(object):
    def __init__(self):
        self.resolution = resolution

    def __call__(self, batch):

        img_path = [item['img_path'] for item in batch]
        width = [item['img'].shape[2] for item in batch]
        indexes = [item['idx'] for item in batch]
        simgs =  torch.stack([item['simg'] for item in batch], 0)
        wcls =  torch.Tensor([item['wcl'] for item in batch])
        swids =  torch.Tensor([item['swids'] for item in batch])
        imgs = torch.ones([len(batch), batch[0]['img'].shape[0], batch[0]['img'].shape[1], max(width)], dtype=torch.float32)
        for idx, item in enumerate(batch):
            try:
                imgs[idx, :, :, 0:item['img'].shape[2]] = item['img']
            except:
                print(imgs.shape)
        item = {'img': imgs, 'img_path':img_path, 'idx':indexes, 'simg': simgs, 'swids': swids, 'wcl':wcls}
        if 'label' in batch[0].keys():
            labels = [item['label'] for item in batch]
            item['label'] = labels
        if 'z' in batch[0].keys():
            z = torch.stack([item['z'] for item in batch])
            item['z'] = z
        return item

