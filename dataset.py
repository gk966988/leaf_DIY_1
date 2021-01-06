#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pdb
import cv2
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import itertools
import random

class MyData(Dataset):
    def __init__(self, root, df, transform=None, phase='train'):
        self.root = root
        self.df = df
        self.transform = transform
        self.phase = phase
        self.ids, self.labels, self.images_name = self._read_img_ids(self.root)

    def _read_img_ids(self, root):
        labels = []
        if self.phase == 'train':
            # df = pd.read_csv(os.path.join(self.root, '{}.csv'.format(self.phase)))
            ids = [[],[],[],[],[]]
            for i in range(len(self.df)):
                label = self.df[i,1]
                labels.append(label)
                ids[label].append(os.path.join(self.root, 'train_images', self.df[i,0]))

            labels = [[i]*len(ids[i]) for i in range(5)]
            labels = list(itertools.chain.from_iterable(labels))  # 将list转换为迭代器
            ids = list(itertools.chain.from_iterable(ids))
            # randnum = random.randint(0, 100)
            # random.seed(randnum)
            # random.shuffle(ids)
            # random.seed(randnum)
            # random.shuffle(labels)
            root_path = labels   # train不需要root_path，此处随便设置的
        else:
            path_dir = os.path.join(root, 'test_images')
            root_path = os.listdir(path_dir)
            ids = [os.path.join(path_dir, e) for e in root_path]
        return ids, labels, root_path

    def __len__(self):
        return len(self.ids)

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, item):
        id = self.ids[item]
        img_name = self.images_name[item]
        image = cv2.imread(id)  # (C, H, W)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[item] if self.phase =='train' else 0
        if self.transform != None:
            image, _ = self.transform(image, label)
        return image, label, img_name

class FGVC7Data(Dataset):
    def __init__(self, root, transform=None, phase='train'):
        self.root = root
        self.transform = transform
        self.phase = phase
        self.ids, self.labels, self.images_name = self._read_img_ids(self.root)

    def _read_img_ids(self, root):
        labels = []
        if self.phase == 'train':
            df = pd.read_csv(os.path.join(self.root, '{}.csv'.format(self.phase)))
            ids = [[],[],[],[],[]]
            for i in range(len(df)):
                label = df.iloc[i].values[1]
                labels.append(label)
                ids[label].append(os.path.join(self.root, 'train_images', df.iloc[i].values[0]))

            labels = [[i]*len(ids[i]) for i in range(5)]
            labels = list(itertools.chain.from_iterable(labels))  # 将list转换为迭代器
            ids = list(itertools.chain.from_iterable(ids))
            randnum = random.randint(0, 100)
            random.seed(randnum)
            random.shuffle(ids)
            random.seed(randnum)
            random.shuffle(labels)
            root_path = labels   # train不需要root_path，此处随便设置的
        else:
            path_dir = os.path.join(root, 'test_images')
            root_path = os.listdir(path_dir)
            ids = [os.path.join(path_dir, e) for e in root_path]
        return ids, labels, root_path

    def __len__(self):
        return len(self.ids)

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, item):
        id = self.ids[item]
        img_name = self.images_name[item]
        image = cv2.imread(id)  # (C, H, W)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[item] if self.phase =='train' else 0
        if self.transform != None:
            image, _ = self.transform(image, label)
        return image, label, img_name
# if __name__ == '__main__':
#     import sys
#     sys.path.append(os.path.abspath('./'))
#     from utils.utils import get_transform
#     data_path = r'F:\competion\6\cassava-leaf-disease-classification'
#     # data_path = r'/home/gongke/data/cassava-leaf-disease-classification'
#     # data_path = r'../data/cassava-leaf-disease-classification'
#     dataset = FGVC7Data(data_path, transform=get_transform((448,448), 'train'), phase='test')
#     from torch.utils.data import DataLoader
#     loader = DataLoader(dataset,batch_size=1)
#     d = []
#     for i, input in enumerate(loader):
#         x, _, y = input
#         d.append(y[0])
#     sub = pd.DataFrame({'image_id': d})
#     print(sub)