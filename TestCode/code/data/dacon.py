import os

from data import common
from data import srdata

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

class Dacon(srdata.SRData):
    def __init__(self, args, train=True):
        super(Dacon, self).__init__(args, train)
        self.repeat = args.test_every // (args.n_train // args.batch_size)

    def _scan(self):
        list_hr = []
        list_lr = [[] for _ in self.scale]

        for filename in os.listdir(self.dir_hr):
            list_hr.append(os.path.join(self.dir_hr,filename))
            for si, s in enumerate(self.scale):
                list_lr[si].append(os.path.joint(self.dir_lr,filename))
        
        return list_hr, list_lr

    def _set_filesystem(self, dir_data):
        data_dir = 'dacon/train' if self.train else 'dacon/val'
        self.apath = os.path.join(dir_data,data_dir) 
        self.dir_hr = os.path.join(self.apath, 'hr')
        self.dir_lr = os.path.join(self.apath, 'lr')
        self.ext = '.png'

    def _name_hrbin(self):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_HR.npy'.format(self.split)
        )

    def _name_lrbin(self, scale):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_LR_X{}.npy'.format(self.split, scale)
        )

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

