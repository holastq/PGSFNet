from torch.utils import data as data
from torchvision.transforms.functional import normalize
import torch
from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.matlab_functions import bgr2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.utils.matlab_functions import rgb2ycbcr
from basicsr.utils.degradations import blur, noisy, JPEG_compress
import numpy as np
import cv2
import os
import random

@DATASET_REGISTRY.register()
class PairedImageDataset(data.Dataset):# for test
    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']


        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)


        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)

def get_RealCE(opt, gt_folder):
    phase = opt['phase']
    gt_folder = os.path.join(gt_folder)

    fl_13mmdir = os.path.join(gt_folder, "13mm")
    fl_26mmdir = os.path.join(gt_folder, "26mm")
    fl_52mmdir = os.path.join(gt_folder, "52mm")

    paths = []
    scale = opt['scale']

    imlist = os.listdir(fl_13mmdir)
    if phase in ["val", "train"]:
        valid_list = open(os.path.join(gt_folder, "valid_list.txt"), "r").readlines()
        valid_list = [line.replace("\n", "") for line in valid_list]
        imlist = valid_list

    for imname in imlist:
        impath_13mm = os.path.join(fl_13mmdir, imname)
        impath_26mm = os.path.join(fl_26mmdir, imname)
        impath_52mm = os.path.join(fl_52mmdir, imname)

        if phase in ["val"]:
            paths.append(
                    {"lq_path": impath_52mm, "gt_path": impath_52mm, 'syn_degr': False})
            if scale == 4:
                paths.append(
                    {"lq_path": impath_13mm, "gt_path": impath_52mm, 'syn_degr': False}) # impath_52mm
            elif scale == 2:
                paths.append(
                    {"lq_path": impath_26mm, "gt_path": impath_52mm, 'syn_degr': False})
                paths.append(
                    {"lq_path": impath_13mm, "gt_path": impath_26mm, 'syn_degr': False})
            else:
                raise Exception("Sorry, Real-CE has no this scale", scale)


        else:
            paths.append(
                    {"lq_path": impath_52mm, "gt_path": impath_52mm, 'syn_degr': False})
            if scale == 4:
                paths.append(
                        {"lq_path": impath_13mm, "gt_path": impath_52mm, 'syn_degr': False})
            paths.append(
                    {"lq_path": impath_26mm, "gt_path": impath_52mm, 'syn_degr': False})
            paths.append(
                    {"lq_path": impath_52mm, "gt_path": impath_52mm, 'syn_degr': True})

    return paths

def demo(opt, gt_folder):
    phase = opt['phase']
    lq_folder = os.path.join(gt_folder)
    fl_lq = os.path.join(lq_folder, "lq")

    paths = []
    scale = opt['scale']

    imlist = os.listdir(fl_lq)

    for imname in imlist:
        impath_lq = os.path.join(fl_lq, imname)
        paths.append(
                {"lq_path": impath_lq, 'syn_degr': False})
    return paths

@DATASET_REGISTRY.register()
class PairedImageDatasetRealCEwDEC(data.Dataset):
    def __init__(self, opt):
        super(PairedImageDatasetRealCEwDEC, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folders, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.gt_folder = self.gt_folders

        if not type(self.gt_folder) == list:
            self.gt_folders = [self.gt_folders]

        self.paths = []

        for gt_folder in self.gt_folders:

            # print("gt_folder:", gt_folder)

            if self.io_backend_opt['type'] == 'lmdb':
                self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
                self.io_backend_opt['client_keys'] = ['lq', 'gt']
                self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
            elif "RealCE" in gt_folder:
                self.paths.extend(get_RealCE(self.opt, gt_folder))
                print("get the path of realce!")
            else:
                self.paths.extend(demo(self.opt, gt_folder))

        self.nSamples = len(self.paths)
        print("nSamples:", self.nSamples)

    def __len__(self):
        return self.nSamples

    def degradation(self, image_hr, training):# degradate lq
        H, W, C = image_hr.shape
        # Down-sample
        img_lq = cv2.resize(image_hr, (W // 2, H // 2), interpolation=cv2.INTER_CUBIC)
        if random.random() > 0.3 or not training:
            img_lq = blur(img_lq, not training)
        if random.random() > 0.3 or not training:
            img_lq = noisy("gauss", img_lq, not training)
        if random.random() > 0.3 or not training:
            img_lq = JPEG_compress(img_lq, not training)
        img_lq = cv2.resize(img_lq, (W, H), interpolation=cv2.INTER_CUBIC)
        return img_lq

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        
        if self.opt['phase'] in ["val", "train"]:
            gt_path = self.paths[index]['gt_path']
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        H, W = img_lq.shape[:2]
        if self.opt['phase'] in ["val", "train"]:
            img_gt = cv2.resize(img_gt, (W//2, H//2), interpolation=cv2.INTER_CUBIC)
            img_lq = cv2.resize(img_lq, (W //2// scale, H //2// scale), interpolation=cv2.INTER_CUBIC)

        else:
            img_lq = cv2.resize(img_lq, (W //2// scale, H //2// scale), interpolation=cv2.INTER_CUBIC)

        # if self.paths[index]['syn_degr']:
        #     img_lq = self.degradation(img_lq, self.opt['phase'] == "train")

        H_lq, W_lq = img_lq.shape[:2]
        if self.opt['phase'] in ["val", "train"]:
            img_gt = img_gt[:H_lq * scale, :W_lq * scale]
         
        if self.paths[index]['syn_degr']:
            pass

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            if self.opt['phase'] in ["val", "train"]:
                img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        if self.opt['phase'] == 'val':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        if self.opt['phase'] in ["val", "train"]:
            img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        else:
            img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)

        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            if self.opt['phase'] in ["val", "train"]:
                normalize(img_gt, self.mean, self.std, inplace=True)
                
        if self.opt['phase'] in ["val", "train"]:
            ret_data = {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}
        ret_data = {'lq': img_lq, 'lq_path': lq_path}

        return ret_data

    def __len__(self):
        return len(self.paths)
    