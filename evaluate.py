import argparse
import os
import sys
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from tqdm import tqdm
from collections import OrderedDict
import glob
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.autograd import Variable

from models.networks import define_G

import utils
from vad_dataloader_attention import VadDataset
from losses import *

import torchvision.transforms as transforms

import cv2
import scipy.io

import matplotlib.pyplot as plt

def ObjectLoss_evaluate(test_dataloader, generator, labels_list, videos, dataset, device, frame_height=256, frame_width=256): 
    #init
    psnr_list = {}
    roi_psnr_list = {}
    for key in videos.keys():
        psnr_list[key] = []
        roi_psnr_list[key] = []

    if dataset == "ShanghaiTech":
        object_loss = ObjectLoss(device, 2, dynamic=True)
    else:
        object_loss = ObjectLoss(device, 2)

    video_num = 0
    frame_index = 0
    label_length = videos[sorted(videos.keys())[video_num]]['length']

    # test
    generator.eval()
    for k, (frames, objects, bboxes, flow) in enumerate(tqdm(test_dataloader, desc='test', leave=False)):
        if k == label_length - 4 * (video_num + 1):
            video_num += 1
            label_length += videos[sorted(videos.keys())[video_num]]['length']
            frame_index = 0
            # bboxes = np.load(os.path.join(test_bboxes,bboxes_list[video_num] ), allow_pickle=True) 

        frames = frames.cuda()
        flow = flow.cuda()
        input = frames[:, :-1, ]
        target = frames[:, -1, ]

        try:
            objects = objects.cuda()
            outputs = generator(input, objects[:,:,2])
        except:
            output = generator(input, None)          


        mse_imgs = object_loss((outputs + 1) / 2, (target + 1) / 2, flow, bboxes).item()
        psnr_list[ sorted(videos.keys())[video_num] ].append(utils.psnr(mse_imgs))
        
    # normalize and evaluate
    anomaly_score_total_list = []
    for video_name in sorted(videos.keys()):
        video_bboxes = np.load("./bboxes/{}/test/{}.npy".format(dataset, video_name), allow_pickle=True)
        maxpsnr = np.max(psnr_list[video_name])
        for i, frame_bboxes in enumerate(video_bboxes):
            if len(frame_bboxes) == 0 :
                psnr_list[video_name][i] = maxpsnr
        
        # smooth:
        if dataset == "ShanghaiTech":
            psnr_list[video_name] = scipy.signal.savgol_filter(psnr_list[video_name], 53, 3)
        elif dataset == "avenue" or dataset == "ped2":
            psnr_list[video_name] = scipy.signal.savgol_filter(psnr_list[video_name], 5, 3)

        if dataset == "ped2" or dataset == "ShanghaiTech": 
            anomaly_score_total_list += utils.anomaly_score_list(psnr_list[video_name])
        elif dataset == "avenue":    
            anomaly_score_total_list += psnr_list[video_name]

    if dataset == "avenue":
        anomaly_score_total_list = utils.anomaly_score_list(anomaly_score_total_list)

    # TODO
    anomaly_score_total_list = np.asarray(anomaly_score_total_list)    
    frame_AUC = utils.AUC(anomaly_score_total_list, np.expand_dims(1 - labels_list, 0))

    return frame_AUC, psnr_list





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="../../VAD_datasets/ped2/testing/frames/")
    parser.add_argument("--dataset", default="ped2")
    parser.add_argument('--gpu', default='0')
    parser.add_argument("--weight", default=None)    
    # for flownet2, no need to modify
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    args = parser.parse_args()

    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu
    else:
        torch.cuda.set_device(int(args.gpu))
    utils.set_gpu(args.gpu)

    # test data:
    test_folder = args.datadir
    bbox_folder = "./bboxes/{}/test".format(args.dataset)
    flow_folder = "./flow/{}/test".format(args.dataset)
    dataset = args.dataset


    test_dataset = VadDataset(args,video_folder= test_folder, bbox_folder = bbox_folder, flow_folder=flow_folder,
                        transform=transforms.Compose([transforms.ToTensor()]),
                        resize_height=256, resize_width=256)

    test_dataloader = DataLoader(test_dataset, batch_size=1,
                                 shuffle=False, num_workers=0, drop_last=False)


    # model_init
    weight_path = args.weight
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    
    # generator.load_state_dict(torch.load(weight_path))
    ngf = 64
    netG = 'resnet_6blocks_attention'
    norm = 'instance'
    no_dropout = False
    init_type = 'normal'
    init_gain = 0.02
    gpu_ids = []
    generator = define_G(3, 3, ngf, netG, norm, not no_dropout, init_type, init_gain, gpu_ids)
    
    weight = torch.load(weight_path)
    generator.load_state_dict(weight, strict=False)
    generator.cuda()
    # print(generator)

    labels = scipy.io.loadmat('./data/{}_frame_labels.mat'.format(dataset))
   
    # init
    videos = OrderedDict()
    videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
    labels_list = []
    label_length = 0
    psnr_list = {}
    for video in sorted(videos_list):
        video_name = os.path.split(video)[-1]
        videos[video_name] = {}
        videos[video_name]['path'] = video
        videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
        videos[video_name]['frame'].sort()
        videos[video_name]['length'] = len(videos[video_name]['frame'])
        labels_list = np.append(labels_list, labels[video_name][0][4:])
        label_length += videos[video_name]['length']
        psnr_list[video_name] = []

    frame_AUC , psnr_list = ObjectLoss_evaluate(test_dataloader, generator, labels_list, videos, dataset = dataset, device = device) 
    print(frame_AUC)