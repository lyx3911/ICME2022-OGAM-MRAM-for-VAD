import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import os
import time
from pathlib import Path

from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages
from yolov5.utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from yolov5.utils.plots import plot_one_box
from yolov5.utils.torch_utils import select_device, load_classifier, time_synchronized
import sys
sys.path.insert(0, './yolov5')

import cv2


def get_yolo_roi(img_path, model, device, dataset_name):
    # 面积大于阈值
    if dataset_name == "ped2":
        min_area_thr = 10*10
    elif dataset_name == "avenue":
        min_area_thr = 30*30
    elif dataset_name == "ShanghaiTech":
        min_area_thr = 8*8
    else: 
        raise NotImplementedError
    

    dataset = LoadImages(img_path, img_size=640)
    for path, img, im0s, vid_cap in dataset:
        p, s, im0 = Path(path), '', im0s

        # print(device)
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.25, 0.45)

        bboxs = []
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                # results
                bboxs = [] 
                for *xyxy, conf, cls in reversed(det):
                    box = [int(x.cpu().item()) for x in xyxy]
                    if (box[3]-box[1]+1)*(box[2]-box[0]+1) > min_area_thr:
                        bboxs.append( tuple(box) )

        return bboxs

def delCoverBboxes(bboxes, dataset_name):
    # 合并交并比大于阈值的框
    if dataset_name == "ped2":
        cover_thr = 0.6
    elif dataset_name == "avenue":
        cover_thr = 0.6
    elif dataset_name == "ShanghaiTech":
        cover_thr = 0.65
    else:
        raise NotImplementedError

    xmin = np.array([bbox[0] for bbox in bboxes])
    ymin = np.array([bbox[1] for bbox in bboxes])
    xmax = np.array([bbox[2] for bbox in bboxes])
    ymax = np.array([bbox[3] for bbox in bboxes])
    bbox_areas = (ymax-ymin+1) * (xmax-xmin+1)

    sort_idx = bbox_areas.argsort()#Index of bboxes sorted in ascending order by area size
    
    keep_idx = []
    for i in range(sort_idx.size):
        #Calculate the point coordinates of the intersection
        x11 = np.maximum(xmin[sort_idx[i]], xmin[sort_idx[i+1:]]) 
        y11 = np.maximum(ymin[sort_idx[i]], ymin[sort_idx[i+1:]])
        x22 = np.minimum(xmax[sort_idx[i]], xmax[sort_idx[i+1:]])
        y22 = np.minimum(ymax[sort_idx[i]], ymax[sort_idx[i+1:]])
        #Calculate the intersection area
        w = np.maximum(0, x22-x11+1)    
        h = np.maximum(0, y22-y11+1)  
        overlaps = w * h
        
        ratios = overlaps / bbox_areas[sort_idx[i]]
        num = ratios[ratios > cover_thr]
        if num.size == 0:  
            keep_idx.append(sort_idx[i])

    return [bboxes[i] for i in keep_idx]


def get_motion_roi(img_batch, bboxes, dataset_name):
    # 结合yolo提取出的目标位置和图像的adsdiff计算roi区域
    if dataset_name == 'ped2':
        area_thr = 10 * 10
        binary_thr = 18
        extend = 2
        gauss_mask_size = 3
    elif dataset_name == 'avenue':
        area_thr = 30 * 30
        binary_thr = 18
        extend = 2
        gauss_mask_size = 5
    elif dataset_name == 'ShanghaiTech':
        area_thr = 8 * 8
        binary_thr = 15
        extend = 2
        gauss_mask_size = 5
    else:
        raise NotImplementedError
        
    sum_grad = 0
    for i in range(len(img_batch)-1):    
        img1 = cv2.imread(img_batch[i])
        img2 = cv2.imread(img_batch[i+1])
        img1 = cv2.GaussianBlur(img1, (gauss_mask_size, gauss_mask_size), 0)
        img2 = cv2.GaussianBlur(img2, (gauss_mask_size, gauss_mask_size), 0)
        # 帧差法
        grad = cv2.absdiff(img1, img2)
        sum_grad = grad + sum_grad
    sum_grad = cv2.threshold(sum_grad, binary_thr, 255, cv2.THRESH_BINARY)[1]

    #sum_grad中减去yolo提取出的目标框
    for bbox in bboxes:
        extend_x1 = max(0, bbox[0]-extend)
        extend_y1 = max(0, bbox[1]-extend)
        extend_x2 = min(bbox[2]+extend, sum_grad.shape[1])
        extend_y2 = min(bbox[3]+extend, sum_grad.shape[0])
        sum_grad[extend_y1:extend_y2+1, extend_x1:extend_x2+1] = 0
    
    sum_grad = cv2.cvtColor(sum_grad, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(sum_grad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fg_bboxes = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        area = (w+1) * (h+1)
        if area > area_thr and w / h < 10 and h / w < 10:
            extend_x1 = int(max(0, x-extend))
            extend_y1 = int(max(0, y-extend))
            extend_x2 = int(min(x+w+extend, sum_grad.shape[1]))
            extend_y2 = int(min(y+h+extend, sum_grad.shape[0]))
            fg_bboxes.append((extend_x1, extend_y1, extend_x2, extend_y2))

    return fg_bboxes


def draw_bbox(img, b_box, color, width):
    for box in b_box:
        (xmin, ymin, xmax, ymax) = box 
        print(box)
        cv2.rectangle(img,(xmin,ymin),(xmax,ymax), color,width)
    return img

def overlap_area(box1, box2):
    (xmin, ymin, xmax, ymax) = box1
    (amin, bmin, amax, bmax) = box2
    if (xmax <= amin or amax<=xmin) and (ymax<=bmin or bmax<=ymin):
        return 0
    else:
        len = min(xmax, amax) - max(xmin, amin)
        wid = min(ymax, bmax) - max(ymin, bmin)
        return len*wid

def area(box):
    return abs(box[2]-box[0])*(box[3]-box[1])
    
def MergeRoI(roi_boxes):
    roi = []
    for box1 in roi_boxes:
        area = (box1[0]-box1[2])*(box1[1]-box1[3])
        for box2 in roi_boxes:
            if box1 != box2:
                if overlap_area(box1, box2) > 0 and overlap_area(box1, box2)/area > 0.3:
                    xmin = min(box1[0], box2[0])
                    ymin = min(box1[1], box2[1])
                    xmax = max(box1[2], box2[2])
                    ymax = max(box1[3], box2[3])
                    box1 = (xmin, ymin, xmax, ymax)
                    area = (box1[0]-box1[2])*(box1[1]-box1[3])

            for box2 in roi:
                if box1 != box2:
                    if overlap_area(box1, box2) > 0 and overlap_area(box1, box2)/area > 0.3:
                        xmin = min(box1[0], box2[0])
                        ymin = min(box1[1], box2[1])
                        xmax = max(box1[2], box2[2])
                        ymax = max(box1[3], box2[3])
                        box1 = (xmin, ymin, xmax, ymax)
                        area = (box1[0]-box1[2])*(box1[1]-box1[3])
                        roi.remove(box2)
        roi.append(box1)
    return roi

def YoloRoI(frames, dataset, model, device):
    yolo_boxes = get_yolo_roi(frames, model, device, dataset)
    yolo_boxes = delCoverBboxes(yolo_boxes, dataset)    
    return yolo_boxes

def RoI(frames, dataset, model, device):
    yolo_boxes = get_yolo_roi(frames[int(len(frames)/2)], model, device, dataset)
    yolo_boxes = delCoverBboxes(yolo_boxes, dataset)
    motion_boxes = get_motion_roi(frames, yolo_boxes, dataset)
    bboxes = delCoverBboxes(yolo_boxes+motion_boxes, dataset)
    return bboxes

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", default='yolov5/weights/yolov5s.pt')
    parser.add_argument("--datadir", default="../../VAD_datasets/ped2/training/frames")
    parser.add_argument("--datatype", default="ped2")
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--save_path", default="./bboxes/ped2/train/")
    args = parser.parse_args()
    
    # load yolov3 model
    weights = args.weight
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    model = attempt_load(weights, map_location=device)  # load FP32 model

    frame_path = args.datadir # frame path
    clips = sorted(os.listdir(frame_path))
    print(clips)

    for clip in clips:
        path = os.path.join(frame_path,clip)
        filenames = sorted(os.listdir(path))  
        save_file = os.path.join(args.save_path, str(clip)+".npy")
        clips_roi = []
        #读取图片开始预测
        for index in range(2,len(filenames)-2):       
            img1 = os.path.join(path, filenames[index-2])
            img2 = os.path.join(path, filenames[index-1])
            img3 = os.path.join(path, filenames[index])
            img4 = os.path.join(path, filenames[index+1])
            img5 = os.path.join(path, filenames[index+2])

            roi = RoI([img1, img2, img3, img4, img5], args.datatype, model, device)

            clips_roi.append(roi)

        # 保存
        np.save(save_file, clips_roi)
        print("save {}".format(clip))

