import numpy as np
from os.path import *
# from scipy.misc import imresize
from imageio import imread
from . import flow_utils
import cv2

def read_gen(file_name):
    ext = splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        # im = imread(file_name)
        # im = imresize(im, (384, 512))
        im = cv2.imread(file_name)
        im = cv2.resize(im, (384,512))
        if im.shape[2] > 3:
            return im[:,:,:3]
        else:
            return im
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    elif ext == '.flo':
        return flow_utils.readFlow(file_name).astype(np.float32)
    return []
