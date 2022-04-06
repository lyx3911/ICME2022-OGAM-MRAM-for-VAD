import torch
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import cv2


from models import FlowNet2  # the path is depended on where you create this module
from utils_flownet2.frame_utils import read_gen  # the path is depended on where you create this module
from utils_flownet2 import flow_utils, tools
import torchvision.transforms as transforms

import torch.nn.functional as F


if __name__ == '__main__':
    # obtain the necessary args for construct the flownet framework
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    
    args = parser.parse_args()

    # initial a Net
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = FlowNet2(args).to(device)
    # load the state_dict
    dict = torch.load("FlowNet2_checkpoint.pth.tar")
    net.load_state_dict(dict["state_dict"])
    
    # load the image pair, you can find this operation in dataset.py
    pim1 = cv2.imread("/data0/lyx/VAD_datasets/ped2/training/frames/01/001.jpg")
    pim1 = cv2.resize(pim1, (512, 384))
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    pim1 = transform(pim1).unsqueeze(dim=0).to(device)

    pim2 = cv2.imread("/data0/lyx/VAD_datasets/ped2/training/frames/01/002.jpg")
    pim2 = cv2.resize(pim2, (512, 384))
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    pim2 = transform(pim2).unsqueeze(dim=0).to(device)

    pred_flow_esti_tensor = torch.cat([pim1.view(-1, 3, 1, pim1.shape[-2], pim1.shape[-1]),
                                    pim2.view(-1, 3, 1, pim2.shape[-2], pim2.shape[-1])], 2)

    result = net(pred_flow_esti_tensor * 255.0).squeeze()


    # pim1 = read_gen("data/chairs/005.jpg")
    # pim2 = read_gen("data/chairs/006.jpg")
    # print(pim1.shape)
    # images = [pim1, pim2]

    # images = np.array(images).transpose(3, 0, 1, 2)
    # print(images.shape)
    # im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()

    # process the image pair to obtian the flow
    # result = net(im).squeeze()


    # save flow, I reference the code in scripts/run-flownet.py in flownet2-caffe project
    def writeFlow(name, flow):
        f = open(name, 'wb')
        f.write('PIEH'.encode('utf-8'))
        np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
        flow = flow.astype(np.float32)
        flow.tofile(f)
        f.flush()
        f.close()


    data = result.data.cpu().numpy().transpose(1, 2, 0)
    print(data.shape) 
 

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    data2 = transform(data)
    print(data2.shape)
    print(data2.unsqueeze(0).shape)
    data2 = data2.unsqueeze(0)
    data2 = F.interpolate(data2, size=([128,128]), mode='bilinear', align_corners=False)
    data2 = data2.squeeze(0).numpy().transpose(1,2,0)
    print(data2.shape)   

    
    img = flow_utils.flow2img(data)
    img2 = flow_utils.flow2img(data2)
    # plt.imshow(img)
    # plt.show()
    print('11')
    plt.imsave('data/test.png', img)
    plt.imsave('data/test2.png', img2)

    writeFlow("data/0000001-img.flo", data)

    # flow_utils.visulize_flow_file(
    #     os.path.join(flow_folder, '%06d.flo' % (batch_idx * args.inference_batch_size + i)), flow_vis_folder)

