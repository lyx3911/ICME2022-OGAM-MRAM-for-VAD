import torch
import torch.nn as nn
from torch.autograd import Variable
class Flow_Loss(nn.Module):
    def __init__(self):
        super(Flow_Loss,self).__init__()

    def forward(self, gen_flows,gt_flows):

        return torch.mean(torch.abs(gen_flows - gt_flows))

class Intensity_Loss(nn.Module):
    def __init__(self,l_num):
        super(Intensity_Loss,self).__init__()
        self.l_num=l_num
    def forward(self, gen_frames,gt_frames):

        return torch.mean(torch.abs((gen_frames-gt_frames)**self.l_num))

class Gradient_Loss(nn.Module):
    def __init__(self,alpha,channels):
        super(Gradient_Loss,self).__init__()
        self.alpha=alpha
        filter=torch.FloatTensor([[-1.,1.]]).cuda()

        self.filter_x = filter.view(1,1,1,2).repeat(1,channels,1,1)
        self.filter_y = filter.view(1,1,2,1).repeat(1,channels,1,1)


    def forward(self, gen_frames,gt_frames):


        # pos=torch.from_numpy(np.identity(channels,dtype=np.float32))
        # neg=-1*pos
        # filter_x=torch.cat([neg,pos]).view(1,pos.shape[0],-1)
        # filter_y=torch.cat([pos.view(1,pos.shape[0],-1),neg.vew(1,neg.shape[0],-1)])
        gen_frames_x=nn.functional.pad(gen_frames,(1,0,0,0))
        gen_frames_y=nn.functional.pad(gen_frames,(0,0,1,0))
        gt_frames_x=nn.functional.pad(gt_frames,(1,0,0,0))
        gt_frames_y=nn.functional.pad(gt_frames,(0,0,1,0))

        gen_dx=nn.functional.conv2d(gen_frames_x,self.filter_x)
        gen_dy=nn.functional.conv2d(gen_frames_y,self.filter_y)
        gt_dx=nn.functional.conv2d(gt_frames_x,self.filter_x)
        gt_dy=nn.functional.conv2d(gt_frames_y,self.filter_y)

        grad_diff_x=torch.abs(gt_dx-gen_dx)
        grad_diff_y=torch.abs(gt_dy-gen_dy)

        return torch.mean(grad_diff_x**self.alpha+grad_diff_y**self.alpha)

class Adversarial_Loss(nn.Module):
    def __init__(self):
        super(Adversarial_Loss,self).__init__()
    def forward(self, fake_outputs):
        return torch.mean((fake_outputs-1)**2/2)
class Discriminate_Loss(nn.Module):
    def __init__(self):
        super(Discriminate_Loss,self).__init__()
    def forward(self,real_outputs,fake_outputs ):
        return torch.mean((real_outputs-1)**2/2)+torch.mean(fake_outputs**2/2)

class ObjectLoss(nn.Module):
    def __init__(self, device, l_num, dynamic=False):
        super(ObjectLoss, self).__init__()
        self.device =device
        self.l_num=l_num
        self.dynamic = dynamic
    def area(self, bbox):
        return (bbox[2]-bbox[0])*(bbox[3]-bbox[1])

    def forward(self, outputs, target, flow, bboxes):
        cof = torch.ones((outputs.shape[0], outputs.shape[2], outputs.shape[3])).to(self.device)
        boxcof = 2
        flowcof = 2
        for bbox in bboxes:
            if self.dynamic == True:
                cof[:, bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1+3500/self.area(bbox)
            else:
                cof[:, bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1+boxcof


        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        rad = (u ** 2 + v ** 2)
        rad = rad.view(rad.shape[0], -1)

        min = rad.min(1)[0]
        max = rad.max(1)[0]

        rad = (rad - min) / (max - min)

        cof = torch.mul(cof, flowcof * (1+rad.view(rad.shape[0], flow.shape[-2], flow.shape[-1])))

        return torch.mean(torch.mul(cof, torch.abs((outputs - target) ** self.l_num)))