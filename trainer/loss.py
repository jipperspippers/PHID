import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from pytorch_wavelets import DWTForward


def mydistance(x,y):
    F = x-y
    g = torch.sqrt(torch.pow(F, 2) + 1e-6)
    mg = g.sum(4, keepdim=True).sum(3, keepdim=True).sum(2,keepdim=True)/ (g.size(2)*g.size(3)*g.size(4))
    return mg.squeeze(4).squeeze(3).squeeze(2)

def cal_sam(X,Y):
    esp = 1e-6
    InnerPro = torch.sum(X*Y,1,keepdim=True)
    len1 = torch.norm(X,p=2,dim=1,keepdim=True)
    len2 = torch.norm(Y,p=2,dim=1,keepdim=True)
    divisor = len1*len2
    mask = torch.eq(divisor,0)
    divisor = divisor + (mask.float())*esp
    cosA = torch.sum(InnerPro/divisor,1).clamp(-1+esp,1-esp)
    sam = torch.acos(cosA)
    return torch.mean(sam)/np.pi


class FW_Loss(torch.nn.Module):
    def __init__(self,lam=0.5,alpha=0.25,):
        super(FW_Loss, self).__init__()
        self.DWT = DWTForward(J=3, wave='haar').cuda() 
        self.DWT.eval()
        
        self.l1loss = nn.L1Loss()
        self.l2loss = nn.MSELoss()
        
        self.alpha = alpha
        self.lam = lam
        self.distance = mydistance
        self.sam = cal_sam

    def forward(self,y,gt):
        batch_size = y.size()[0]
        loss = self.l1loss(y,gt)
        loss_sam = self.sam(y,gt)

        yl,yh = self.DWT(y.squeeze(1))
        gtl,gth = self.DWT(gt.squeeze(1))
        loss_l = self.l2loss(yl,gtl)
        
        total_loss = self.lam * loss  + self.alpha * loss_l -(self.lam) * loss_sam
        
        yh = [yh[i].transpose(2, 1) for i in range(len(yh))]
        gth = [gth[i].transpose(2, 1) for i in range(len(gth))]

        for i in range(len(yh)):
            omega = self.distance(yh[i],gth[i])
            omega = torch.sqrt(omega)
            
            loss_h = self.distance(yh[i],gth[i])  
            loss_wh = torch.bmm(omega.unsqueeze(dim=1), loss_h.unsqueeze(dim=2)).sum()
            total_loss += self.alpha * pow(2,2-i) * loss_wh.squeeze()/batch_size
            
        return total_loss