import torch
import torch.nn as nn
import model.block3d as B
from pytorch_wavelets import DWTInverse,DWTForward

def cal_gradient_vertical(F):
    c_x = F.size(-2)
    g = F[:, :, :, 1:, 1:] - F[:, :, :, :c_x - 1, 1:]
    return g  
  
def cal_gradient_horizontal(F):
    c_x = F.size(-1)
    g = F[:, :, :, 1:, 1:] - F[:, :, :, 1:, :c_x - 1]
    return g

def spa_gradient(F):
    x = cal_gradient_horizontal(F)
    y = cal_gradient_vertical(F)
    g = torch.sqrt(torch.pow(x, 2) + torch.pow(y,2) +1e-6)
    mg = g.sum(4, keepdim=True).sum(3, keepdim=True)/ (F.size(3) * F.size(4))
    return mg

# gradient-aware channel attention module
class GCALayer(nn.Module):
    def __init__(self, channel, f_channel=16):
        super(GCALayer, self).__init__()

        self.gradient = spa_gradient
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv_du = nn.Sequential(nn.Conv3d(channel, f_channel, 1, padding=0, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv3d(f_channel, channel, 1, padding=0, bias=True),
                                     nn.Sigmoid())
    def forward(self, x):
        y = self.gradient(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class HAT(nn.Module):
    def __init__(self, in_channels=3,num_features=64, act_type='prelu',
                 norm_type=None):
        super(HAT, self).__init__()
        
        self.gca = GCALayer(in_channels)
        self.convhead = B.ConvBlock(in_channels+1, num_features, kernel_size=3, act_type=act_type, norm_type=norm_type)
        self.convbody = B.ResidualDenseBlock_5C(num_features, num_features//2, (1,3,3),1,(0,1,1))
        self.convtail = B.ConvBlock(num_features, in_channels, kernel_size=3, act_type=act_type, norm_type=norm_type)
        
    def forward(self, xh,xl):
        xh = self.gca(xh)
        f = self.convhead(torch.cat((xh,xl),dim=1))
        f = self.convbody(f)
        f = self.convtail(f)
        return f

class LAT(nn.Module):
    def __init__(self, in_channels=1,num_features=32, act_type='prelu',act_mode = 'L',nb = 2,
                 norm_type=None):
        super(LAT, self).__init__()
        
        self.convhead = B.ConvBlock(in_channels, num_features, kernel_size=3, act_type=act_type, norm_type=norm_type)
        self.convbody = B.sequential(*[B.ResBlock(num_features, num_features, (1,3,3),1,(0,1,1),bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.convtail = B.ConvBlock(num_features, in_channels, kernel_size=3, act_type=act_type, norm_type=norm_type)
        
    def forward(self, xl):
        f = self.convhead(xl)
        f = self.convbody(f)
        f = self.convtail(f)
        return f

class SCRB(nn.Module):
    def __init__(self,  n_feats, in_channels, out_channels, bias=True, act=nn.ReLU(True)):
        super(SCRB, self).__init__()

        self.p3d1 = nn.Sequential(nn.Conv3d(in_channels, n_feats, (3,1,1),1,(1,0,0), bias=bias),
                                  nn.Conv3d(n_feats, in_channels, (1,3,3),1,(0,1,1), bias=bias), act)
        self.p3d2 = nn.Sequential(nn.Conv3d(in_channels*2, n_feats, (3,1,1),1,(1,0,0), bias=bias),
                                  nn.Conv3d(n_feats, in_channels, (1,3,3),1,(0,1,1), bias=bias), act)
        self.c1 = nn.Conv3d(in_channels*3, out_channels, 1, 1, 0)                     

    def forward(self, x):
        x1 = self.p3d1(x.unsqueeze(1))
        x2 = self.p3d2(torch.cat((x.unsqueeze(1),x1),dim=1))
        x3 = self.c1(torch.cat((x.unsqueeze(1),x1,x2),dim=1))+ x.unsqueeze(1)
        return x3.squeeze(1)

class PHID(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, dwt=3,):
        super(PHID, self).__init__()

        self.num_features = num_features

        self.DWT = DWTForward(J=dwt, wave='haar').cuda() 
        self.IDWT = DWTInverse(wave='haar').cuda()
        
        self.Conv_L = LAT(in_channels,num_features=self.num_features//2)
        self.Conv_H = HAT(in_channels*3,num_features=self.num_features)
        
        self.ResB = SCRB(self.num_features,in_channels,out_channels)

    def forward(self, x):
        xl,xh = self.DWT(x.squeeze(1))

        xh = [xh[i].transpose(2, 1) for i in range(len(xh))]
        yl = self.Conv_L(xl.unsqueeze(1)).squeeze(1)
        for i in reversed(range(len(xh))):
            yh = [self.Conv_H(xh[i],yl.unsqueeze(1)).transpose(2,1)]
            yl = self.IDWT((yl,yh))
            yl = self.ResB(yl)

        return yl.unsqueeze(1)