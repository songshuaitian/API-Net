'''
    Enhance Feature Network --------> part 3
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class DRDB(nn.Module):
    def __init__(self, in_chans, growth_rate=32):
        super(DRDB, self).__init__()
        in_ch_ = in_chans#  3
        self.Dcov1 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate#35
        self.Dcov2 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate#67
        self.Dcov3 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate#99
        self.Dcov4 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate#131
        self.Dcov5 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate#163
        self.conv = nn.Conv2d(in_ch_, in_chans, 1, padding=0)

    def forward(self, x):
        x1 = self.Dcov1(x)
        x1 = F.relu(x1)
        x1 = torch.cat([x, x1], dim=1)
        x2 = self.Dcov2(x1)
        x2 = F.relu(x2)
        x2 = torch.cat([x1, x2], dim=1)
        x3 = self.Dcov3(x2)
        x3 = F.relu(x3)
        x3 = torch.cat([x2, x3], dim=1)
        x4 = self.Dcov4(x3)
        x4 = F.relu(x4)
        x4 = torch.cat([x3, x4], dim=1)
        x5 = self.Dcov5(x4)
        x5 = F.relu(x5)
        x5 = torch.cat([x4, x5], dim=1)#channel = 163
        x6 = self.conv(x5)#channel = 3
        out = x + F.relu(x6)
        return out
class CALayer(nn.Module):
    def __init__(self,dim):
        super(CALayer,self).__init__()
        self.GAP=nn.AdaptiveAvgPool2d(1)
        self.ca=nn.Sequential(
            nn.Conv2d(dim,dim//8,1,padding=0,bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//8,dim,1,padding=0,bias=True),
            nn.Sigmoid()
        )
    def forward(self,x):
        y=self.GAP(x)
        y=self.ca(y)
        return x * y


class SALayer(nn.Module):
    def __init__(self,dim):
        super(SALayer,self).__init__()
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.GMP = nn.AdaptiveMaxPool2d(1)#学会了再看一下
        self.sa = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, padding=0, bias=True),  # 卷积核啥的都得看
            nn.Sigmoid()
        )

    def forward(self,x):
        x1=self.GAP(x)
        x2=self.GMP(x)
        y=torch.cat([x1,x2],dim=1)
        y=self.sa(y)
        return x * y

class EFN_model(nn.Module):
    def __init__(self,in_chans,out_chans):
        super(EFN_model,self).__init__()
        # self.dim_x = 3#输入的维度
        # self.dim = 64
        self.drdb = DRDB(in_chans)
        self.conv1 = nn.Conv2d(in_chans,in_chans,3,padding=1,bias=True)
        self.ca = CALayer(in_chans)
        self.sa = SALayer(in_chans)
        self.conv2 = nn.Conv2d(in_chans*2,in_chans,3,padding=1,bias=True)
        self.conv3 = nn.Conv2d(in_chans*2, out_chans, 3, padding=1,bias=True)

    def forward(self,x):
        x1 = self.drdb(x)
        x2 = self.conv1(x1)
        x3 = self.ca(x2)
        x3 = self.sa(x3)
        x3 = torch.cat([x2,x3],dim=1)
        x4 = self.conv2(x3)
        x5 = torch.cat([x,x4],dim=1)
        x5 = self.conv3(x5)
        return x5


#
# model = EFN_model()
# image = torch.randn(1, 3, 256, 256)
# print(model(image))
#
#



# model = model(image)
# model = torch.reshape(model,(3,256,256))
# trans_PIL = torchvision.transforms.ToPILImage()
# img = trans_PIL(model)
# img.show()

