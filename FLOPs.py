import torch
from thop import profile


Note=open('./flops_record.txt',mode='w')
Flops_List = []


### -------> SPNet <------- ###
from SPNet.model import SPNet

net = SPNet(32,50).cuda()
input, depth = torch.randn(1, 3, 256, 256).cuda(), torch.randn(1, 1, 256, 256).cuda()
flops, _ = profile(net, inputs=(input, depth,))
print('SPNet FLOPs = ' + str(flops/1000**3) + 'G')
Flops_List.append('SPNet FLOPs = ' + str(flops/1000**3) + 'G')



### -------> HAINet <------- ###
from HAINet.HAI_models import HAIMNet_VGG

net = HAIMNet_VGG().cuda()
input, depth = torch.randn(1, 3, 256, 256).cuda(), torch.randn(1, 1, 256, 256).cuda()
flops, _ = profile(net, inputs=(input, depth,))
print('HAINet FLOPs = ' + str(flops/1000**3) + 'G')
Flops_List.append('HAINet FLOPs = ' + str(flops/1000**3) + 'G')



### -------> DCF <------- ###
from DCF.DCF_models import DCF_VGG
from DCF.DCF_ResNet_models import DCF_ResNet
from DCF.fusion import fusion
from DCF.depth_calibration_models import discriminator, depth_estimator

model_rgb = DCF_ResNet().cuda()
model_depth = DCF_ResNet().cuda()
model = fusion().cuda()
model_discriminator = discriminator(n_class=2).cuda()
model_estimator = depth_estimator().cuda()

input, depth = torch.randn(1, 3, 256, 256).cuda(), torch.randn(1, 1, 256, 256).cuda()

_, res_r,x3_r,x4_r,x5_r = model_rgb(input)
# depth calibration
score= model_discriminator(depth)
score = torch.softmax(score,dim=1)
x3_, x4_, x5_ = x3_r.detach(), x4_r.detach(), x5_r.detach()
pred_depth = model_estimator(input,x3_, x4_, x5_)
depth_calibrated = torch.mul(depth, score[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 256, 256)) \
                   + torch.mul(pred_depth, score[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 256, 256))
depth_calibrated = torch.cat([depth_calibrated, depth_calibrated, depth_calibrated], dim=1)
# Depth Stream
_, res_d,x3_d,x4_d,x5_d = model_depth(depth_calibrated)
# Fusion Stream (CRM)
_,res,_,_,_ = model(x3_r,x4_r,x5_r,x3_d,x4_d,x5_d)

flops_rgb,_ = profile(model_rgb, (input,))
flops_discriminator,_ = profile(model_discriminator, (depth,))
flops_estimator,_ = profile(model_estimator,(input,x3_, x4_, x5_,))
flops_depth,_ = profile(model_depth,(depth_calibrated,))
flops_fusion,_ = profile(model,(x3_r,x4_r,x5_r,x3_d,x4_d,x5_d,))
print("DCF FLOPs = " + str((flops_rgb/1000**3)+(flops_depth/1000**3)+(flops_fusion/1000**3)+(flops_discriminator/1000**3)+(flops_estimator/1000**3)) + 'G')
Flops_List.append("DCF FLOPs = " + str((flops_rgb/1000**3)+(flops_depth/1000**3)+(flops_fusion/1000**3)+(flops_discriminator/1000**3)+(flops_estimator/1000**3)) + 'G')



### -------> CLNet <------- ###
from CLNet.ResNet_models_combine import Saliency_feat_endecoder

net = Saliency_feat_endecoder(channel=64).cuda()
input, depth = torch.randn(1, 3, 256, 256).cuda(), torch.randn(1, 3, 256, 256).cuda()
flops, _ = profile(net, inputs=(input, depth,))
print('CLNet FLOPs = ' + str(flops/1000**3) + 'G')
Flops_List.append('CLNet FLOPs = ' + str(flops/1000**3) + 'G')



### -------> MMNet <------- ###
from MMNet.MMNet import RGBres2net50,Depthres2net50,FusionNet

rgb_net = RGBres2net50().cuda()
depth_net = Depthres2net50().cuda()
fusion_net = FusionNet().cuda()

input, depth = torch.randn(1, 3, 256, 256).cuda(), torch.randn(1, 1, 256, 256).cuda()
n, c, h, w = input.size()
depth = depth.view(n, h, w, 1).repeat(1, 1, 1, c)
depth = depth.transpose(3, 1)
depth = depth.transpose(3, 2)
R1,R2,R3,R4,R5 = rgb_net(input)
D1,D2,D3,D4,D5 = depth_net(depth)
outputs_all = fusion_net(R1,R2,R3,R4,R5,D1,D2,D3,D4,D5)
flops_rgb,_ = profile(rgb_net, (input,))
flops_depth,_ = profile(depth_net, (depth,))
flops_fusion,_ = profile(fusion_net,(R1,R2,R3,R4,R5,D1,D2,D3,D4,D5,))
print("MMNet FLOPs = " + str((flops_rgb/1000**3)+(flops_depth/1000**3)+(flops_fusion/1000**3)))
Flops_List.append("MMNet FLOPs = " + str((flops_rgb/1000**3)+(flops_depth/1000**3)+(flops_fusion/1000**3)) + 'G')



### -------> SPSN <------- ###
from SPSN.model import MyModel, weights_init

net = MyModel().cuda()
input, depth = torch.randn(1, 3, 256, 256).cuda(), torch.randn(1, 3, 256, 256).cuda()
ss_map, depth_ss_map = torch.randn(1, 100, 256, 256).cuda(), torch.randn(1, 100, 256, 256).cuda()
flops, _ = profile(net, inputs=(input, depth, ss_map, depth_ss_map))
print('SPSN FLOPs = ' + str(flops/1000**3) + 'G')
Flops_List.append('SPSN FLOPs = ' + str(flops/1000**3) + 'G')



# ### --------> BTSNet <--------###
from BTSNet.BTSNet import BTSNet

net = BTSNet(nInputChannels=3, n_classes=1, os=16,).cuda()
input, depth = torch.randn(1, 3, 256, 256).cuda(), torch.randn(1, 3, 256, 256).cuda()
flops, _ = profile(net, inputs=(input, depth,))
print('BTSNet FLOPs = ' + str(flops/1000**3) + 'G')
Flops_List.append('BTSNet FLOPs = ' + str(flops/1000**3) + 'G')



# ### --------> CDNet <--------###
from CDNet import network

net = network.Net().cuda()
input, depth = torch.randn(1, 3, 256, 256).cuda(), torch.randn(1, 1, 256, 256).cuda()
flops, _ = profile(net, inputs=(input, depth,))
print('CDNet FLOPs = ' + str(flops/1000**3) + 'G')
Flops_List.append('CDNet FLOPs = ' + str(flops/1000**3) + 'G')



### --------> CFIDNet <--------###
from CFIDNet.model import Net
import argparse

parser = argparse.ArgumentParser()
# Architecture settings
parser.add_argument('--ratio', type=int, default=8)
parser.add_argument('--branches', type=int, default=4)  # multi-scale features branches
parser.add_argument('--channels', type=int, default=64)
config = parser.parse_args()

net = Net(config).cuda()
## Note: Calculated padded input size per channel: (4 x 4). Kernel size: (5 x 5). Kernel size can't be greater than actual input size
## increase the tensor scale
input, depth = torch.randn(1, 3, 320, 320).cuda(), torch.randn(1, 3, 320, 320).cuda()
flops, _ = profile(net, inputs=(input, depth,))
print('CFIDNet FLOPs = ' + str(flops/1000**3) + 'G')
Flops_List.append('CFIDNet FLOPs = ' + str(flops/1000**3) + 'G')



### --------> DANet <--------###
from DANet.DANet import RGBD_sal

net = RGBD_sal().cuda()
input, depth = torch.randn(1, 3, 256, 256).cuda(), torch.randn(1, 1, 256, 256).cuda()
flops, _ = profile(net, inputs=(input, depth,))
print('DANet FLOPs = ' + str(flops/1000**3) + 'G')
Flops_List.append('DANet FLOPs = ' + str(flops/1000**3) + 'G')



### --------> DFMNet <--------###
from DFMNet.net import DFMNet

net = DFMNet().cuda()
input, depth = torch.randn(1, 3, 256, 256).cuda(), torch.randn(1, 1, 256, 256).cuda()
flops, _ = profile(net, inputs=(input, depth,))
print('DFMNet FLOPs = ' + str(flops/1000**3) + 'G')
Flops_List.append('DFMNet FLOPs = ' + str(flops/1000**3) + 'G')



### --------> MobileSal <--------###
from MobileSal.model import MobileSal

net = MobileSal().cuda()
input, depth = torch.randn(1, 3, 256, 256).cuda(), torch.randn(1, 1, 256, 256).cuda()
flops, _ = profile(net, inputs=(input, depth,))
print('MobileSal FLOPs = ' + str(flops/1000**3) + 'G')
Flops_List.append('MobileSal FLOPs = ' + str(flops/1000**3) + 'G')



### --------> SSF <--------###
from SSF.model import model_VGG

net = model_VGG().cuda()
input, depth = torch.randn(1, 3, 256, 256).cuda(), torch.randn(1, 1, 256, 256).cuda()
n,c, h, w = input.size()
depth1 = depth.view(n,h, w, 1).repeat(1,1, 1, c)
depth1 = depth1.transpose(3, 2)
depth1 = depth1.transpose(2, 1)
flops, _ = profile(net, inputs=(input, depth1, depth))
print('SSF FLOPs = ' + str(flops/1000**3) + 'G')
Flops_List.append('SSF FLOPs = ' + str(flops/1000**3) + 'G')



### -------> SwinNet <------- ###
## Note: AssertionError: Input image size (256*256) doesn't match model (384*384).
## increase the tensor scale
from SwinNet.Swin_Transformer import SwinTransformer,SwinNet

net = SwinNet().cuda()
input, depth = torch.randn(1, 3, 384, 384).cuda(), torch.randn(1, 3, 384, 384).cuda()
flops, _ = profile(net, inputs=(input, depth,))
print('SwinNet FLOPs = ' + str(flops/1000**3) + 'G')
Flops_List.append('SwinNet FLOPs = ' + str(flops/1000**3) + 'G')



### -------> MPDNet <------- ####
# Note: The MPDNet models will be released after the work is accepted



for line in Flops_List:
    Note.write(line+'\n')
Note.close()