import torch 
import thop
from models.resnet import *
from copy import deepcopy

if __name__ == '__main__':
    path = './checkpoint/sc_ckpt.pth'
    ckpt = torch.load(path)
    print('best acc: {:.2f}'.format(ckpt['acc']))
    model = ResNet50()
    im = torch.randn(1, 3, 32, 32) # input image in BCHW format
    flops, params = thop.profile(model, (im, ))
    print('params: {:.2f} parameters (M)'.format(params / 1E6))
    print('flops: {:.2f} GFLOPs'.format(flops / 1E9 * 2))