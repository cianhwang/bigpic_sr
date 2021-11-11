import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

def calc_ssim(img1, img2):
    SSIM = 0.0
    for i in range(img1.size(0)):
        a, b = img1.cpu().numpy()[i, 0], img2.cpu().numpy()[i, 0]
        SSIM += ssim(a, b, data_range=b.max()-b.min())
    return SSIM/img1.size(0)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
