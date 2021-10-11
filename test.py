import argparse

import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2

from models import SRCNN,EDSR
from utils import calc_psnr, calc_ssim
from datasets import Camera
import glob
import os
from tqdm import tqdm
from utils import AverageMeter, calc_psnr, calc_ssim
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('--output-path', type=str, default='test')
    parser.add_argument('--model', type=str, default='EDSR')
    parser.add_argument('--num-channels', type=int, default=1)
    parser.add_argument('--n-post-blocks', type=int, default=0)
    
    parser.add_argument('--lam', type=float, default=0.633e-6)
    parser.add_argument('--n_photon', type=int, default=100)
    parser.add_argument('--f_num', type=str, default="32,48")
    parser.add_argument('--p', type=float, default=6.6e-6)
    parser.add_argument('--kernel', type=str, default="jinc")
    parser.add_argument('--scale', type=int, default=1)
    
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.model == 'EDSR':
        model = EDSR(num_channels=args.num_channels,scale=args.scale,n_post_blocks=args.n_post_blocks)
    elif args.model == 'SRCNN':
        assert args.scale==1
        model = SRCNN(num_channels=args.num_channels)
    else:
        raise("model not recognized. Try EDSR or SRCNN")
    model = nn.DataParallel(model).to(device)

    weight = glob.glob(args.weights_file+'/*.pth')[0]
    model.load_state_dict(torch.load(weight))

    model.eval()
    
    paths = sorted(glob.glob(args.image_file+'/*.png'))
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        
    cameras = []
    for kernel in args.kernel.split(","):
        for f_num in args.f_num.split(","):
            cameras.append(Camera(lam=args.lam, f_num=float(f_num), n_photon=args.n_photon, p=args.p, kernel=kernel,scale=args.scale))

    with tqdm(total=len(paths)) as t:
        for path in paths:
            index = path[-8:-4]
            image = cv2.imread(path, 0)[:1280, :1280]
            cv2.imwrite('{}/{}_hr.png'.format(args.output_path, index), 
                        image[128:-128, 128:-128])
            image = np.array(image).astype(np.float32)/255.
            ys = []
            for camera in cameras:
                y = camera.forward(image)[128//args.scale:-128//args.scale, 128//args.scale:-128//args.scale]
                ys.append(y)
            y = np.stack(ys,axis=0)
            y_t = torch.from_numpy(y).float().to(device).unsqueeze(0)
            image_t = torch.from_numpy(image[128:-128, 128:-128]).float().to(device).unsqueeze(0).unsqueeze(0)
            
            for i in range(y_t.size(1)):
                yy = y_t[:,i:i+1,:,:]
                psnr = calc_psnr(image_t, F.resize(yy, (1024, 1024), InterpolationMode.BICUBIC) )
                ssim = calc_ssim(image_t, F.resize(yy, (1024, 1024), InterpolationMode.BICUBIC) )
                y = yy.cpu().numpy().squeeze(0).squeeze(0)
                cv2.imwrite('{}/{}_degrad_{}x{}x{}_{:.2f}_{:.4f}.png'.format(
                    args.output_path, index,
                    cameras[i].f_num, cameras[i].n_photon, cameras[i].kernel,
                    psnr,ssim), 
                np.clip(y*255.0, 0.0, 255.0).astype(np.uint8))

            with torch.no_grad():
                pred = model(y_t).clamp(0.0, 1.0)

            psnr = calc_psnr(image_t, pred)
            ssim = calc_ssim(image_t, pred)
            pred = pred.cpu().numpy().squeeze(0).squeeze(0)

            output = np.clip(pred*255.0, 0.0, 255.0).astype(np.uint8)
            cv2.imwrite('{}/{}_pred_{:.2f}_{:.4f}.png'.format(
                args.output_path,index,psnr,ssim), 
                        output)
            t.update(1)
