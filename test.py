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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('--output-path', type=str, default='test')
    parser.add_argument('--model', type=str, default='EDSR')
    parser.add_argument('--f_num', type=int, default=16)
    parser.add_argument('--n_photon', type=int, default=100)
    
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.model == 'EDSR':
        model = EDSR()
    elif args.model == 'SRCNN':
        model = SRCNN()
    else:
        raise("model not recognized. Try EDSR or SRCNN")
    model = nn.DataParallel(model).to(device)

    #state_dict = model.state_dict()
    #for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
    #    if n in state_dict.keys():
    #        state_dict[n].copy_(p)
    #    else:
    #        raise KeyError(n)
    model.load_state_dict(torch.load(args.weights_file))

    model.eval()
    
    paths = sorted(glob.glob(args.image_file+'/*.png'))
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    with tqdm(total=len(paths)) as t:
        for path in paths:
            index = path[-8:-4]
            image = cv2.imread(path, 0)[:1024, :1024]
            cv2.imwrite('{}/{}_hr.png'.format(args.output_path, index), 
                        image)
            image = np.array(image).astype(np.float32)/255.
            y = Camera(f_num=args.f_num, n_photon=args.n_photon).forward(image)
            y_t = torch.from_numpy(y).float().to(device).unsqueeze(0).unsqueeze(0)

            image_t = torch.from_numpy(image).float().to(device).unsqueeze(0).unsqueeze(0)
            psnr = calc_psnr(image_t, y_t)
            ssim = calc_ssim(image_t, y_t)
            cv2.imwrite('{}/{}_degrad_{:.2f}_{:.4f}.png'.format(
                args.output_path, index,psnr,ssim), 
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
