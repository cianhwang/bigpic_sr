import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from models import SRCNN, EDSR
from datasets import Evalset, Camera
from utils import AverageMeter, calc_psnr, calc_ssim

import glob


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--is_pred', action='store_true')
    parser.add_argument('--model', type=str, default='EDSR')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=123) 
    parser.add_argument('--num-channels', type=int, default=1)

    parser.add_argument('--lam', type=float, default=0.633e-6)
    parser.add_argument('--n_photon', type=int, default=100)
    parser.add_argument('--f_num', type=str, default="32,48")
    parser.add_argument('--p', type=float, default=6.6e-6)
    parser.add_argument('--kernel', type=str, default="jinc")

    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    if args.model == 'EDSR':
        model = EDSR(num_channels=args.num_channels)
    elif args.model == 'SRCNN':
        model = SRCNN(num_channels=args.num_channels)
    else:
        raise("model not recognized. Try EDSR or SRCNN")
    model = nn.DataParallel(model).to(device)
    weight = glob.glob(args.model_path + '/*.pth')[0]
    model.load_state_dict(torch.load(weight))
    eval_dataset = Evalset(args,patch_size=1024)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    model.eval()
    epoch_psnr = AverageMeter()
    epoch_ssim = AverageMeter()
    with tqdm(total=len(eval_dataset)) as t:
        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)
            
            if args.is_pred:
                with torch.no_grad():
                    preds = model(inputs).clamp(0.0, 1.0)
            else:
                preds = inputs
                
            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))
            epoch_ssim.update(calc_ssim(labels, preds), len(inputs))
            t.set_postfix({'psnr':'{:.2f}'.format(epoch_psnr.avg), 'ssim':'{:.4f}'.format(epoch_ssim.avg)})
            t.update(len(inputs))

    print('{}x{}x{} psnr: {:.2f} | ssim: {:.4f}'.format(args.n_photon, args.f_num, args.kernel, epoch_psnr.avg, epoch_ssim.avg))
