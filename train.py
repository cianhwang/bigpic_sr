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
from datasets import Trainset, Evalset, Camera
from utils import AverageMeter, calc_psnr, calc_ssim
from tensorboardX import SummaryWriter
import json

from losses import PerceptualLoss

def Trainer(args):
    #args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))
    args.outputs_dir = os.path.join(args.outputs_dir, '{}x{}'.format(args.n_photon, args.f_num))
    print('[*] Saving outputs to {}'.format(args.outputs_dir))
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)
        
    args.logs_dir = os.path.join(args.logs_dir, '{}x{}'.format(args.n_photon, args.f_num))
    print('[*] Saving tensorboard logs to {}'.format(args.logs_dir))
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    writer = SummaryWriter(args.logs_dir)
    logs_path = os.path.join(args.logs_dir, 'logs.json')
    with open(logs_path, 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    if args.model == 'EDSR':
        model = EDSR()
    elif args.model == 'SRCNN':
        model = SRCNN()
    else:
        raise("model not recognized. Try EDSR or SRCNN")
    model = nn.DataParallel(model).to(device)
    
    if args.criterion == 'mse':
        criterion = nn.MSELoss()
    elif args.criterion == 'l1':
        criterion = nn.L1Loss()
    elif args.criterion == 'l1+perceptual':
        criterion = PerceptualLoss(nn.L1Loss(), 0.06)
        criterion.initialize(nn.MSELoss())
    elif args.criterion == 'mse+perceptual':
        criterion = PerceptualLoss()
        criterion.initialize(nn.MSELoss())
       
    #optimizer = optim.Adam([
    #    {'params': model.conv1.parameters()},
    #    {'params': model.conv2.parameters()},
    #    {'params': model.conv3.parameters(), 'lr': args.lr * 0.1}
    #], lr=args.lr)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_dataset = Trainset(args.train_file, Camera(f_num = args.f_num, n_photon=args.n_photon, kernel=args.kernel))
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    eval_dataset = Evalset(args.eval_file, Camera(f_num = args.f_num, n_photon=args.n_photon, kernel=args.kernel))
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)
    
    best_psnr = 0.0
    for epoch in range(args.num_epochs):
        train(args, epoch, model, train_dataloader, criterion, optimizer, writer, device)
        best_psnr = validate(args, epoch, model, eval_dataloader, best_psnr, writer, device)
    
def train(args, epoch, model, train_dataloader, criterion, optimizer, writer, device):
    model.train()
    epoch_losses = AverageMeter()

    with tqdm(total=len(train_dataloader)*args.batch_size) as t:
        t.set_description('epoch {}/{}'.format(epoch+1, args.num_epochs))

        for data in train_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            preds = model(inputs)

            loss = criterion(preds, labels)

            epoch_losses.update(loss.item(), len(inputs))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
            t.update(len(inputs))
    writer.add_scalar('Stats/training_loss', epoch_losses.avg, epoch+1)
        
def validate(args, epoch, model, eval_dataloader, best_psnr, writer, device):
    model.eval()
    epoch_psnr = AverageMeter()
    epoch_ssim = AverageMeter()

    with tqdm(total=len(eval_dataloader)) as t:
        t.set_description('validate')
        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))
            epoch_ssim.update(calc_ssim(labels, preds), len(inputs))
            
            t.set_postfix({'psnr':'{:.2f}'.format(epoch_psnr.avg), 'ssim':'{:.4f}'.format(epoch_ssim.avg)})
            t.update(len(inputs))

    writer.add_scalar('Stats/eval_psnr', epoch_psnr.avg, epoch+1)
    writer.add_scalar('Stats/eval_ssim', epoch_ssim.avg, epoch+1)

    if epoch_psnr.avg > best_psnr:
        best_psnr = epoch_psnr.avg
        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'best.pth'))
    
    return best_psnr
              

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, default='outputs')
    parser.add_argument('--logs_dir', type=str, default='runs')
    #parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--model', type=str, default='EDSR')
    parser.add_argument('--criterion', type=str, default='mse')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=400)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    
    parser.add_argument('--n_photon', type=str, default="100,1000,10000,100000")
    parser.add_argument('--f_num', type=str, default="16,20,24")
    parser.add_argument('--kernel', type=str, default='jinc')
    args = parser.parse_args()
              
    n_photos = args.n_photon.split(',')
    f_nums = args.f_num.split(',')
              
    for n_photo in n_photos:
        for f_num in f_nums:
            args.n_photon = int(n_photo)
            args.f_num = int(f_num)
            Trainer(args)