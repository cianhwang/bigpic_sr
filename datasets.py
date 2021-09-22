import h5py
import numpy as np
from torch.utils.data import Dataset


import cv2
from scipy.special import j1
import glob
import torch.nn.functional as F
import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from skimage.measure import block_reduce

from kernels import Kernels
import argparse

        
class Camera:
    def __init__(self, lam=0.633e-6, f_num=16, n_photon=1e2, p=6.6e-6, unit=0.1e-6, kernel='jinc'):
        k_r = int(5*f_num*lam/unit)
        H = Kernels(lam=lam, f_num=f_num, unit=unit, k_r = k_r).select_kernel(kernel)
        block_size = int(p/unit)//2*2+1
        size = ((k_r*2+1)//block_size//2*2-1)*block_size
        bleed = (k_r*2+1-size)//2
        H_crop = H[bleed:-bleed, bleed:-bleed]
        H = block_reduce(H_crop, block_size=(block_size, block_size), func=np.sum)
        H = H/H.sum()
        assert H.shape[0]%2 == 1
        self.k_r = H.shape[0]//2
        
        self.n_photon = n_photon
        self.H_t = torch.from_numpy(H).float().unsqueeze(0).unsqueeze(0)
        
    def jinc(self,rho):
        f = j1(rho)/rho
        f[rho == 0] = 0.5
        return f
    
    def forward(self, img):
        img_t = torch.from_numpy(img).float()
        img_t = F.pad(img_t, (self.k_r, self.k_r, self.k_r, self.k_r)).unsqueeze(0).unsqueeze(0)
        blurry_img = F.conv2d(img_t, self.H_t).squeeze().numpy()
        noisy_img = np.random.poisson(blurry_img * self.n_photon)
        img_sensor = (noisy_img).astype(np.float)
        norm_img_sensor = img_sensor/self.n_photon
        return norm_img_sensor   
            
class Trainset(Dataset):
    def __init__(self, args, patch_size=256):
        self.files = sorted(glob.glob(args.train_file + '/*.png')) #[:8000]
        self.camera = []
        assert len(args.kernel.split(",")) == 1 or len(args.f_num.split(",")) == 1
        assert len(args.kernel.split(","))*len(args.f_num.split(",")) == args.num_channels
        for kernel in args.kernel.split(","):
            for f_num in args.f_num.split(","):
                self.camera.append(Camera(f_num=int(f_num), n_photon=args.n_photon, kernel=kernel))
        self.transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.RandomCrop(patch_size+256),
                            transforms.RandomHorizontalFlip()
                        ])
    
    def __getitem__(self, idx):
        gt = cv2.imread(self.files[idx], 0)
        gt = np.array(self.transform(gt))/255.0 #/160.0
        imgs = []
        for camera in self.camera:
            imgs.append(camera.forward(gt))
        img = np.stack(imgs,axis=0)
        edge = 128 
        gt_t = torch.from_numpy(gt).float().unsqueeze(0)[:, edge:-edge, edge:-edge]
        img_t =  torch.from_numpy(img).float()[:, edge:-edge, edge:-edge]
        return img_t, gt_t

    def __len__(self):
        return len(self.files)
    
class Evalset(Dataset):
    def __init__(self, args, patch_size=512):
        self.files = sorted(glob.glob(args.eval_file + '/*.png')) #[8000:]
        self.camera = []
        assert len(args.kernel.split(",")) == 1 or len(args.f_num.split(",")) == 1
        assert len(args.kernel.split(","))*len(args.f_num.split(",")) == args.num_channels
        for kernel in args.kernel.split(","):
            for f_num in args.f_num.split(","):
                self.camera.append(Camera(f_num=int(f_num), n_photon=args.n_photon, kernel=kernel))
        self.transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.CenterCrop(patch_size+256)
                        ])
    
    def __getitem__(self, idx):
        gt = cv2.imread(self.files[idx], 0)
        gt = np.array(self.transform(gt))/255.0 #/160.0
        imgs = []
        for camera in self.camera:
            imgs.append(camera.forward(gt))
        img = np.stack(imgs,axis=0)
        edge = 128
        gt_t = torch.from_numpy(gt).float().unsqueeze(0)[:, edge:-edge, edge:-edge]
        img_t =  torch.from_numpy(img).float()[:, edge:-edge, edge:-edge]
        return img_t, gt_t

    def __len__(self):
        return len(self.files)
    
def print_stat(narray, narray_name = "array"):
    print(narray_name, "shape: ", narray.shape, "dtype:", narray.dtype)
    arr = narray.flatten()
    print(narray_name , "stat: max: {}, min: {}, mean: {}, std: {}".format(arr.max(), arr.min(), arr.mean(), arr.std()))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--n_photon', type=int, default=1000)
    parser.add_argument('--f_num', type=str, default="48")
    parser.add_argument('--kernel', type=str, default="jinc")
    args = parser.parse_args()
    trainset = Trainset(args)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
    valset = Evalset(args, 1024)
    valloader = torch.utils.data.DataLoader(dataset=valset, batch_size=1, shuffle=True)
    dataiter = iter(valloader)
    images, labels = dataiter.next()
    print_stat(images, "images")
    print_stat(labels, "labels")
    for i in range(images.size(1)):
        plt.imshow(images[0, i].numpy(), cmap='gray')
        plt.show()
    plt.imshow(labels[0, 0].numpy(), cmap='gray')
    plt.show()
