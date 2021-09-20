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

        
class Camera:
    def __init__(self, lam=0.633e-6, f_num=16, n_photon=1e2, p=6.6e-6, unit=0.1e-6, kernel='jinc'):
        k_r = int(3.66*f_num*lam/unit)
        X, Y = np.meshgrid(
            (np.arange(0, k_r*2+1)-k_r)*unit, 
            (np.arange(0, k_r*2+1)-k_r)*unit
        )
        if kernel == 'jinc':
            H = (2*self.jinc(np.pi/lam*(X**2+Y**2)**0.5/f_num))**2
        elif kernel == 'gauss':
            sigma = 1.22*lam*f_num/3.0
            H = np.exp(-(X**2 + Y**2)/(2*sigma**2))
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
    def __init__(self, path, camera=Camera(), patch_size=256):
        self.files = sorted(glob.glob(path + '/*.png')) #[:8000]
        self.camera = camera
        self.transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.RandomCrop(256),
                            transforms.RandomHorizontalFlip()
                        ])
    
    def __getitem__(self, idx):
        gt = cv2.imread(self.files[idx], 0)
        gt = np.array(self.transform(gt))/255.0 #/160.0
        img = self.camera.forward(gt)
        gt_t = torch.from_numpy(gt).float().unsqueeze(0)
        img_t =  torch.from_numpy(img).float().unsqueeze(0)
        return img_t, gt_t

    def __len__(self):
        return len(self.files)
    
class Evalset(Dataset):
    def __init__(self, path, camera=Camera(), patch_size=512):
        self.files = sorted(glob.glob(path + '/*.png')) #[8000:]
        self.camera = camera
        self.transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.CenterCrop(patch_size)
                        ])
    
    def __getitem__(self, idx):
        gt = cv2.imread(self.files[idx], 0)
        gt = np.array(self.transform(gt))/255.0 #/160.0
        img = self.camera.forward(gt)
        gt_t = torch.from_numpy(gt).float().unsqueeze(0)
        img_t =  torch.from_numpy(img).float().unsqueeze(0)
        return img_t, gt_t

    def __len__(self):
        return len(self.files)
    
def print_stat(narray, narray_name = "array"):
    print(narray_name, "shape: ", narray.shape, "dtype:", narray.dtype)
    arr = narray.flatten()
    print(narray_name , "stat: max: {}, min: {}, mean: {}, std: {}".format(arr.max(), arr.min(), arr.mean(), arr.std()))
    
if __name__ == '__main__':
    camera = Camera()
    trainset = Trainset('/media/qian/7f6908d4-b97f-4a1e-ba90-d502c5308801/DIV2K_train_HR', camera)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
    valset = Evalset('/media/qian/7f6908d4-b97f-4a1e-ba90-d502c5308801/DIV2K_valid_HR', camera)
    valloader = torch.utils.data.DataLoader(dataset=valset, batch_size=1, shuffle=True)
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    print_stat(images, "images")
    print_stat(labels, "labels")
    plt.imshow(images[0, 0].numpy(), cmap='gray')
    plt.show()
    plt.imshow(labels[0, 0].numpy(), cmap='gray')
    plt.show()
