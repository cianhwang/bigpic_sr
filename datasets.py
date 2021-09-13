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


# class TrainDataset(Dataset):
#     def __init__(self, h5_file):
#         super(TrainDataset, self).__init__()
#         self.h5_file = h5_file

#     def __getitem__(self, idx):
#         with h5py.File(self.h5_file, 'r') as f:
#             return np.expand_dims(f['lr'][idx] / 255., 0), np.expand_dims(f['hr'][idx] / 255., 0)

#     def __len__(self):
#         with h5py.File(self.h5_file, 'r') as f:
#             return len(f['lr'])


# class EvalDataset(Dataset):
#     def __init__(self, h5_file):
#         super(EvalDataset, self).__init__()
#         self.h5_file = h5_file

#     def __getitem__(self, idx):
#         with h5py.File(self.h5_file, 'r') as f:
#             return np.expand_dims(f['lr'][str(idx)][:, :] / 255., 0), np.expand_dims(f['hr'][str(idx)][:, :] / 255., 0)

#     def __len__(self):
#         with h5py.File(self.h5_file, 'r') as f:
#             return len(f['lr'])
        
class Camera:
    def __init__(self, lam=0.633e-6, f_num=32, n_photon=1e2, p=3e-6):
        self.k_r = int(3*f_num*lam/p)
        X, Y = np.meshgrid((np.arange(0, self.k_r*2+1)-self.k_r)*p, (np.arange(0, self.k_r*2+1)-self.k_r)*p)
        H = (2*self.jinc(2*np.pi/lam*(X**2+Y**2)**0.5/f_num))**2
        H = H/H.sum()
        
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
    
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(256),
    transforms.RandomHorizontalFlip()
])
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(1024)
])
        
class Trainset(Dataset):
    def __init__(self, path, camera=Camera(), transform=train_transform):
        self.files = sorted(glob.glob(path + '/*.png')) #[:8000]
        self.camera = camera
        self.transform = transform
    
    def __getitem__(self, idx):
        gt = cv2.imread(self.files[0], 0)
        gt = np.array(self.transform(gt))/255.0 #/160.0
        img = self.camera.forward(gt)
        gt_t = torch.from_numpy(gt).float().unsqueeze(0)
        img_t =  torch.from_numpy(img).float().unsqueeze(0)
        return img_t, gt_t

    def __len__(self):
        return len(self.files)
    
class Evalset(Dataset):
    def __init__(self, path, camera=Camera(), transform=val_transform):
        self.files = sorted(glob.glob(path + '/*.png')) #[8000:]
        self.camera = camera
        self.transform = transform
    
    def __getitem__(self, idx):
        gt = cv2.imread(self.files[0], 0)
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
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    print_stat(images, "images")
    print_stat(labels, "labels")
    plt.imshow(images[0, 0].numpy(), cmap='gray')
    plt.show()
    plt.imshow(labels[0, 0].numpy(), cmap='gray')
    plt.show()
