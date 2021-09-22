import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1

def jinc(rho):
    f = j1(rho)/rho
    f[rho == 0] = 0.5
    return f

class Kernels:
    def __init__(self, lam = 0.633e-6, f_num = 48, unit = 0.1e-6, k_r = 10000):
        self.lam = lam
        self.f_num = f_num
        self.unit = unit
        self.k_r = k_r

    def H(self):
        X, Y = np.meshgrid((np.arange(0, self.k_r*2+1)-self.k_r)*self.unit, (np.arange(0, self.k_r*2+1)-self.k_r)*self.unit)
        return (2*jinc(np.pi/self.lam*(X**2+Y**2)**0.5/self.f_num))**2

    def gauss(self):
        sigma = 1.22*self.lam*self.f_num/3.0
        X, Y = np.meshgrid((np.arange(0, self.k_r*2+1)-self.k_r)*self.unit, (np.arange(0, self.k_r*2+1)-self.k_r)*self.unit)
        return np.exp(-(X**2 + Y**2)/(2*sigma**2))

    def H_dx(self):
        X, Y = np.meshgrid((np.arange(0, self.k_r*2+1)-self.k_r)*self.unit, (np.arange(0, self.k_r*2+2)-self.k_r)*self.unit)
        H = 2*jinc(np.pi/self.lam*(X**2+Y**2)**0.5/self.f_num)
        return (H[:-1] - H[1:])**2

    def H_dy(self):
        X, Y = np.meshgrid((np.arange(0, self.k_r*2+2)-self.k_r)*self.unit, (np.arange(0, self.k_r*2+1)-self.k_r)*self.unit)
        H = 2*jinc(np.pi/self.lam*(X**2+Y**2)**0.5/self.f_num)
        return (H[:,:-1] - H[:,1:])**2
  
    def H_d45(self):
        X, Y = np.meshgrid((np.arange(0, self.k_r*2+2)-self.k_r)*self.unit, (np.arange(0, self.k_r*2+2)-self.k_r)*self.unit)
        H = 2*jinc(np.pi/self.lam*(X**2+Y**2)**0.5/self.f_num)
        return (H[:-1,:-1] - H[1:,1:])**2
    
    def H_dx(self):
        X, Y = np.meshgrid((np.arange(0, self.k_r*2+1)-self.k_r)*self.unit, (np.arange(0, self.k_r*2+2)-self.k_r)*self.unit)
        H = 2*jinc(np.pi/self.lam*(X**2+Y**2)**0.5/self.f_num)
        return (H[:-1] - H[1:])**2

    def H_dy(self):
        X, Y = np.meshgrid((np.arange(0, self.k_r*2+2)-self.k_r)*self.unit, (np.arange(0, self.k_r*2+1)-self.k_r)*self.unit)
        H = 2*jinc(np.pi/self.lam*(X**2+Y**2)**0.5/self.f_num)
        return (H[:,:-1] - H[:,1:])**2

    def H_lap(self):
        X, Y = np.meshgrid((np.arange(0, self.k_r*2+3)-self.k_r)*self.unit, (np.arange(0, self.k_r*2+3)-self.k_r)*self.unit)
        H = 2*jinc(np.pi/self.lam*(X**2+Y**2)**0.5/self.f_num)
        H_dx = H[:-1] - H[1:]
        H_dy = H[:,:-1] - H[:,1:]
        H_dx2 = H_dx[:-1] - H_dx[1:]
        H_dy2 = H_dy[:,:-1] - H_dy[:,1:]
        return (H_dx2[:,:-2] + H_dy2[:-2])**2

    def select_kernel(self, name):
        if name == 'jinc':
            return self.H()
        elif name == 'gauss':
            return self.gauss()
        elif name == 'jinc_dx':
            return self.H_dx()
        elif name == 'jinc_dy':
            return self.H_dy()
        elif name == 'jinc_lap':
            return self.H_lap()
        else:
            raise TypeError("Unsupported kernel name", name)

if __name__ == '__main__':
    H = Kernels().select_kernel('jinc_lap')
    plt.imshow(H)
    plt.show()