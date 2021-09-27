import numpy as np
from scipy import interpolate as interp
import matplotlib.pyplot as plt

def print_stat(narray, narray_name = 'Array'):
    arr = narray.flatten()
    print(narray_name + " stat: max: {}, min: {}, mean: {}, std: {}".format(arr.max(), arr.min(), arr.mean(), arr.std()))

def read_raw(fileName, frame = 1, rows = 2174, runL = 3968):

    rIm = np.fromfile(fileName, dtype = np.dtype('i2'))

    rIm = np.reshape(rIm, (frame, rows, runL))

    return rIm[..., :3864][:, ::-1, ::-1]

def demosaic(rawImage):  
    H, W = rawImage.shape
    red=rawImage[0:H:2,0:W:2]
    redU=interp.RectBivariateSpline(np.arange(0,H//2),np.arange(0,W//2),red)
    red=redU(np.arange(0,H//2,.5),np.arange(0,W//2,.5))
    blue=rawImage[1:H:2,1:W:2]
    blueU=interp.RectBivariateSpline(np.arange(0,H//2),np.arange(0,W//2),blue)
    blue=blueU(np.arange(0,H//2,.5),np.arange(0,W//2,.5))
    green=rawImage;
    for er in range(0,H-1,2):
        for odr in range(0,W-2,2):
            green[er,odr]=np.mean([green[er-1,odr], green[er,odr-1],green[er+1,odr], green[er,odr+1]])
    for er in range(1,H-1,2):
        for odr in range(1,W-2,2):
            green[er,odr]=np.mean([green[er-1,odr], green[er,odr-1],green[er+1,odr], green[er,odr+1]])
    greenU=interp.RectBivariateSpline(np.arange(0,H),np.arange(0,W-2),green[0:H,0:W-2])
    green=greenU(np.arange(0,H,1),np.arange(0,W,1))
    imageOut=np.dstack((red,green,blue))
    imageOut=255*imageOut/np.max(imageOut)
    imageOut=imageOut.astype(np.uint8)
    return imageOut

def adjustColor(inputImage,rc=1,bc=1,gc=1,gain=1,gamma = 0.5, contrast=1):
    fI=inputImage.astype(np.float)
    fI[:,:,0]=rc*fI[:,:,0]
    fI[:,:,1]=gc*fI[:,:,1]
    fI[:,:,2]=bc*fI[:,:,2]#
    fI=gain*fI/np.max(fI)
    fI = np.power(fI, 1/gamma)
    fI=255*np.tanh(contrast*fI)
    fI=fI.astype(np.uint8)
    return fI

def plot_histogram(Im, bit_length = 8):
    if len(Im.shape) == 3: # rgb image
        red, green, blue = Im[..., 0], Im[..., 1], Im[..., 2]
    elif len(Im.shape) == 2: # raw image RGGB
        red, g1, g2, blue = Im[::2, ::2], Im[1::2, ::2], Im[::2, 1::2], Im[1::2, 1::2]
        green = [g1, g2]
    else:
        raise Exception("wrong dimension.")
        
    max_value = 2**bit_length
    
    histr,bins=np.histogram(red,np.arange(0,max_value))
    histg,bins=np.histogram(green,np.arange(0,max_value))
    histb,bins=np.histogram(blue,np.arange(0,max_value))
    
    plt.plot(np.arange(0, max_value-1),histr,'r', np.arange(0,max_value-1),histg,'g', np.arange(0,max_value-1),histb, 'b')
    


