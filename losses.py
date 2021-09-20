import torchvision.models
import torch
from torch import nn

class PerceptualLoss():
    
    def __init__(self, aux_loss_fn=nn.MSELoss(), lam=0.006):
        self.aux_loss_fn = aux_loss_fn
        self.lam = lam

    def contentFunc(self):
        conv_3_3_layer = 14
        cnn = torchvision.models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        model = model.eval()
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        return model

    def initialize(self, loss):
        with torch.no_grad():
            self.criterion = loss
            self.contentFunc = self.contentFunc()

    def get_loss(self, fakeIm, realIm):
        if fakeIm.size(1) == 1:
            fakeIm = fakeIm.repeat(1, 3, 1, 1)
            realIm = realIm.repeat(1, 3, 1, 1)
        fakeIm = (fakeIm + 1) / 2.0
        realIm = (realIm + 1) / 2.0
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return self.lam*torch.mean(loss) + 0.5*self.aux_loss_fn(fakeIm, realIm)

    def __call__(self, fakeIm, realIm):
        return self.get_loss(fakeIm, realIm)