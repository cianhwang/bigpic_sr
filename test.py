import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2

from models import SRCNN,EDSR
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr
from datasets import Camera


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-file', type=str, required=True)
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
    model = model.to(device)

    #state_dict = model.state_dict()
    #for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
    #    if n in state_dict.keys():
    #        state_dict[n].copy_(p)
    #    else:
    #        raise KeyError(n)
    model.load_state_dict(torch.load(args.weights_file))

    model.eval()

    image = cv2.imread(args.image_file, 0)
    cv2.imwrite('test/hr.png', image)
    image = np.array(image).astype(np.float32)/255.
    y = Camera(f_num=args.f_num, n_photon=args.n_photon).forward(image)
    cv2.imwrite('test/degrade.png', np.clip(y*255.0, 0.0, 255.0).astype(np.uint8))
    y = torch.from_numpy(y).float().to(device)
    y = y.unsqueeze(0).unsqueeze(0)

    image_t = torch.from_numpy(image).float().to(device).unsqueeze(0).unsqueeze(0)
    psnr = calc_psnr(image_t, y)
    print('Orig PSNR: {:.2f}'.format(psnr))

    with torch.no_grad():
        preds = model(y).clamp(0.0, 1.0)

    psnr = calc_psnr(image_t, preds)
    print('Pred PSNR: {:.2f}'.format(psnr))

    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    output = np.clip(preds, 0.0, 255.0).astype(np.uint8)
    cv2.imwrite('test/pred.png', output)
