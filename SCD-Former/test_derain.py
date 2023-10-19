import numpy as np
import os,sys
import argparse
from tqdm import tqdm
from einops import rearrange, repeat

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from ptflops import get_model_complexity_info

dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'../dataset/'))
sys.path.append(os.path.join(dir_name,'..'))

import scipy.io as sio
import utils
import math
# from model import UNet,Uformer
from dataset.dataset_derain import *
from skimage import img_as_float32, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
import cv2
from model import UNet,ISA,RSA

def net_process(model_R,model_B, image, flip=False):
    input = torch.from_numpy(image.transpose((2, 0, 1))).float()
    input = input.unsqueeze(0).cuda()
    if flip:
        input = torch.cat([input, input.flip(3)], 0)
    with torch.no_grad():
        output_R, list = model_R(input)
        output = model_B(input,list)
    _, _, h_i, w_i = input.shape
    _, _, h_o, w_o = output.shape
    if flip:
        output = (output[0] + output[1].flip(2)) / 2
    else:
        output = output[0]

    output = torch.clamp(output, 0, 1).data.cpu().numpy().squeeze().transpose((1, 2, 0))
    return output


parser = argparse.ArgumentParser(description='Image denoising evaluation on SIDD')
parser.add_argument('--input_dir', default='/mnt/data/yeyuntong/Projects/Transformer/Uformer/liyi_img',
    type=str, help='Directory of validation images')
parser.add_argument('--nir_dir', default='../dataset/Adverse_Multimodal_Dataset_v1/RainFree_Denoised/',
    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='/mnt/data/yeyuntong/Projects/Transformer/Uformer/Rainformer_results/real_cases/',
    type=str, help='Directory for results')
parser.add_argument('--weights', default='./mnt/data/yeyuntong/Projects/Transformer/Uformer/logs/SPA/models/',
    type=str, help='Path to weights')
parser.add_argument('--gpus', default='3', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--arch', default='Uformer_B', type=str, help='arch')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
parser.add_argument('--embed_dim', type=int, default=32, help='number of data loading workers')    
parser.add_argument('--win_size', type=int, default=8, help='number of data loading workers')
parser.add_argument('--token_projection', type=str,default='linear', help='linear/conv token projection')
parser.add_argument('--token_mlp', type=str,default='leff', help='ffn/leff token mlp')
parser.add_argument('--dd_in', type=int, default=3, help='dd_in')

# args for vit
parser.add_argument('--vit_dim', type=int, default=256, help='vit hidden_dim')
parser.add_argument('--vit_depth', type=int, default=12, help='vit depth')
parser.add_argument('--vit_nheads', type=int, default=8, help='vit hidden_dim')
parser.add_argument('--vit_mlp_dim', type=int, default=512, help='vit mlp_dim')
parser.add_argument('--vit_patch_size', type=int, default=16, help='vit patch_size')
parser.add_argument('--global_skip', action='store_true', default=False, help='global skip connection')
parser.add_argument('--local_skip', action='store_true', default=False, help='local skip connection')
parser.add_argument('--vit_share', action='store_true', default=False, help='share vit module')

parser.add_argument('--train_ps', type=int, default=128, help='patch size of training sample')
args = parser.parse_args()


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

# if args.save_images:
result_dir_img = os.path.join(args.result_dir, 'result')
utils.mkdir(result_dir_img)

visualization_dir_img = os.path.join(args.result_dir, 'visualization')
utils.mkdir(visualization_dir_img)

img_options_val = {'val_h': 1536, 'val_w': 2048}
test_dataset = get_validation_deblur_data(args.input_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

model_restoration = ISA(img_size=args.train_ps, embed_dim=args.embed_dim, win_size=8, token_projection='linear',
                              token_mlp='leff', modulator=True)
model_restoration_R = RSA(img_size=args.train_ps, embed_dim=args.embed_dim, win_size=8, token_projection='linear',
                                token_mlp='leff', modulator=True)
model_restoration = torch.nn.DataParallel(model_restoration)
model_restoration_R = torch.nn.DataParallel(model_restoration_R)
utils.load_checkpoint(model_restoration,'/mnt/data/yeyuntong/Projects/Transformer/Uformer/logs/SPA/models/model_latest.pth')
utils.load_checkpoint(model_restoration_R,'/mnt/data/yeyuntong/Projects/Transformer/Uformer/logs/SPA/models/model_R_best.pth')
print("===>Testing using weights: ", args.weights)

model_restoration.cuda()
model_restoration.eval()
model_restoration_R.cuda()
model_restoration_R.eval()
model_restoration = torch.nn.DataParallel(model_restoration)
model_restoration_R = torch.nn.DataParallel(model_restoration_R)

test_patch_size = 128
stride_rate=1/2

with torch.no_grad():
    psnr_val_derain = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        target = data_test[0].cuda()
        input_ = data_test[1].cuda()
        filenames = data_test[2]

        input_numpy = np.transpose(input_.squeeze(0).cpu().numpy(), [1, 2, 0])
        ori_h, ori_w, _ = input_numpy.shape

        if ori_h > test_patch_size or ori_w > test_patch_size:
            pad_h = max(test_patch_size - ori_h, 0)
            pad_w = max(test_patch_size - ori_w, 0)
            pad_h_half = int(pad_h / 2)
            pad_w_half = int(pad_w / 2)
            if pad_h > 0 or pad_w > 0:
                input_numpy = cv2.copyMakeBorder(input_numpy, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                                 cv2.BORDER_CONSTANT)
            new_h, new_w, _ = input_numpy.shape
            stride_h = int(np.ceil(test_patch_size * stride_rate))
            stride_w = int(np.ceil(test_patch_size * stride_rate))
            grid_h = int(np.ceil(float(new_h - test_patch_size) / stride_h) + 1)
            grid_w = int(np.ceil(float(new_w - test_patch_size) / stride_w) + 1)
            rgb_restored = np.zeros((new_h, new_w, 3), dtype=float)
            count_crop = np.zeros((new_h, new_w), dtype=float)
            for index_h in range(0, grid_h):
                for index_w in range(0, grid_w):
                    s_h = index_h * stride_h
                    e_h = min(s_h + test_patch_size, new_h)
                    s_h = e_h - test_patch_size
                    s_w = index_w * stride_w
                    e_w = min(s_w + test_patch_size, new_w)
                    s_w = e_w - test_patch_size
                    image_crop = input_numpy[s_h:e_h, s_w:e_w].copy()
                    count_crop[s_h:e_h, s_w:e_w] += 1
                    rgb_restored[s_h:e_h, s_w:e_w, :] += net_process(model_restoration_R,model_restoration, image_crop)
            rgb_restored /= np.expand_dims(count_crop, 2)
            rgb_restored = rgb_restored[pad_h_half:pad_h_half + ori_h, pad_w_half:pad_w_half + ori_w]
            prediction = cv2.resize(rgb_restored, (ori_w, ori_h), interpolation=cv2.INTER_LINEAR)
        else:
            r_restored,list = model_restoration_R(input_)
            rgb_restored = model_restoration(input_,list)
            rgb_restored = torch.clamp(rgb_restored, 0, 1).data.cpu().numpy().squeeze().transpose((1, 2, 0))

        rgb_restored_img = img_as_ubyte(rgb_restored)
        target_img = img_as_ubyte(target.data.cpu().numpy().squeeze().transpose((1, 2, 0)))
        input_img = img_as_ubyte(input_.data.cpu().numpy().squeeze().transpose((1, 2, 0)))
        visualization = np.concatenate([input_img, rgb_restored_img, target_img], axis=0)
        print(filenames[0])
        utils.save_img(os.path.join(args.result_dir, 'result/', filenames[0] + '.png'), rgb_restored_img)
        # utils.save_img(os.path.join(args.result_dir, 'visualization/', filenames[0] + '.png'), visualization)


