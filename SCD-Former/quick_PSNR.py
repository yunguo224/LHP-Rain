import cv2
import os
# from skimage.measure import compare_psnr, compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import fnmatch
import numpy as np



# result_path = r'G:\Projects\CodeInSupUnsup\SPACompareMethod\SSIR'
# gt_path = r'G:\Dataset\Spatial_Real_Dataset\Testing\real_test_1000\train\Bs'

result_path = r'/mnt/data/yeyuntong/Projects/Transformer/Uformer/results/UG2/result/'
gt_path = r'/mnt/data/yeyuntong/Projects/Datasets/WeatherStream/test/gt/'
aver_psnr = 0.
aver_ssim = 0.
img_list = os.listdir(result_path)
count = 1
try:
  for name in img_list:
      pred_Bs = cv2.imread(os.path.join(result_path, name))
      Bs = cv2.imread(os.path.join(gt_path, name.replace('PNG','png')))
      print(pred_Bs.shape)
      print(Bs.shape)
      tmp_psnr = compare_psnr(pred_Bs, Bs)
      tmp_ssim = compare_ssim(pred_Bs, Bs, multichannel=True)
      aver_psnr = (aver_psnr * count + tmp_psnr) / (count + 1)
      aver_ssim = (aver_ssim * count + tmp_ssim) / (count + 1)
      count += 1
      print(tmp_ssim, tmp_psnr)
except:
  pass

print('average PSNR: ', aver_psnr)
print('average SSIM: ', aver_ssim)
