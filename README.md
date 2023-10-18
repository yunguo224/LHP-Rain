# From Sky to the Ground: A Large-scale Benchmark and Simple Baseline Towards Real Rain Removal (ICCV 2023)
Yun Guo^, Xueyao Xiao^, <a href='https://owuchangyuo.github.io/'>Yi Chang*</a>, Shumin Deng, <a href='[https://owuchangyuo.github.io/](http://faculty.hust.edu.cn/yanluxin/zh_CN/)'>Luxin Yan</a>

Paper link: [[arxiv]](https://arxiv.org/abs/2308.03867) [[ICCV]](https://openaccess.thecvf.com/content/ICCV2023/html/Guo_From_Sky_to_the_Ground_A_Large-scale_Benchmark_and_Simple_ICCV_2023_paper.html)

## Project website: [[link]](https://yunguo224.github.io/LHP-Rain.github.io/) (Benchmark available now!)

<hr>
<i>Learning-based image deraining methods have made great progress. However, the lack of large-scale high-quality paired training samples is the main bottleneck to hamper the real image deraining (RID). To address this dilemma and advance RID, we construct a Large-scale High-quality Paired real rain benchmark (LHP-Rain), including 3000 video sequences with 1 million high-resolution (1920*1080) frame pairs. The advantages of the proposed dataset over the existing ones are three-fold: rain with higher-diversity and larger-scale, image with higher-resolution and higher quality ground-truth. Specifically, the real rains in LHP-Rain not only contain the classical rain streak/veiling/occlusion in the sky, but also the splashing on the ground overlooked by deraining community. Moreover, we propose a novel robust low-rank tensor recovery model to generate the GT with better separating the static background from the dynamic rain. In addition, we design a simple transformer-based single image deraining baseline, which simultaneously utilize the self-attention and cross-layer attention within the image and rain layer with discriminative feature representation. Extensive experiments verify the superiority of the proposed dataset and deraining method over state-of-the-art.</i>
<hr>
<img src="img/Figure1-examples.png" width="960" alt="demo">

## Benchmark Download
We provide full version, simple version and high-level annotations of LHP-Rain. The benchmark has been updated in [Project website](https://yunguo224.github.io/LHP-Rain.github.io/).

## Package dependencies
The project is built with PyTorch 1.9.0, Python3.7, CUDA11.1. For package dependencies, you can install them by:
```bash
pip install -r requirements.txt
```

## Training
To train SCD-Former, you can begin the training by:
```bash
python train/train_derain.py --arch Uformer_B --batch_size 8 --gpu '0,1' --train_ps 256 --train_dir ./train --val_ps 256 --val_dir ./test --env _derain --nepoch 3000 --checkpoint 500 --warmup
```

## Evaluation
To evaluate SCD-Former, you can run:

```sh
python test_derain.py
```

## Citation
If you find this project useful in your research, please consider citing:
```
@InProceedings{Guo_2023_ICCV,
    author    = {Guo, Yun and Xiao, Xueyao and Chang, Yi and Deng, Shumin and Yan, Luxin},
    title     = {From Sky to the Ground: A Large-scale Benchmark and Simple Baseline Towards Real Rain Removal},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {12097-12107}
}
```
## Acknowledgement
The code of SCD-Former is based on [Uformer](https://github.com/ZhendongWang6/Uformer).
## Contact
Please contact us if there is any question or suggestion(Yun Guo guoyun@hust.edu.cn, Yi Chang yichang@hust.edu.cn).
