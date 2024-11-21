
![legnet_arch](docs/legnet.png)

## This repository is the official implementation of "LEGNet: A Lightweight Edge-Gaussian Driven Backbone for Object Detection on Low-Quality Remote Sensing Images".
## Abstract
Remote sensing object detection (RSOD) presents unique challenges in computer vision due to the lower resolution and often degraded quality of aerial and satellite images. These limitations lead to blurred, incomplete, or unclear object object features, complicating detection tasks and impacting model robustness. To overcome these challenges, we propose LEGNet, a lightweight and effective backbone network designed to enhance feature representation for remote sensing images. LEGNet introduces a novel low-quality feature enhancement module that combines edge-based feature extraction with Gaussian modeling to sharpen object boundaries and handle feature uncertainty. Specifically, the model leverages the Scharr filter to preserve critical edge details, providing superior rotational invariance and positional accuracy. Furthermore, Gaussian convolution kernels refine the representations of objects with uncertain features by emphasizing salient features and effectively suppressing background noise. Extensive experiments on 4 standard RSOD benchmarks confirmed the effectiveness of LEGNet, demonstrating significant improvements in both detection accuracy and computational complexity.

## Introduction

The master branch is built on MMRotate which works with **PyTorch 1.6+**.

LEGNet backbone code is placed under mmrotate/models/backbones/, and the train/test configure files are placed under configs/legnet/ 

## Pretrained Weights of Backbones

Imagenet 300-epoch pre-trained LEGNet-Tiny backbone: [Download](https://github.com/lwCVer/LWGANet/releases/download/weights/lwganet_l0_e297.pth)

Imagenet 300-epoch pre-trained LEGNet-Small backbone: [Download](https://github.com/lwCVer/LWGANet/releases/download/weights/lwganet_l1_e239.pth)

## Results and models

DOTA1.0

|           Model            |  mAP  | Angle | training mode | Batch Size |                                     Configs                                      |                                                              Download                                                               |
|:--------------------------:|:-----:| :---: |---------------|:----------:|:--------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------:|
| LEGNet-Tiny (1024,1024,200) | 79.37 | le90  | single-scale  |    2\*4    | [orcnn_legnet_tiny_dota10_test_ss_e36.py](./configs/legnet/orcnn_legnet_tiny_dota10_test_ss_e36.py) |          [model](https://github.com/lwCVer/LWGANet/releases/download/weights/ORCNN_LWGANet_L2_fpn_le90_dota10_ss_e30.pth)           |
| LEGNet-Small (1024,1024,200) | 80.03 | le90  | single-scale  |    2\*4    | [orcnn_legnet_small_dota10_test_ss_e36.py](./configs/legnet/orcnn_legnet_small_dota10_test_ss_e36.py) |          [model](https://github.com/lwCVer/LWGANet/releases/download/weights/ORCNN_LWGANet_L2_fpn_le90_dota10_ss_e30.pth)           |


DOTA1.5

|         Model         |  mAP  | Angle | training mode | Batch Size |                                             Configs                                              |                                                     Download                                                     |
| :----------------------: |:-----:| :---: |---| :------: |:------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------:|
| LEGNet-Small (1024,1024,200) | 72.89 | le90  | single-scale |    2\*4     | [orcnn_legnet_small_dota10_test_ss_e36.py](./configs/legnet/orcnn_legnet_small_dota15_test_ss_e36.py) | [model](https://github.com/lwCVer/LWGANet/releases/download/weights/ORCNN_LWGANet_L2_fpn_le90_dota15_ss_e30.pth) |

FAIR-v1.0

|         Model         |  mAP  | Angle | training mode | Batch Size |                                             Configs                                              |                                                     Download                                                     |
| :----------------------: |:-----:| :---: |---| :------: |:------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------:|
| LEGNet-Small (1024,1024,500) | 48.35 | le90  | multi-scale |    2\*4     | [orcnn_legnet_small_fairv1_test_ms_e12.py](./configs/legnet/orcnn_legnet_small_fairv1_test_ms_e12.py) | [model](https://github.com/lwCVer/LWGANet/releases/download/weights/ORCNN_LWGANet_L2_fpn_le90_dota15_ss_e30.pth) |

DIOR-R 

|                    Model                     |  mAP  | Batch Size |
| :------------------------------------------: |:-----:| :--------: |
|                   LEGNet-Small                  | 68.40 |    1\*8    |

## Installation

MMRotate depends on [PyTorch](https://pytorch.org/), [MMCV](https://github.com/open-mmlab/mmcv) and [MMDetection](https://github.com/open-mmlab/mmdetection).
Below are quick steps for installation.
Please refer to [Install Guide](https://mmrotate.readthedocs.io/en/latest/install.html) for more detailed instruction.

```shell
conda create -n LEGNet-Det python=3.8 -y
conda activate LEGNet-Det
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
pip install -U openmim
mim install mmcv-full
mim install mmdet
# git clone https://github.com/open-mmlab/mmrotate.git
# cd mmrotate
pip install -v -e .
```

## Get Started

Please see [get_started.md](docs/en/get_started.md) for the basic usage of MMRotate.
We provide [colab tutorial](demo/MMRotate_Tutorial.ipynb), and other tutorials for:

- [learn the basics](docs/en/intro.md)
- [learn the config](docs/en/tutorials/customize_config.md)
- [customize dataset](docs/en/tutorials/customize_dataset.md)
- [customize model](docs/en/tutorials/customize_models.md)
- [useful tools](docs/en/tutorials/useful_tools.md)

## Acknowledgement

MMRotate is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks. We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new methods.

## Citation


## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for Non-commercial use only.
Any commercial use should get formal permission first.
