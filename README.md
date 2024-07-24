# Diffusion Model Learning: 2024 U-SURF program 

Learning Diffusion Model(DDPM) and basics of PyTorch to code DDPM

## Environment Required

- python 3.8
- PyTorch 2.3.1
- cuda 11.8

## Codes

### pytorch learning jupyter notebooks based on [차근차근 실습하며 배우는 파이토치 딥러닝 프로그래밍](https://github.com/wikibook/pytorchdl2)
  - [pytorch_learning_mnist.ipynb](/pytorch_learning_mnist.ipynb) : download & use MNIST dataset in pytorch
  - [pytorch_learning_CNN.ipynb](/pytorch_learning_CNN.ipynb) : how to use CNN in pytorch

### Diffusion model coding
  - [forward_process.ipynb](/forward_process.ipynb) : forward process - using MNIST
  
  - Learning U-net & diffusion model with pytorch 
    - [https://huggingface.co/blog/annotated-diffusion](https://huggingface.co/blog/annotated-diffusion) : referenced website
    - [Unet_learning.ipynb](./Unet_learning.ipynb) : U-net & Diffusion model Learning
    - [Unet.py](./Unet.py) : U-net as complete module
  
  - [diffusion_model.ipynb](/diffusion_model.ipynb) : Diffusion model training & sampling using MNIST dataset
  
  - [DiffusionModel.py](./DiffusionModel.py) : Diffusion model as complete module - containing Guided sampling, DDIM sampling
  
  - Guided Diffusion model coding
    - [guided_diffusion_training.ipynb](./guided_diffusion_training.ipynb) : Guided Diffusion model training
    - [GuidedUnet.py](./GuidedUnet.py) : GuidedUnet - modified Unet to accept integer conditioning variable
    - [guided_diffusion_sampling.ipynb](./guided_diffusion_sampling.ipynb) : Guided Diffusion model sampling
  
  - DDIM coding
    - [DDIM_training.ipynb](./DDIM_training.ipynb) : train model for DDIM sampling
    - [DDIM_sampling.ipynb](./DDIM_sampling.ipynb) : DDIM sampling from DDPM model

### Paper reviews
  - [Diffusion Model - Presentation](./diffusion_presentation.pdf)
  - [DDIM paper - Presentation](./DDIM_presentation.pdf)
  - [SnapFusion paper - Presentation](./snapfusion_presentation.pdf)