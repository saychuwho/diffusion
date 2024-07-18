# Diffusion Model Learning: 2024 U-SURF program 

Learning Diffusion Model(DDPM) and basics of PyTorch to code DDPM

## pytorch learning jupyter notebooks based on [차근차근 실습하며 배우는 파이토치 딥러닝 프로그래밍](https://github.com/wikibook/pytorchdl2)
  - [download & use MNIST dataset in pytorch](/pytorch_learning_mnist.ipynb)
  - [how to use CNN in pytorch](/pytorch_learning_CNN.ipynb) 

## Diffusion model coding
  - [forward process - using MNIST](/forward_process.ipynb)
  
  - Learning U-net & diffusion model with pytorch 
    - [referenced website](https://huggingface.co/blog/annotated-diffusion)
    - [U-net & Diffusion model Learning](./Unet_learning.ipynb)
    - [U-net as complete module](./Unet.py)
  
  - [Diffusion model using MNIST dataset](/diffusion_model.ipynb)
  
  - [Diffusion model as complete module - containing Guided sampling, DDIM sampling](./DiffusionModel.py)
  
  - [Guided Diffusion model](./guided_diffusion.ipynb)
    - [GuidedUnet - modified Unet to accept integer conditioning variable](./GuidedUnet.py)
    - [Guided Diffusion model only inference](./guided_diffusion_only_inference.ipynb)
  
  - [DDIM sampling from DDPM model](./DDIM_sampling.ipynb)

## Paper reviews
  - [Diffusion Model](./diffusion_presentation.pdf)
  - [DDIM paper](./DDIM_발표_추성재.pdf)