""" functions gathered from diffusion_model.ipynb """

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


""" Beta Scheduling """

""" from https://huggingface.co/blog/annotated-diffusion """ 
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def exp_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6,6,timesteps)
    tmp = torch.exp(betas)
    tmp_max = tmp.max()
    tmp_min = tmp.min()
    tmp_2 = (tmp - tmp_min) / (tmp_max - tmp_min)
    return tmp_2 * (beta_end - beta_start) + beta_start

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def total_beta_schedule(beta_schedule, timestep, prev_betas=None):
    if prev_betas == None: 
        betas = beta_schedule(timestep)
    else:
        betas = prev_betas

    # used in training
    alphas = torch.cumprod(1. - betas, axis=0)
    sqrt_alphas = torch.sqrt(alphas)
    sqrt_one_minus_alphas = torch.sqrt(1. - alphas)

    # used in sampling
    sampling_1 = 1. / torch.sqrt(1. - betas)
    sampling_2 = betas / sqrt_one_minus_alphas

    return betas, alphas, sqrt_alphas, sqrt_one_minus_alphas, sampling_1, sampling_2



""" Training Part - Default """
def training(data_loader, epochs, timestep, net, optimizer:optim.Adam, device, sqrt_alphas, sqrt_one_minus_alphas):
    taken_step = 0

    training_taken_step = []
    training_loss = []

    for i in range(epochs):
        print(f">> epoch : {i}")
        for datas, labels in data_loader: # sample a data point
            optimizer.zero_grad()
            
            taken_step += 1

            batch_size = len(labels)

            datas = datas.to(device)

            t = torch.randint(0, timestep, (batch_size,), device=device).long()# sample a point along the Markov chain
            
            # get alpha values
            sqrt_alphas_t = extract(sqrt_alphas, t, datas.shape)
            sqrt_one_minus_alphas_t = extract(sqrt_one_minus_alphas, t, datas.shape)

            epsilon = torch.randn_like(datas).to(device) # sample a noise vector
        
            z_t = sqrt_alphas_t * datas + sqrt_one_minus_alphas_t * epsilon # Evaluate noisy latent variable

            output = net(z_t, t)

            loss = F.mse_loss(epsilon, output) # compute loss term >> change loss term into l1 loss

            if taken_step % 100 == 0:
                print("Loss: ", loss.item())

            # save loss
            training_taken_step.append(taken_step)
            training_loss.append(loss)

            loss.backward()

            optimizer.step()

    return training_loss, training_taken_step, taken_step


""" Training part - Guided (MNIST) """
def training_guided(data_loader, epochs, timestep, net, optimizer:optim.Adam, device, sqrt_alphas, sqrt_one_minus_alphas):
    taken_step = 0

    training_taken_step = []
    training_loss = []

    for i in range(epochs):
        print(f">> epoch : {i} / with condition variable")
        for datas, labels in data_loader: # sample a data point
            optimizer.zero_grad()
            
            taken_step += 1
            batch_size = len(labels)
            datas = datas.to(device)

            t = torch.randint(0, timestep, (batch_size,), device=device).long()# sample a point along the Markov chain
            c = torch.tensor(labels, device=device).long()    

            # get alpha values
            sqrt_alphas_t = extract(sqrt_alphas, t, datas.shape)
            sqrt_one_minus_alphas_t = extract(sqrt_one_minus_alphas, t, datas.shape)

            epsilon = torch.randn_like(datas).to(device) # sample a noise vector
            z_t = sqrt_alphas_t * datas + sqrt_one_minus_alphas_t * epsilon # Evaluate noisy latent variable
            output = net(z_t, t, c)
            loss = F.mse_loss(epsilon, output) # compute loss term >> change loss term into l1 loss

            if taken_step % 100 == 0:
                print("Loss: ", loss.item())

            # save loss
            training_taken_step.append(taken_step)
            training_loss.append(loss)

            loss.backward()
            optimizer.step()

        # 코드를 줄일 수 있을거 같은데, 일단은 이렇게 만들어보자.
        print(f">> epoch : {epochs} // without conditioning variable")
        for datas, labels in data_loader:
            optimizer.zero_grad()
            
            taken_step += 1
            batch_size = len(labels)
            datas = datas.to(device)

            t = torch.randint(0, timestep, (batch_size,), device=device).long()# sample a point along the Markov chain

            # get alpha values
            sqrt_alphas_t = extract(sqrt_alphas, t, datas.shape)
            sqrt_one_minus_alphas_t = extract(sqrt_one_minus_alphas, t, datas.shape)

            epsilon = torch.randn_like(datas).to(device) # sample a noise vector
            z_t = sqrt_alphas_t * datas + sqrt_one_minus_alphas_t * epsilon # Evaluate noisy latent variable
            output = net(z_t, t)
            loss = F.mse_loss(epsilon, output) # compute loss term >> change loss term into l1 loss

            if taken_step % 100 == 0:
                print("Loss: ", loss.item())

            # save loss
            training_taken_step.append(taken_step)
            training_loss.append(loss)

            loss.backward()
            optimizer.step()


    return training_loss, training_taken_step, taken_step


""" sampling from net """
@torch.no_grad()
def sample(x:torch.Tensor, t, betas, sampling_1, sampling_2, net, device):
    # evaluate network output
    net_t = torch.tensor([t], device=device)

    sampling_1_t = extract(sampling_1, net_t , x.shape)
    sampling_2_t = extract(sampling_2, net_t, x.shape)
    betas_t = extract(betas, net_t, x.shape)

    mu = sampling_1_t * (x - sampling_2_t * net(x, net_t))
    epsilon = torch.randn_like(x).to(device)
    x = mu + torch.sqrt(betas_t) * epsilon

    return x

@torch.no_grad()
def sampling(timestep, betas, sampling_1, sampling_2, net, device, noise=None):
    if noise==None:
        noise = torch.randn((1,1,32,32)) # sample from final latent space
    x = noise.to(device)
    # sampling_1 = sampling_1.to(device)
    # sampling_2 = sampling_2.to(device)
    # betas = betas.to(device)

    for t in reversed(range(1, timestep)):
        x = sample(x, t, betas, sampling_1, sampling_2, net, device)

    last_t = torch.tensor([0], device=device)

    sampling_1_0 = extract(sampling_1, last_t, x.shape)
    sampling_2_0 = extract(sampling_2, last_t, x.shape)

    x = sampling_1_0 * (x - sampling_2_0 * net(x, last_t))

    return x

""" sampling - Guided (MNIST) """
@torch.no_grad()
def sample_guided(z_t:torch.Tensor, t:int, c, delta, betas, sampling_1, sampling_2, net, device):
    # evaluate network output
    net_t = torch.tensor([t], device=device)
    net_c = torch.tensor([c], device=device)
    net_c_null = None

    sampling_1_t = extract(sampling_1, net_t, z_t.shape)
    sampling_2_t = extract(sampling_2, net_t, z_t.shape)
    betas_t = extract(betas, net_t, z_t.shape)

    # sample
    mu_1 = sampling_1_t * (z_t - sampling_2_t * net(z_t, net_t, net_c))
    epsilon_1 = torch.randn_like(z_t).to(device)
    z_next_1 = mu_1 + torch.sqrt(betas_t) * epsilon_1

    mu_2 = sampling_1_t * (z_t - sampling_2_t * net(z_t, net_t, net_c_null))
    epsilon_2 = torch.randn_like(z_t).to(device)
    z_next_2 = mu_2 + torch.sqrt(betas_t) * epsilon_2

    z_next = (z_next_1 * delta)*(z_next_2 * (1-delta))

    return z_next

@torch.no_grad()
def sampling_guided(timestep, condition:int, delta, betas, sampling_1, sampling_2, net, device, noise=None):
    if noise==None:
        noise = torch.randn((1,1,32,32)) # sample from final latent space
    x = noise.to(device)
    # sampling_1 = sampling_1.to(device)
    # sampling_2 = sampling_2.to(device)
    # betas = betas.to(device)

    for t in reversed(range(1, timestep)):
        x = sample_guided(x, t, condition, delta, betas, sampling_1, sampling_2, net, device)

    last_t = torch.tensor([0], device=device)
    last_c = torch.tensor([condition], device=device)
    last_c_null = None
    sampling_1_0 = extract(sampling_1, last_t, x.shape)
    sampling_2_0 = extract(sampling_2, last_t, x.shape)
    betas_0 = extract(betas, last_t, x.shape)

    mu_1 = sampling_1_0 * (x - sampling_2_0 * net(x, last_t, last_c))
    epsilon_1 = torch.randn_like(x).to(device)
    z_next_1 = mu_1 + torch.sqrt(betas_0) * epsilon_1

    mu_2 = sampling_1_0 * (x - sampling_2_0 * net(x, last_t, last_c_null))
    epsilon_2 = torch.randn_like(x).to(device)
    z_next_2 = mu_2 + torch.sqrt(betas_0) * epsilon_2

    x = (z_next_1 * delta)*(z_next_2 * (1-delta))

    return x

""" sampling - DDIM """

@torch.no_grad()
def sample_DDIM(x:torch.Tensor, 
                t, 
                sqrt_alphas, 
                sqrt_one_minus_alphas, 
                net, 
                device):
    # evaluate network output
    net_t = torch.tensor([t], device=device)
    net_t_minus_1 = torch.tensor([t-1], device=device)

    sqrt_alpha_t = extract(sqrt_alphas, net_t , x.shape)
    sqrt_alpha_t_minus_1 = extract(sqrt_alphas, net_t_minus_1, x.shape)
    sqrt_one_minus_alpha_t = extract(sqrt_one_minus_alphas, net_t, x.shape)
    sqrt_one_minus_alpha_t_minus_1 = extract(sqrt_one_minus_alphas, net_t_minus_1, x.shape)
    
    x_1 = sqrt_alpha_t_minus_1 * ((x - sqrt_one_minus_alpha_t * net(x, net_t))/sqrt_alpha_t)
    x_2 = sqrt_one_minus_alpha_t_minus_1 * net(x, net_t)

    return x_1 + x_2

@torch.no_grad()
def sampling_DDIM(timestep,
                  sqrt_alphas,
                  sqrt_one_minus_alphas,  
                  net, 
                  device, 
                  noise=None,
                  length=None):
    
    # generate initial noise
    if noise==None:
        noise = torch.randn((1,1,32,32)) # sample from final latent space
    x = noise.to(device)
    # sampling_1 = sampling_1.to(device)
    # sampling_2 = sampling_2.to(device)
    # betas = betas.to(device)

    # generate sequence tau
    tau = range(1, timestep)
    if length != None:
        tau = sorted(list(np.random.choice(range(2,timestep), length, replace=False)))

    for t in reversed(tau):
        t = int(t)
        print(f"tau : {t}") # tmp
        x = sample_DDIM(x, t, sqrt_alphas, sqrt_one_minus_alphas, net, device)
    
    t = 2    
    x = sample_DDIM(x, t, sqrt_alphas, sqrt_one_minus_alphas, net, device)

    t_1 = torch.tensor([1], device=device)

    sqrt_alpha_1 = extract(sqrt_alphas, t_1, x.shape)
    sqrt_one_minus_alpha_1 = extract(sqrt_one_minus_alphas, t_1, x.shape)

    x = (x - sqrt_one_minus_alpha_1 * net(x, t_1)) / sqrt_alpha_1

    return x



""" save & load models (for inference) """
# net is saved in "./checkpoints/{net_name}_net.pt"
# betas are saved in "./checkpoints/{net_name}_betas.pt"
def save_net_betas(net:torch.nn.Module, 
                   betas:torch.Tensor,
                   sampling_1:torch.Tensor,
                   sampling_2:torch.Tensor, 
                   net_name="default"):
    
    save_net_path = f"./checkpoints/{net_name}_net.pt"
    save_betas_path = f"./checkpoints/{net_name}_betas.pt"

    # save net
    torch.save(net.state_dict(), save_net_path)

    # save betas
    torch.save({'betas':betas,
                'sampling_1':sampling_1,
                'sampling_2':sampling_2}, save_betas_path)


def load_net_betas(net:torch.nn.Module,
                   device,
                   net_name="default"):
    
    save_net_path = f"./checkpoints/{net_name}_net.pt"
    save_betas_path = f"./checkpoints/{net_name}_betas.pt"
    
    # load net
    net.load_state_dict(torch.load(save_net_path))
    net.eval()
    net.to(device)

    # load betas
    loaded_schedule = torch.load(save_betas_path)
    loaded_betas = loaded_schedule['betas']
    loaded_sampling_1 = loaded_schedule['sampling_1']
    loaded_sampling_2 = loaded_schedule['sampling_2']

    return net, loaded_betas, loaded_sampling_1, loaded_sampling_2