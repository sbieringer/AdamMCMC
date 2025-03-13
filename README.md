# AdamMCMC

![Sampling with AdamMCMC and stochastic gradient Hamiltonian Monte Carlo (HMC) in comparison to stochastic gradient descent and Adam](figs/01_convergence_compare.png)

This is the implementation of [AdamMCMC](https://arxiv.org/abs/2312.14027). It is a (stochastic) Metropolis-Hastings algorithm, with proposals $`\tilde{\vartheta}_{t+1}`$ sampled from a prolate distribution 

```math
    \tilde{\vartheta}_{t+1} \sim q_1(\vartheta \mid \vartheta_t, m_{t+1}, v_{t+1}) = \mathcal{N}\left(\vartheta; \, \vartheta_t-u_t(m_{t+1}, v_{t+1}), \Sigma_t(m_{t+1}, v_{t+1}) \right)
```

centered in Adam-update steps $`\vartheta_t-u_t(m_{t+1})`$.
The elliptical covariance of the proposal distribution

```math
    \Sigma_t=\sigma^2 \mathbb{1}_P+\sigma^2_\nabla u_t(m_{t+1}) \, u_{t}(m_{t+1})^\top
```

allows efficient sampling at low $\sigma$, where the algorithm behaves similar to the [Adam](https://arxiv.org/abs/1412.6980)-optimizer.

<div style="text-align: center;">
  <img src="./figs/02b_accuracy_accept_sigma.png" alt="Sampling at low $\sigma$ with and without the elliptical proposal distribution" width="500"/>
</div>

Increasing the width of the proposal distribution $\sigma$ allows adapting the uncertainty prediction of the ensemble of weight samples.

![Adapting the prediction with $\sigma$](figs/03_uncertainty.png)

## Structure

The implementation of ParticleNet is created from the [ParT](https://github.com/jet-universe/particle_transformer) implementation  using the `weaver` [package](https://github.com/hqucms/weaver-core). 
  * <code>src/AdamMCMC.py</code> defines our AdamMCMC implementation which can by used in exchange for your usual PyTorch Optimizer
  * <code>src/MCMC_weaver_util.py</code> wraps the weaver training code for the use with MCMC methods
  * <code>train_METHOD.py</code> can be used for Network training or sampling
  * <code>eval.py</code> calculate the Network output for multiple weigth samples
  * <code>test.ipynb</code>  and <code>src/compare_adammccm_sgHMC.ipynb</code> are used for plotting

## Basic Usage

An full instructive example of converting a PyTorch training to AdamMCMC sampling is provided seperately at https://github.com/sbieringer/how_to_bayesianise_your_NN. 

### Initialization

```python
import torch
import torch.nn as nn
import numpy as np

# For the model
import normflows as nf

# For MCMC Bayesian
from src.AdamMCMC import MCMC_by_bp

# For the data
from sklearn.datasets import make_moons


# Define data
data, _ = make_moons(4096, noise=0.05)
data = torch.from_numpy(data).float()
data = data.to(device)

# Define model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base = nf.distributions.base.DiagGaussian(2)
num_layers = 5
flows = []
for i in range(num_layers):
    param_map = nf.nets.MLP([1, 32, 32, 2], init_zeros=False)
    flows.append(nf.flows.AffineCouplingBlock(param_map))
    flows.append(nf.flows.Permute(2, mode='swap'))
MCMC_model = nf.NormalizingFlow(base, flows).to(device)
MCMC_model.device = device

# Initialize AdamMCMC
epochs = 10001
batchsize = len(data)
lr = 1e-3
temp = 1 #lambda
sigma = .02 #noise
loop_kwargs = {
            'MH': True, #This is a little more than x2 runtime but necessary
            'verbose': epochs<10,
            'fixed_batches': True, #set this to True so the loss is calculated 2 times per step, set to False only for batchsize = len(data)
            'sigma_adam_dir': 800, #choose on the order of the number of parameters of the network
            'extended_doc_dict': False,
            'full_loss': None, #second loss function can be passed for exact MH-corrections over the full data
}

optimizer = torch.optim.Adam(MCMC_model.parameters(), lr=lr, betas=(0.999, 0.999))
adamMCMC = MCMC_by_bp(MCMC_model, optimizer, temp, sigma)

```

### Training/sampling loop

```python
flow_loss_epoch, acc_prob_epoch, accepted_epoch = np.zeros(epochs), np.zeros(epochs),  np.zeros(epochs)

eps = tqdm(range(epochs))
for epoch in eps:    
    optimizer.zero_grad()

    perm = torch.randperm(len(data)).to(device)
    for i_step in range((len(data)-1)//batchsize+1):
        x = data[perm[i_step*batchsize:(i_step+1)*batchsize].to(device)]

        # Need to definde the loss function as a callable
        flow_loss = lambda: -torch.sum(MCMC_model.log_prob(x)) 
        flow_loss_old,accept_prob,accepted,_,_ = adamMCMC.step(flow_loss, **loop_kwargs)

        flow_loss_epoch[epoch] += flow_loss_old.numpy(force=True)/len(x)
        acc_prob_epoch[epoch] = accept_prob
        accepted_epoch[epoch] = accepted

        #save the ensemble after some burn-in time (to converge) in sufficiently large intervals
        #if you loaded a pretrained model, you can also reduce/skip the burn-in
        if epoch>4999 and epoch%1000==0:
            torch.save(MCMC_model.state_dict(), f"./models/MCMC_model_{epoch}.pth")

        eps.set_postfix({'flow_loss': flow_loss_old.item()/len(x), 'accept_prob': accept_prob})

```

### <code>train_METHOD.py</code> arguments

  * <code>train_adam.py</code>:
    - beta1_adam: $\beta_1 = \beta_2$ running average parameters of first and second order momentum of Adam (`default=0.99`)
    - batchsize is fixed at $512$ and lr at $10^{-3}$

  * <code>train_sgHMC.py</code>:
    - lr: learning rate (`default=10^-2`)
    - C: friction term of sgHMC
    - resample_mom: Enables momentum resampling

  * <code>train_MCMC.py</code>:
    - temp: temperature parameter as described in the [paper](https://arxiv.org/abs/2312.14027)
    - sigma: standard deviation of the proposal distribution $\sigma$ = `sigma`$/\sqrt{\sharp \vartheta}$
    - sigma_adam_dir_denom: covariance factor in update direction $\sigma_\Delta$ = `sigma_adam_dir_denom`$/\sqrt{\sharp \vartheta}$
    - optim_str: `"Adam"` or `"SGD"`, sets the PyTorch optimizer used for calculating the update steps
    - beta1_adam: $\beta_1 = \beta_2$ running average parameters of first and second order momentum of Adam (`default=0.99`)
    - bs: batchisze (`default=512`)
    - lr: learning rate (`default=10^-3`)
    - full_loss: Enabels using the loss calculated on the full set of data for the Metropolis-Hastings-Correction

## Citation

For more Details see our Publication ["AdamMCMC: Combining Metropolis Adjusted Langevin with Momentum-based Optimization"](https://arxiv.org/abs/2312.14027)

```bibtex
@unpublished{Bieringer_2023_adammcmc,
    author = "Bieringer, Sebastian and Kasieczka, Gregor and Steffen, Maximilian F. and Trabs, Mathias",
    title = "{AdamMCMC: Combining Metropolis Adjusted Langevin with Momentum-based Optimization}",
    eprint = "2312.14027",
    archivePrefix = "arXiv",
    primaryClass = "stat.ML",
    month = "12",
    year = "2023",
}.
```
