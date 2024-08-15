# AdamMCMC

This is the repository for our publication ["AdamMCMC: Combining Metropolis Adjusted Langevin with Momentum-based Optimization"](https://arxiv.org)

## Structure

The implementation of ParticleNet is created from the [ParT](https://github.com/jet-universe/particle_transformer/tree/main) implementation  using the `weaver` [package](https://github.com/hqucms/weaver-core/tree/main). 
  * <code>src/AdamMCMC.py</code> defines our AdamMCMC implementation and can by used in exchange for your usual PyTorch Optimizer
  * <code>src/MCMC_weaver_util.py</code> wraps some weaver code for the use with MCMC methods
  * <code>src/train_METHOD.py</code> can be used for Network training or sampling
  * <code>src/eval.py</code> and <code>src/eval_ood.py</code> calculate the Network output for multiple weigth samples
  * <code>src/test.ipynb</code>, <code>src/test_ood.ipynb</code>  and <code>src/compare_adammccm_sgHMC.ipynb</code> are used for plotting
