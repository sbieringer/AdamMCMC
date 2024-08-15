# %%
from __future__ import division

import numpy as np 
import matplotlib.pyplot as plt
import torch
import os
from argparse import ArgumentParser, ArgumentTypeError

from numpy.random import gamma
from torch.optim import Optimizer

from weaver.train import train_load, test_load, model_setup, optim
from weaver.utils.data.fileio import _read_parquet

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

# %%
def mkdir(save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    return save_dir

def smooth(x, kernel_size=5):
    if kernel_size == 1:
        return x
    else:
        assert kernel_size % 2 != 0
        x_shape = x.shape
        x_tmp = np.array([x[i:x_shape[0]-kernel_size+i+1] for i in range(kernel_size)])
        edge1 = x[:int((kernel_size-1)/2)]
        edge2 = x[-int((kernel_size-1)/2):]
        x_out = np.concatenate((edge1, np.mean(x_tmp, 0),edge2),0)
        assert x_shape == x_out.shape
        return x_out #np.mean(np.array(x).reshape(-1, kernel_size),1)

# %%
class tb_helper_offline():
    def __init__(self, scalars, path, batch_train_count=0, batch_val_count=0):
        self.path = path
        self.scalars = {key: np.zeros(scalars[key]) for key in scalars}

        self.custom_fn = False
        self.batch_train_count = batch_train_count
        self.batch_val_count = batch_val_count

    def write_scalars(self, entry):
        for scalar_entry in entry:
            key, val, batch = scalar_entry
            if key in self.scalars:
                self.scalars[key][batch] = val

    def set_batch_train_count(self, batch_train_count):
        self.batch_train_count = batch_train_count

    def set_batch_val_count(self, batch_val_count):
        self.batch_val_count = batch_val_count

    def save(self):
        for key in self.scalars:
            np.save(self.path + key + '.npy', self.scalars[key])

    def load(self):
        for key in self.scalars:
            self.scalars[key] = np.load(self.path + key + '.npy')

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


parser = ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--C', type=float, default=0.2)
parser.add_argument('--resample_mom', type=str2bool, default=False)
runargs = parser.parse_args()

# %%
model_path = './top_landscape/models/'
data_path = "./top_landscape/data/"

data_config = "./top_landscape/particle_transformer/data/TopLandscape/top_kin.yaml"
model_config = "./top_landscape/particle_transformer/networks/example_ParticleNet.py"

log_path = './top_landscape/logs/'

# %%
class empty():
    def __init__(self):
        pass

project = empty()
project.data_train = [data_path + '/train_file.parquet']

table = _read_parquet(project.data_train, {'label'})
project.len_data_train = len(table)
del table

project.data_val = [data_path + '/val_file.parquet']
project.data_test = [data_path + '/test_file.parquet']
project.data_config = data_config
project.network_config = model_config
project.num_workers = 1
project.fetch_step = 1
project.in_memory = True
project.batch_size = 512
project.samples_per_epoch = 2400*project.batch_size #2400
project.samples_per_epoch_val = 800*project.batch_size #800
project.num_epochs = 100
project.gpus = 0
project.start_lr = 1e-2
project.optimizer = "adam" 
project.log = log_path + '/ParticleNet_Test.log'
project.predict = False
project.predict_output = ''

project.regression_mode = False
project.extra_selection = None
project.extra_test_selection = None
project.data_fraction = 1
project.file_fraction = 1
project.fetch_by_files = False
project.train_val_split = 0.8
project.no_remake_weights = False
project.demo = False
project.lr_finder = None
project.tensorboard = None
project.tensorboard_custom_fn = None
project.network_option = []
project.load_model_weights = None
project.exclude_model_weights = None
project.steps_per_epoch = None
project.steps_per_epoch_val = None
project.optimizer_option = [('betas', '(0.99, 0.99)')]
project.lr_scheduler = "flat"#"flat+decay"
project.warmup_steps = 0
project.load_epoch = None
project.use_amp = False
project.predict_gpus = None
project.export_onnx = None
project.export_opset = 15
project.io_test = False
project.copy_inputs = False
project.print = False
project.profile = False
project.backend = None
project.cross_validation = None

project.local_rank = None if project.backend is None else int(os.environ.get("LOCAL_RANK", "0"))

project.model_prefix = mkdir(model_path + f'/ParticleNet_{project.optimizer}_lr{project.start_lr}_opt{project.optimizer_option}/')

if project.samples_per_epoch is not None:
    if project.steps_per_epoch is None:
        project.steps_per_epoch = project.samples_per_epoch // project.batch_size
    else:
        raise RuntimeError('Please use either `--steps-per-epoch` or `--samples-per-epoch`, but not both!')

if project.samples_per_epoch_val is not None:
    if project.steps_per_epoch_val is None:
        project.steps_per_epoch_val = project.samples_per_epoch_val // project.batch_size
    else:
        raise RuntimeError('Please use either `--steps-per-epoch-val` or `--samples-per-epoch-val`, but not both!')

if project.steps_per_epoch_val is None and project.steps_per_epoch is not None:
    project.steps_per_epoch_val = round(project.steps_per_epoch * (1 - project.train_val_split) / project.train_val_split)
if project.steps_per_epoch_val is not None and project.steps_per_epoch_val < 0:
    project.steps_per_epoch_val = None

if not "Acc" in os.listdir(project.model_prefix):
    mkdir(project.model_prefix+"Acc/")
if not "Loss" in os.listdir(project.model_prefix):
    mkdir(project.model_prefix+"Loss/")


# %%

# %%
#from https://github.com/JavierAntoran/Bayesian-Neural-Networks/blob/master/src/Stochastic_Gradient_HMC_SA/optimizers.py

class H_SA_SGHMC(Optimizer):
    """ Stochastic Gradient Hamiltonian Monte-Carlo Sampler that uses scale adaption during burn-in
        procedure to find some hyperparamters. A gaussian prior is placed over parameters and a Gamma
        Hyperprior is placed over the prior's standard deviation"""

    def __init__(self, params, lr=1e-2, base_C=0.05, gauss_sig=0.1, alpha0=10, beta0=10):

        self.eps = 1e-6
        self.alpha0 = alpha0
        self.beta0 = beta0

        if gauss_sig == 0:
            self.weight_decay = 0
        else:
            self.weight_decay = 1 / (gauss_sig ** 2)

        if self.weight_decay <= 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(self.weight_decay))
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if base_C < 0:
            raise ValueError("Invalid friction term: {}".format(base_C))

        defaults = dict(
            lr=lr,
            base_C=base_C,
        )
        super(H_SA_SGHMC, self).__init__(params, defaults)

    def step(self, burn_in=False, resample_momentum=False, resample_prior=False):
        """Simulate discretized Hamiltonian dynamics for one step"""
        loss = None

        for group in self.param_groups:  # iterate over blocks -> the ones defined in defaults. We dont use groups.
            for p in group["params"]:  # these are weight and bias matrices
                if p.grad is None:
                    continue
                state = self.state[p]  # define dict for each individual param
                if len(state) == 0:
                    state["iteration"] = 0
                    state["tau"] = torch.ones_like(p)
                    state["g"] = torch.ones_like(p)
                    state["V_hat"] = torch.ones_like(p)
                    state["v_momentum"] = torch.zeros_like(
                        p)  # p.data.new(p.data.size()).normal_(mean=0, std=np.sqrt(group["lr"])) #
                    state['weight_decay'] = self.weight_decay

                state["iteration"] += 1  # this is kind of useless now but lets keep it provisionally

                if resample_prior:
                    alpha = self.alpha0 + p.data.nelement() / 2
                    beta = self.beta0 + (p.data ** 2).sum().item() / 2
                    gamma_sample = gamma(shape=alpha, scale=1 / (beta), size=None)
                    #                     print('std', 1/np.sqrt(gamma_sample))
                    state['weight_decay'] = gamma_sample

                base_C, lr = group["base_C"], group["lr"]
                weight_decay = state["weight_decay"]
                tau, g, V_hat = state["tau"], state["g"], state["V_hat"]

                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                # update parameters during burn-in
                if burn_in:  # We update g first as it makes most sense
                    tau.add_(-tau * (g ** 2) / (
                                V_hat + self.eps) + 1)  # specifies the moving average window, see Eq 9 in [1] left
                    tau_inv = 1. / (tau + self.eps)
                    g.add_(-tau_inv * g + tau_inv * d_p)  # average gradient see Eq 9 in [1] right
                    V_hat.add_(-tau_inv * V_hat + tau_inv * (d_p ** 2))  # gradient variance see Eq 8 in [1]

                V_sqrt = torch.sqrt(V_hat)
                V_inv_sqrt = 1. / (V_sqrt + self.eps)  # preconditioner

                if resample_momentum:  # equivalent to var = M under momentum reparametrisation
                    state["v_momentum"] = torch.normal(mean=torch.zeros_like(d_p),
                                                       std=torch.sqrt((lr ** 2) * V_inv_sqrt))
                v_momentum = state["v_momentum"]

                noise_var = (2. * (lr ** 2) * V_inv_sqrt * base_C - (lr ** 4))
                noise_std = torch.sqrt(torch.clamp(noise_var, min=1e-16))
                # sample random epsilon
                noise_sample = torch.normal(mean=torch.zeros_like(d_p), std=torch.ones_like(d_p) * noise_std)

                # update momentum (Eq 10 right in [1])
                v_momentum.add_(- (lr ** 2) * V_inv_sqrt * d_p - base_C * v_momentum + noise_sample)

                # update theta (Eq 10 left in [1])
                p.data.add_(v_momentum)

        return loss

# %%
from weaver.utils.nn.tools import _flatten_preds, _logger, Counter

import sys
sys.path.append('../..')
from src.MCMC_Adam import MCMC_by_bp as AdamMCMC

import time
import tqdm

def train_classification_sgHMC(
        model, loss_func, MCMC, scheduler, train_loader, dev, epoch, burn_in, resample_momentum, n_points=1, steps_per_epoch=None,
        tb_helper=None):
    model.train()

    data_config = train_loader.dataset.config

    label_counter = Counter()
    total_loss = 0
    num_batches = 0
    total_correct = 0
    entry_count = 0
    count = 0
    start_time = time.time()

    maxed_out_mbb_batches  = 0

    with tqdm.tqdm(train_loader) as tq:
        for X, y, _ in tq:
            inputs = [X[k].to(dev) for k in data_config.input_names]
            label = y[data_config.label_names[0]].long().to(dev)
            entry_count += label.shape[0]
            try:
                mask = y[data_config.label_names[0] + '_mask'].bool().to(dev)
            except KeyError:
                mask = None
            MCMC.zero_grad()
            with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
                model_output = model(*inputs)
                logits, label, _ = _flatten_preds(model_output, label=label, mask=mask)
                loss = loss_func(logits, label)*n_points

            loss.backward()
            
            t1 = time.time()
            _ = MCMC.step(burn_in, resample_momentum)
            resample_momentum = False #set resampling to false for the rest of the epoch
            t2 = time.time()

            if scheduler and getattr(scheduler, '_update_per_step', False):
                scheduler.step()
            if scheduler.get_last_lr()[0] <= getattr(scheduler, 'min_lr', 0):
                scheduler._update_per_step = False

            _, preds = logits.max(1)
            loss = loss.item()
                
            num_examples = label.shape[0]
            label_counter.update(label.numpy(force=True))
            num_batches += 1
            count += num_examples
            correct = (preds == label).sum().item()
            total_loss += loss
            total_correct += correct

            tq.set_postfix({
                'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else MCMC.defaults['lr'],
                'Loss': '%.5f' % loss,
                'AvgLoss': '%.5f' % (total_loss / num_batches),
                'Acc': '%.5f' % (correct / num_examples),
                'AvgAcc': '%.5f' % (total_correct / count)})

            if tb_helper:
                tb_helper.write_scalars([
                    ("Loss/train", loss, tb_helper.batch_train_count + num_batches),
                    ("Acc/train", correct / num_examples, tb_helper.batch_train_count + num_batches),
                    #('Acceptance_rate', a, tb_helper.batch_train_count + num_batches),
                    ('lr', '%.2e' % scheduler.get_last_lr()[0] if scheduler else MCMC.defaults['lr'], tb_helper.batch_train_count + num_batches),
                ])
                if tb_helper.custom_fn:
                    with torch.no_grad():
                        tb_helper.custom_fn(model_output=model_output, model=model,
                                            epoch=epoch, i_batch=num_batches, mode='train')

            if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (entry_count, entry_count / time_diff))
    _logger.info('Train AvgLoss: %.5f, AvgAcc: %.5f' % (total_loss / num_batches, total_correct / count))
    _logger.info('Train class distribution: \n    %s', str(sorted(label_counter.items())))

    if tb_helper:
        tb_helper.write_scalars([
            ("Loss/train (epoch)", total_loss / num_batches, epoch),
            ("Acc/train (epoch)", total_correct / count, epoch),
        ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode='train')
        # update the batch state
        tb_helper.batch_train_count += num_batches

    if scheduler and not getattr(scheduler, '_update_per_step', False):
        scheduler.step()


# %%
train_loader, val_loader, data_config, train_input_names, train_label_names = train_load(project)
model_MCMC, model_info_MCMC, loss_func_MCMC = model_setup(project, data_config, device=device)
model_MCMC = model_MCMC.to(device)
model_MCMC.device = device
model_MCMC.load_state_dict(torch.load('./models/base_model.pt'))

lr = runargs.lr
min_lr = 1e-6
C = runargs.C
gauss_sig = 0
burn_in_epochs = project.num_epochs#-10
resample_momentum =  runargs.resample_mom

res_str = '_res_momentum' if resample_momentum else '' 
project.model_prefix = mkdir(model_path + f'/ParticleNet_sgHMC_lr{lr}_C{C}_gauss_sig{gauss_sig}_burn{burn_in_epochs}'+res_str+'/')
if not "Acc" in os.listdir(project.model_prefix):
    mkdir(project.model_prefix+"Acc/")
if not "Loss" in os.listdir(project.model_prefix):
    mkdir(project.model_prefix+"Loss/")
project.log = log_path + '/ParticleNet_Test_sgHMC.log'

#scheduler._update_per_step = True
#scheduler.min_lr = min_lr

MCMC = H_SA_SGHMC(model_MCMC.parameters(), lr=lr, base_C=C)
scheduler = torch.optim.lr_scheduler.ExponentialLR(MCMC, 1)


# %%
print(f"initiated model with {sum(p.numel() for p in model_MCMC.parameters())} parameters")

# %%
offline_tb_MCMC = tb_helper_offline({"Loss/train":  project.steps_per_epoch*project.num_epochs+1, 
                                    "Acc/train":  project.steps_per_epoch*project.num_epochs+1,
                                    "Acceptance_rate": project.steps_per_epoch*project.num_epochs+1,
                                    "lr": project.steps_per_epoch*project.num_epochs+1}, 
                                    project.model_prefix)

# %%
load = False
if not load:
    best_valid_metric = np.inf if project.regression_mode else 0
    grad_scaler = torch.cuda.amp.GradScaler() if project.use_amp else None
    for epoch in range(project.num_epochs):
        if project.load_epoch is not None:
            if epoch <= project.load_epoch:
                continue

        train_classification_sgHMC(model_MCMC, loss_func_MCMC, MCMC, scheduler, train_loader, device, epoch, epoch<burn_in_epochs, resample_momentum, project.len_data_train,
            steps_per_epoch=project.steps_per_epoch, tb_helper=offline_tb_MCMC)
        
        if project.model_prefix and (project.backend is None or project.local_rank == 0):
            dirname = os.path.dirname(project.model_prefix)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname)
            state_dict = model_MCMC.module.state_dict() if isinstance(
                model_MCMC, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else model_MCMC.state_dict()
            torch.save(state_dict, project.model_prefix + '_epoch-%d_state.pt' % epoch)
            torch.save(MCMC.state_dict(), project.model_prefix + '_epoch-%d_optimizer.pt' % epoch)

    # valid_metric = evaluate(model_MCMC, val_loader, device, epoch, loss_func=loss_func,
    #                         steps_per_epoch=project.steps_per_epoch_val)
    # is_best_epoch = (
    #     valid_metric < best_valid_metric) if project.regression_mode else(
    #     valid_metric > best_valid_metric)
    # if is_best_epoch:
    #     best_valid_metric = valid_metric
    #     if project.model_prefix and (project.backend is None or project.local_rank == 0):
    #         shutil.copy2(project.model_prefix + '_epoch-%d_state.pt' %
    #                     epoch, project.model_prefix + '_best_epoch_state.pt')
    #         # torch.save(model, args.model_prefix + '_best_epoch_full.pt') 
    # print('Epoch #%d: Current validation metric: %.5f (best: %.5f)' %
    #             (epoch, valid_metric, best_valid_metric))

        offline_tb_MCMC.save()
else:
    offline_tb_MCMC.load()

# %%
for key in offline_tb_MCMC.scalars:
    plt.figure(figsize=(8,4))
    s = 1001 if key == "Acceptance_rate" else 101
    len_MCMC = len(offline_tb_MCMC.scalars[key][1:])
    if key == "Acceptance_rate":
        plt.plot(offline_tb_MCMC.scalars[key][1:], alpha = 0.3, color = 'C0')
        plt.plot(smooth(np.clip(offline_tb_MCMC.scalars[key], 0, 1), s)[1:-1], label = f"AdamMCMC mean acceptance {np.mean(np.clip(offline_tb_MCMC.scalars[key], 0, 1)):3.2}", color = 'C0')

        plt.ylim(-0.05, 1.1)
    else:
        plt.plot(smooth(offline_tb_MCMC.scalars[key][:len_MCMC], s)[1:], label = "AdamMCMC", color = 'C0')
        plt.plot(offline_tb_MCMC.scalars[key][1:][:len_MCMC], alpha = 0.3, color = 'C0')

    plt.xlabel('steps')
    plt.ylabel(key)
    #plt.xlim(0,240_000)
    plt.grid()
    plt.legend(loc = 'upper right')
    plt.show()
    plt.savefig(project.model_prefix+key+'.pdf')


# %%



