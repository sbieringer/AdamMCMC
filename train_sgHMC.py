# %%
from __future__ import division

import numpy as np 
import matplotlib.pyplot as plt
import torch
import os
from argparse import ArgumentParser, ArgumentTypeError

from src.util import *
from src.MCMC_weaver_utils import tb_helper_offline, train_classification_sgHMC

from weaver.train import train_load, test_load, model_setup, optim
from weaver.utils.data.fileio import _read_parquet
from weaver.utils.nn.tools import _flatten_preds, _logger, Counter

import sys
sys.path.append('../..')
from src.MCMC_Adam import MCMC_by_bp as AdamMCMC

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

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
            steps_per_epoch=project.steps_per_epoch, tb_helper=offline_tb_MCMC, grad_scaler=grad_scaler)
        
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



