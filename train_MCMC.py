# %%
import numpy as np 
import matplotlib.pyplot as plt
import torch
import os
from copy import deepcopy as dc

from src.AdamMCMC import MCMC_by_bp as AdamMCMC
from src.MCMC_weaver_utils import train_classification_MCMC, tb_helper_offline
from src.util import *

from weaver.train import train_load, model_setup, optim
from weaver.utils.nn.tools import train_classification as train
from weaver.utils.nn.tools import evaluate_classification as evaluate
from weaver.utils.nn.tools import _flatten_preds
from argparse import ArgumentParser, ArgumentTypeError
from weaver.utils.data.fileio import _read_parquet


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
parser.add_argument('--temp', type=float, default=1)
parser.add_argument('--sigma', type=float, default=0.2)
parser.add_argument('--sigma_adam_dir_denom', type=float, default=100)
parser.add_argument('--optim_str', type=str, default='Adam')
parser.add_argument('--beta1_adam', type=float, default=0.99)
parser.add_argument('--bs', type=int, default=512)
parser.add_argument('--lr', type=float, default=5e-6)
parser.add_argument('--full_loss', type=str2bool, default=False)
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
project.batch_size = runargs.bs
project.samples_per_epoch = 500*project.batch_size if runargs.full_loss else 2400*project.batch_size
project.samples_per_epoch_val = 800*project.batch_size
project.num_epochs = 100*512//runargs.bs 
project.gpus = 0
project.start_lr = runargs.lr #1e-2
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
project.optimizer_option = [('betas', '(0.9999, 0.99999)')]
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


# %%
train_loader, val_loader, data_config, train_input_names, train_label_names = train_load(project)
model_MCMC, model_info_MCMC, loss_func_MCMC = model_setup(project, data_config, device=device)
model_MCMC = model_MCMC.to(device)
model_MCMC.device = device
model_MCMC.load_state_dict(torch.load('./models/base_model.pt'))

lr = runargs.lr
min_lr = 1e-6
lr_decay = 1# 0.998 # 0.995
temp = runargs.temp
sigma = runargs.sigma
optim_str = runargs.optim_str
betas_adam = (runargs.beta1_adam, runargs.beta1_adam)
pretrain = False #################### Set to False
kickstart = True

if runargs.full_loss:
    project_2 = dc(project)
    project_2.batch_size = 1024 #2**12
    project_2.in_memory = False
    project_2.steps_per_epoch = None
    train_loader_full,_,_,_,_ = train_load(project_2)

def loss_fct_mcmc_full(model):
    # this is just some quick function to iterate over the full dataloader to calculate the full loss. I know it is not the nicest style to use globally defined variables in a function etc.
    with torch.no_grad():
        full_loss = 0
        for i_batch, (X, y, _ )in enumerate(train_loader_full):
            inputs = [X[k].to(device) for k in data_config.input_names]
            label = y[data_config.label_names[0]].long().to(device)
            count = label.shape[0]
            try:
                mask = y[data_config.label_names[0] + '_mask'].bool().to(device)
            except KeyError:
                mask = None
            model_output = model(*inputs)
            logits, label, _ = _flatten_preds(model_output, label=label, mask=mask)
            full_loss += loss_func_MCMC(logits, label)*count
        del model_output, inputs, label
    return full_loss

loop_kwargs = {
             'MH': True, #this is a little more than x2 runtime
             'verbose': False,
             'sigma_adam_dir': sum(p.numel() for p in model_MCMC.parameters())/runargs.sigma_adam_dir_denom if runargs.sigma_adam_dir_denom!=0 else 0, 
             'sigma_factor': 1,
             'extended_doc_dict': False,
             'full_loss': loss_fct_mcmc_full if runargs.full_loss else None,
}

if optim_str == 'Adam':
    optim = torch.optim.Adam(model_MCMC.parameters(), lr=lr, betas=betas_adam)
    optim_str += f'betas{betas_adam}'
elif optim_str == 'SGD':
    optim = torch.optim.SGD(model_MCMC.parameters(), lr=lr)
else:
    try: 
        optim = getattr(torch.optim, optim_str)(lr = lr)
    except:
        raise Exception(f'{optim_str} is not a valid optimizer')
scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, lr_decay)

path_str = f'/ParticleNet_{optim_str}MCMC_lr{lr}_lrdecay{lr_decay}_temp{temp}_sigma{sigma}_sigmaadam{loop_kwargs["sigma_adam_dir"]}_sigmafactor{loop_kwargs["sigma_factor"]}/'

if False:
    ep_load = 98
    optim.load_state_dict(torch.load(model_path + path_str + '_epoch-%d_optimizer.pt' % ep_load))
    model_MCMC.load_state_dict(torch.load( model_path + path_str + '_epoch-%d_state.pt' % ep_load))
    if runargs.full_loss:
        path_str = path_str[:-1] + f'_from{ep_load}/'
    
if project.batch_size != 512:
    path_str = path_str[:-1] + f'_bs{project.batch_size}/'
if pretrain:
    path_str= path_str[:-1] + f'_pretrain/'
    project.num_epochs *= 2
if runargs.full_loss:
    path_str = path_str[:-1] + f'_full_loss/'
    
load_existing_model = path_str in os.listdir(model_path)
project.model_prefix = model_path + path_str
_ = mkdir(project.model_prefix)

if not "Acc" in os.listdir(project.model_prefix):
    mkdir(project.model_prefix+"Acc/")
if not "Loss" in os.listdir(project.model_prefix):
    mkdir(project.model_prefix+"Loss/")
project.log = log_path + '/ParticleNet_Test_MCMC.log'

scheduler._update_per_step = True
scheduler.min_lr = min_lr

if load_existing_model:
    l = os.listdir(project.model_prefix)
    numbers = [int(s.split("-")[1].split("_")[0]) for s in l if "state" in s]
    project.load_epoch = max(numbers)
    if project.load_epoch >= project.num_epochs-1:
        raise AssertionError("Model is already trained")
    kickstart = False

    optim.load_state_dict(torch.load(project.model_prefix + '_epoch-%d_optimizer.pt' % project.load_epoch))
    model_MCMC.load_state_dict(torch.load(project.model_prefix + '_epoch-%d_state.pt' % project.load_epoch))

    print("loaded ", project.model_prefix + + '_epoch-%d_state.pt' % project.load_epoch)

kickstart_offset = 0 if kickstart else 5
offline_tb_MCMC = tb_helper_offline({"Loss/train":  project.steps_per_epoch*project.num_epochs+1+kickstart_offset, 
                                    "Acc/train":  project.steps_per_epoch*project.num_epochs+1+kickstart_offset,
                                    "Loss/eval":  project.num_epochs, 
                                    "Acc/eval":  project.num_epochs,
                                    "Acceptance_rate": project.steps_per_epoch*project.num_epochs+1+kickstart_offset,
                                    "lr": project.steps_per_epoch*project.num_epochs+1+kickstart_offset, 
                                    "time_per_step": project.steps_per_epoch*project.num_epochs+1+kickstart_offset,
                                    "time (epoch)": project.num_epochs},
                                    project.model_prefix)
if load_existing_model:
    offline_tb_MCMC.load()

MCMC = AdamMCMC(model_MCMC, optim, temp, sigma, n_points = project.len_data_train)

# %%
print(f"initiated model with {sum(p.numel() for p in model_MCMC.parameters())} parameters")

best_valid_metric = np.inf if project.regression_mode else 0
grad_scaler = torch.cuda.amp.GradScaler() if project.use_amp else None
for epoch in range(project.num_epochs):
    if project.load_epoch is not None:
        if epoch <= project.load_epoch:
            continue

    if pretrain and epoch < project.num_epochs//2: #for small sigma train one epoch to prevent the algorithm from getting stuck
        train(model_MCMC, loss_func_MCMC, optim, scheduler, train_loader, device, epoch,
              steps_per_epoch=project.steps_per_epoch, grad_scaler=grad_scaler, tb_helper=offline_tb_MCMC)
    
    else:
        if kickstart:
            train(model_MCMC, loss_func_MCMC, optim, scheduler, train_loader, device, epoch,
              steps_per_epoch=kickstart_offset, grad_scaler=grad_scaler, tb_helper=offline_tb_MCMC)
            kickstart = False
        train_classification_MCMC(model_MCMC, optim, loss_func_MCMC, MCMC, scheduler, train_loader, device, epoch,
            steps_per_epoch=project.steps_per_epoch, grad_scaler=grad_scaler, tb_helper=offline_tb_MCMC, loop_kwargs = loop_kwargs)
        
    if project.model_prefix and (project.backend is None or project.local_rank == 0):
        dirname = os.path.dirname(project.model_prefix)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
        state_dict = model_MCMC.module.state_dict() if isinstance(
            model_MCMC, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else model_MCMC.state_dict()
        torch.save(state_dict, project.model_prefix + '_epoch-%d_state.pt' % epoch)
        torch.save(optim.state_dict(), project.model_prefix + '_epoch-%d_optimizer.pt' % epoch)

    valid_metric = evaluate(model_MCMC, val_loader, device, epoch, loss_func=loss_func_MCMC,
                            steps_per_epoch=project.steps_per_epoch_val, tb_helper=offline_tb_MCMC)

    print('Epoch #%d: Current validation metric: %.5f (best: %.5f)' %
                (epoch, valid_metric, best_valid_metric))

    offline_tb_MCMC.save()

offline_tb = tb_helper_offline({"Loss/train":  project.steps_per_epoch*project.num_epochs+1, 
                                "Acc/train":  project.steps_per_epoch*project.num_epochs+1}, 
                                model_path + f'/ParticleNet_{project.optimizer}_lr{project.start_lr}_opt{project.optimizer_option}/')
offline_tb.load()

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

        if key in offline_tb.scalars:
            plt.plot(smooth(offline_tb.scalars[key][:len_MCMC], s)[1:], label = "Adam", color = 'C1')
            plt.plot(offline_tb.scalars[key][1:][:len_MCMC], alpha = 0.3, color = 'C1')

    plt.xlabel('steps')
    plt.ylabel(key)
    #plt.xlim(0,240_000)
    plt.grid()
    plt.legend(loc = 'upper right')
    plt.savefig(project.model_prefix+key+'.pdf')
    plt.show()