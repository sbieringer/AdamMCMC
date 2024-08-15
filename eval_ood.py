import numpy as np 
import matplotlib.pyplot as plt
import torch
import os
from copy import deepcopy as dc
import tqdm

from top_landscape.particle_transformer.dataloader import read_file
from weaver.train import train_load, test_load, model_setup, optim
from weaver.utils.data.fileio import _read_parquet
from weaver.utils.nn.tools import _flatten_preds, _logger, Counter

from argparse import ArgumentParser, ArgumentTypeError

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

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
    
model_path = './top_landscape/models/'
data_path = "./top_landscape/data/"

data_config = "./top_landscape/particle_transformer/data/TopLandscape/top_kin.yaml"
model_config = "./top_landscape/particle_transformer/networks/example_ParticleNet.py"

log_path = './top_landscape/logs/'

class empty():
    def __init__(self):
        pass
    
parser = ArgumentParser()
parser.add_argument('--sigma', type=float, default=1)
parser.add_argument('--signatures_half', type=int, default=1)
runargs = parser.parse_args()

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
project.num_epochs = 60
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
    
train_loader, val_loader, data_config, train_input_names, train_label_names = train_load(project)
model_MCMC, model_info_MCMC, loss_func_MCMC = model_setup(project, data_config, device=device)
model_MCMC = model_MCMC.to(device)
model_MCMC.device = device
model_MCMC.load_state_dict(torch.load('./models/base_model.pt'))

data_OOD_path = "./JetClass/data/JetClass/train_100M/"
data_OOD_config = "./top_landscape/particle_transformer/data/JetClass/JetClass_kin.yaml"

project.data_config = data_OOD_config

def get_logits_and_labels_and_jetpt(model, project, test_loaders = None, data_config = None):
    if test_loaders is None or data_config is None:
        test_loaders, data_config = test_load(project)
    for _, get_test_loader in test_loaders.items():
        test_loader = get_test_loader()
        continue

    logits_list = []
    label_list = []
    jet_pT_list = []

    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for X, y, Z in tq:
                # X, y: torch.Tensor; Z: ak.Array
                inputs = [X[k].to(device) for k in data_config.input_names]
                label = y[data_config.label_names[0]].long().to(device)
                try:
                    mask = y[data_config.label_names[0] + '_mask'].bool().to(device)
                except KeyError:
                    mask = None
                model_output = model(*inputs)
                logits, label, mask = _flatten_preds(model_output, label=label, mask=mask)

                #_, preds = logits.max(1)

                logits_list.append(logits)
                label_list.append(label)
                jet_pT_list.append(Z['jet_pt'])

    logits = torch.cat(logits_list)
    label = torch.cat(label_list)
    jet_pt = torch.cat(jet_pT_list) 

    return logits, label, jet_pt, test_loaders, data_config

def calculate_preds_OOD(model, project, path_tmp, data_strs,  n_epochs = 50, n_samples = 10, stepsize=1):
    for data_str in data_strs:
        test_loaders, data_config = None, None
        if False: #data_str in os.listdir(path_tmp):
            mutual_info = np.load(path_tmp + data_str + f'/mutual_info_{data_str}.npy')
        else:
            path_tmp_save = mkdir(path_tmp + data_str + '/')
            project.data_test = [data_OOD_path + data_str + '_000.root']
            for epoch in range(n_epochs-n_samples*stepsize, n_epochs, stepsize):
                model.load_state_dict(torch.load(path_tmp + f'_epoch-{epoch}_state.pt'))

                logits, labels, jet_pt, test_loaders, data_config = get_logits_and_labels_and_jetpt(model, project, test_loaders, data_config)
                
                if epoch == n_epochs-n_samples*stepsize:
                    logits_out = logits.unsqueeze(-1)
                else:
                    logits_out = torch.cat([logits_out, logits.unsqueeze(-1)], -1)

            mean_logits = torch.mean(logits_out, -1)
            _, preds = torch.mean(logits_out, -1).max(1)
            np.save(path_tmp_save + f"logits_out_{data_str}.npy", logits_out.numpy(force=True))
            np.save(path_tmp_save + f"mean_logits_{data_str}.npy", mean_logits.numpy(force=True))
            np.save(path_tmp_save + f"preds_{data_str}.npy", preds.numpy(force=True))
            np.save(path_tmp_save + f"labels_{data_str}.npy", labels.numpy(force=True))
            np.save(path_tmp_save + f"jet_pt_{data_str}.npy", jet_pt.numpy(force=True))

            scores = torch.softmax(logits_out.float(), dim=1).numpy(force=True)
            post_pred = np.mean(scores, -1)
            post_pred_entropy = -np.sum(post_pred*np.log(post_pred), 1)
            np.save(path_tmp_save + f"post_pred_entropy_{data_str}.npy", post_pred_entropy)

            entropy_expect = np.mean(-np.sum(scores*np.log(scores), 1), -1)
            np.save(path_tmp_save + f"entropy_expect_{data_str}.npy", entropy_expect)

            mutual_info = post_pred_entropy - entropy_expect
            np.save(path_tmp_save + f"mutual_info_{data_str}.npy", mutual_info)
        
def get_preds_OOD(path_tmp, data_str):
    mean_logits = np.load(path_tmp + data_str + f'/mean_logits_{data_str}.npy')
    preds = np.load(path_tmp + data_str + f'/preds_{data_str}.npy')
    labels = np.load(path_tmp + data_str + f'/labels_{data_str}.npy')
    jet_pt = np.load(path_tmp + data_str + f'/jet_pt_{data_str}.npy')
    post_pred_entropy = np.load(path_tmp + data_str + f'/post_pred_entropy_{data_str}.npy')
    entropy_expect = np.load(path_tmp + data_str + f'/entropy_expect_{data_str}.npy')
    mutual_info = np.load(path_tmp + data_str + f'/mutual_info_{data_str}.npy')

    return mean_logits, preds, labels, jet_pt, post_pred_entropy, entropy_expect, mutual_info

lr = 1e-3
min_lr = 1e-6
lr_decay = 1# 0.998 # 0.995
temp = 1.
sigma = runargs.sigma
sigma_adam_dir_denom = 100
sigma_factor = 1
betas_adam = (0.99, 0.99)
optim_str = 'Adam'
optim_str += f'betas{betas_adam}'

n_epochs = 98
n_samples = 10
stepsize = 5

if runargs.signatures_half == 1:
    data_strs = ['TTBar', 'ZJetsToNuNu', 'HToBB', 'HToCC', 'HToGG', ]
else: 
    data_strs = ['HToWW2Q1L', 'HToWW4Q', 'TTBarLep', 'WToQQ', 'ZToQQ']

sigma_adam_dir =  sum(p.numel() for p in model_MCMC.parameters())/sigma_adam_dir_denom

path_tmp = model_path + f'/ParticleNet_{optim_str}MCMC_lr{lr}_lrdecay{lr_decay}_temp{temp}_sigma{sigma}_sigmaadam{sigma_adam_dir}_sigmafactor{sigma_factor}/'

calculate_preds_OOD(model_MCMC, project, path_tmp, data_strs,  n_epochs = n_epochs, n_samples = n_samples, stepsize = stepsize)

