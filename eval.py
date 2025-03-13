import numpy as np 
import matplotlib.pyplot as plt
import torch
import os
import tqdm
from src.MCMC_weaver_utils import tb_helper_offline
from src.util import mkdir

from weaver.train import train_load, test_load, model_setup
from weaver.utils.nn.tools import _flatten_preds

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#####################
### intialization ###
#####################

model_path = './top_landscape/models/'
data_path = "./top_landscape/data/"

data_config = "./top_landscape/particle_transformer/data/TopLandscape/top_kin.yaml"
model_config = "./top_landscape/particle_transformer/networks/example_ParticleNet.py"

log_path = './top_landscape/logs/'

class empty():
    def __init__(self):
        pass

project = empty()
project.data_train = [data_path + '/train_file.parquet']
project.data_val = [data_path + '/val_file.parquet']
project.data_test = [data_path + '/test_file.parquet']
project.data_config = data_config
project.network_config = model_config
project.num_workers = 1
project.fetch_step = 1
project.in_memory = True
project.batch_size = 512
project.samples_per_epoch = 2400*project.batch_size #2400
project.samples_per_epoch_val = 100*project.batch_size #800
project.num_epochs = 1
project.gpus = 0
project.start_lr = 5e-6 #1e-2
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
model, model_info, loss_func = model_setup(project, data_config, device=device)
model = model.to(device)

offline_tb = tb_helper_offline({"Loss/train":  project.steps_per_epoch*project.num_epochs+1, 
                                "Acc/train":  project.steps_per_epoch*project.num_epochs+1}, 
                                project.model_prefix)


def get_logits_and_labels(model, project, test_loaders = None, data_config = None,  batches = None):
    '''
    the evalutation function returning logits predictions and true labels for a model and dataloader
    '''
    if test_loaders is None or data_config is None:
        test_loaders, data_config = test_load(project)
    for _, get_test_loader in test_loaders.items():
        test_loader = get_test_loader()
        continue

    logits_list = []
    label_list = []

    i = 0
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for X, y, Z in tq:
                i += 1
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
                if batches is not None:
                    if i == batches:
                        break

    logits = torch.cat(logits_list)
    label = torch.cat(label_list)

    return logits, label, test_loaders, data_config

n_epochs = 98
n_samples = 10
stepsize = 5

test_loaders, data_config = None, None

#########################################
### specify the runs to evaluate here ###
#########################################

lr = 1e-3 
min_lr = 1e-6
lr_decay = 1
temp = 1.
sigma = 2.0
sigma_factor = 1
betas_adam = (0.99,0.99)

optim_str = 'Adam'
optim_str += f'betas{betas_adam}'

sigma_adam_dir_denom = 0 #100

sigma_adam_dir =  sum(p.numel() for p in model.parameters())/sigma_adam_dir_denom if sigma_adam_dir_denom!= 0 else 0

path_tmp = model_path + f'/ParticleNet_{optim_str}MCMC_lr{lr}_lrdecay{lr_decay}_temp{temp}_sigma{sigma}_sigmaadam{sigma_adam_dir}_sigmafactor{sigma_factor}/'

##########################
### run the evaluation ###
##########################

for epoch in range(n_epochs-n_samples*stepsize, n_epochs, stepsize):
    model.load_state_dict(torch.load(path_tmp + f'_epoch-{epoch}_state.pt'))

    logits, labels, test_loaders, data_config = get_logits_and_labels(model, project, test_loaders, data_config)

    if epoch == n_epochs-n_samples*stepsize:
        logits_out = logits.unsqueeze(-1)
    else:
        logits_out = torch.cat([logits_out, logits.unsqueeze(-1)], -1)

mean_logits = torch.mean(logits_out, -1)
_, preds = mean_logits.max(1)
np.save(path_tmp +"logits_out.npy", logits_out.numpy(force=True))
np.save(path_tmp +"mean_logits.npy", mean_logits.numpy(force=True))
np.save(path_tmp +"preds.npy", preds.numpy(force=True))
np.save(path_tmp +"labels.npy", labels.numpy(force=True))

scores = torch.softmax(logits_out.float(), dim=1).numpy(force=True)+1e-10
post_pred = np.mean(scores.astype(np.longdouble), -1)
post_pred_entropy = -np.sum(np.nan_to_num(post_pred*np.log(post_pred)), 1)
np.save(path_tmp +"post_pred_entropy.npy", post_pred_entropy)

log_scores = logits_out.numpy(force=True).astype(np.longdouble) - np.nan_to_num(np.log(np.sum(np.exp(logits_out.numpy(force=True).astype(np.longdouble)), 1, keepdims=True)))
entropy_expect = np.mean(-np.sum(scores.astype(np.longdouble)*log_scores, 1), -1)
np.save(path_tmp +"entropy_expect.npy", entropy_expect)

mutual_info = post_pred_entropy - entropy_expect
np.save(path_tmp +"mutual_info.npy", mutual_info)
