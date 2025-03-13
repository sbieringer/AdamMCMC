# %%
import numpy as np 
import torch
import os

from src.MCMC_weaver_utils import train_classification as train
from src.util import *

from weaver.train import train_load, model_setup, optim
from weaver.utils.nn.tools import evaluate_classification as evaluate
from argparse import ArgumentParser

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

parser = ArgumentParser()
parser.add_argument('--beta1_adam', type=float, default=0.99)
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
project.num_epochs = 100*512//project.batch_size
project.gpus = 0
project.start_lr = 1e-3
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
project.optimizer_option = [('betas', f'({runargs.beta1_adam}, {runargs.beta1_adam})')]
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

project.model_prefix = model_path + f'/ParticleNet_{project.optimizer}_lr{project.start_lr}_opt{project.optimizer_option}/'
if project.batch_size != 512:
    project.model_prefix = project.model_prefix[:-1] + f'_bs{project.batch_size}/'
_ = mkdir(project.model_prefix)

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
model, model_info, loss_func = model_setup(project, data_config, device=device)
model = model.to(device)
model.load_state_dict(torch.load('./models/base_model.pt'))

if project.optimizer == "sgd":
    try:
        opt = torch.optim.SGD(model.parameters(), project.start_lr, **project.optimizer_options)
    except:
        opt = torch.optim.SGD(model.parameters(), project.start_lr)
    scheduler = None
else:
    opt, scheduler = optim(project, model, device)


# %%
print(f"initiated model with {sum(p.numel() for p in model.parameters())} parameters")

# %%
offline_tb = tb_helper_offline({"Loss/train":  project.steps_per_epoch*project.num_epochs+1, 
                                "Acc/train":  project.steps_per_epoch*project.num_epochs+1, 
                                "time_per_step": project.steps_per_epoch*project.num_epochs+1,
                                "Loss/eval":  project.num_epochs, 
                                "Acc/eval":  project.num_epochs,
                                "time (epoch)": project.num_epochs},
                                project.model_prefix)

best_valid_metric = np.inf if project.regression_mode else 0
grad_scaler = torch.cuda.amp.GradScaler() if project.use_amp else None
for epoch in range(project.num_epochs):
    if project.load_epoch is not None:
        if epoch <= project.load_epoch:
            continue

    train(model, loss_func, opt, scheduler, train_loader, device, epoch,
        steps_per_epoch=project.steps_per_epoch, grad_scaler=grad_scaler, tb_helper=offline_tb)
    
    if project.model_prefix and (project.backend is None or project.local_rank == 0):
        dirname = os.path.dirname(project.model_prefix)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
        state_dict = model.module.state_dict() if isinstance(
            model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else model.state_dict()
        torch.save(state_dict, project.model_prefix + '_epoch-%d_state.pt' % epoch)
        torch.save(opt.state_dict(), project.model_prefix + '_epoch-%d_optimizer.pt' % epoch)

    valid_metric = evaluate(model, val_loader, device, epoch, loss_func=loss_func,
                            steps_per_epoch=project.steps_per_epoch_val, tb_helper=offline_tb)

    print('Epoch #%d: Current validation metric: %.5f (best: %.5f)' %
                (epoch, valid_metric, best_valid_metric))

    offline_tb.save()