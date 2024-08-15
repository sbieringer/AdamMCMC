# %%
import torch
import numpy as np
from weaver.utils.nn.tools import _flatten_preds, _logger, Counter

import sys
sys.path.append('../..')

import time
import tqdm

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
            load = np.load(self.path + key + '.npy')
            self.scalars[key][:len(load)] = load

def train_classification_MCMC(
        model, opt, loss_func, MCMC, scheduler, train_loader, dev, epoch, steps_per_epoch=None, grad_scaler=None,
        tb_helper=None, loop_kwargs={}):
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

   # with tqdm.tqdm(train_loader) as tq:
    for X, y, _ in train_loader:
        start_time_epoch = time.time()
        inputs = [X[k].to(dev) for k in data_config.input_names]
        label = y[data_config.label_names[0]].long().to(dev)
        entry_count += label.shape[0]
        try:
            mask = y[data_config.label_names[0] + '_mask'].bool().to(dev)
        except KeyError:
            mask = None
        opt.zero_grad()
        with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
            t1 = time.time()
            model_output = model(*inputs)
            logits, label, _ = _flatten_preds(model_output, label=label, mask=mask)
            loss = loss_func(logits, label)
            t2 = time.time()

            loss_fct_mcmc = lambda: loss_func(_flatten_preds(model(*inputs), label=label, mask=mask)[0], label)

            _,a,b,_,_ = MCMC.step(loss_fct_mcmc, **loop_kwargs)

        if b: 
            maxed_out_mbb_batches  = 0
        if maxed_out_mbb_batches > 100:
            print('MBB sampling is not convergent, reinitializing the chain')
            MCMC.start = True #This is a hot fix to not get the optimizer stuck to often

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

        # tq.set_postfix({
        #     'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
        #     'Loss': '%.5f' % loss,
        #     'AvgLoss': '%.5f' % (total_loss / num_batches),
        #     'Acc': '%.5f' % (correct / num_examples),
        #     'AvgAcc': '%.5f' % (total_correct / count)})

        time_diff_epoch = time.time() - start_time_epoch - (t2-t1)
        if tb_helper:
            tb_helper.write_scalars([
                ("Loss/train", loss, tb_helper.batch_train_count + num_batches),
                ("Acc/train", correct / num_examples, tb_helper.batch_train_count + num_batches),
                ('Acceptance_rate', a, tb_helper.batch_train_count + num_batches),
                ('lr', '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'], tb_helper.batch_train_count + num_batches),
                ('time_per_step', time_diff_epoch, tb_helper.batch_train_count + num_batches),
            ])
            
            if tb_helper.custom_fn:
                with torch.no_grad():
                    tb_helper.custom_fn(model_output=model_output, model=model,
                                        epoch=epoch, i_batch=num_batches, mode='train')

        if steps_per_epoch is not None and num_batches >= steps_per_epoch:
            break

    time_diff = time.time() - start_time - num_batches*(t2-t1)
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (entry_count, entry_count / time_diff))
    _logger.info('Train AvgLoss: %.5f, AvgAcc: %.5f' % (total_loss / num_batches, total_correct / count))
    _logger.info('Train class distribution: \n    %s', str(sorted(label_counter.items())))

    if tb_helper:
        tb_helper.write_scalars([
            ("Loss/train (epoch)", total_loss / num_batches, epoch),
            ("Acc/train (epoch)", total_correct / count, epoch),
            ("time (epoch)", time_diff, epoch),
        ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode='train')
        # update the batch state
        tb_helper.batch_train_count += num_batches

    if scheduler and not getattr(scheduler, '_update_per_step', False):
        scheduler.step()


def train_classification(
        model, loss_func, opt, scheduler, train_loader, dev, epoch, steps_per_epoch=None, grad_scaler=None,
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
    #with tqdm.tqdm(train_loader) as tq:
    for X, y, _ in train_loader:
        start_time_epoch = time.time()
        inputs = [X[k].to(dev) for k in data_config.input_names]
        label = y[data_config.label_names[0]].long().to(dev)
        entry_count += label.shape[0]
        try:
            mask = y[data_config.label_names[0] + '_mask'].bool().to(dev)
        except KeyError:
            mask = None
        opt.zero_grad()

        with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
            model_output = model(*inputs)
            logits, label, _ = _flatten_preds(model_output, label=label, mask=mask)
            loss = loss_func(logits, label)
        if grad_scaler is None:
            loss.backward()
            opt.step()
        else:
            grad_scaler.scale(loss).backward()
            grad_scaler.step(opt)
            grad_scaler.update()

        if scheduler and getattr(scheduler, '_update_per_step', False):
            scheduler.step()

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
            'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
            'Loss': '%.5f' % loss,
            'AvgLoss': '%.5f' % (total_loss / num_batches),
            'Acc': '%.5f' % (correct / num_examples),
            'AvgAcc': '%.5f' % (total_correct / count)})

        time_diff_epoch = time.time()-start_time_epoch
        if tb_helper:
            tb_helper.write_scalars([
                ("Loss/train", loss, tb_helper.batch_train_count + num_batches),
                ("Acc/train", correct / num_examples, tb_helper.batch_train_count + num_batches),
                ('time_per_step', time_diff_epoch, tb_helper.batch_train_count + num_batches),

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
            ("time (epoch)", time_diff, epoch),
        ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode='train')
        # update the batch state
        tb_helper.batch_train_count += num_batches

    if scheduler and not getattr(scheduler, '_update_per_step', False):
        scheduler.step()