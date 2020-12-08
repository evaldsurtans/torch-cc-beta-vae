import csv
import logging
import math
from  _datetime import datetime
import os
from collections import OrderedDict
import json

import torch
import numpy as np
import torchvision.datasets
from torchvision.utils import make_grid

from modules import tensorboard_utils
from modules.file_utils import FileUtils

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import argparse
from torch.utils.data import DataLoader
import torch_optimizer as optim
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(add_help=False)

parser.add_argument('-id', default=0, type=int)
parser.add_argument('-sequence_name', default='sequence', type=str)
parser.add_argument('-run_name', default='run', type=str)

parser.add_argument('-model', default='model_1', type=str)
parser.add_argument('-dataset', default='dataset_1_emnist', type=str)

parser.add_argument('-batch_size', default=16, type=int)
parser.add_argument('-learning_rate', default=1e-3, type=float)

parser.add_argument('-optimizer', default='radam', type=str)
parser.add_argument('-huber_loss_delta', default=1.0, type=float)

parser.add_argument('-device', default='cuda', type=str)

parser.add_argument('-epochs', default=20, type=int)
parser.add_argument('-debug_batch_count', default=0, type=int) # 0 = release version

parser.add_argument('-embedding_size', default=32, type=int)

parser.add_argument('-gamma', default=30.0, type=float)
parser.add_argument('-C_0', default=0.0, type=float)
parser.add_argument('-C_n', default=5.0, type=float)
parser.add_argument('-C_interval', default=10000, type=int)

args, args_other = parser.parse_known_args()

path_sequence = f'./results/{args.sequence_name}'
args.run_name += ('-' + datetime.utcnow().strftime(f'%y-%m-%d--%H-%M-%S'))
path_run = f'./results/{args.sequence_name}/{args.run_name}'
FileUtils.createDir(path_run)
path_artifacts = f'./artifacts/{args.sequence_name}/{args.run_name}'
FileUtils.createDir(path_artifacts)
FileUtils.writeJSON(f'{path_run}/args.json', args.__dict__)

summary_writer = tensorboard_utils.CustomSummaryWriter(
    logdir=path_run
)

rootLogger = logging.getLogger()
logFormatter = logging.Formatter("%(asctime)s [%(process)d] [%(thread)d] [%(levelname)s]  %(message)s")
rootLogger.level = logging.INFO #level

base_name = os.path.basename(path_sequence)
fileHandler = logging.FileHandler(f'{path_run}/log-{base_name}.txt')
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

Model = getattr(__import__('modules_core.' + args.model, fromlist=['Model']), 'Model')
DataSet = getattr(__import__('modules_core.' + args.dataset, fromlist=['DataSet']), 'DataSet')

logging.info(f'path_run: {path_run}')
logging.info(json.dumps(args.__dict__, indent=4))

if not torch.cuda.is_available():
    args.device = 'cpu'
    logging.info('CUDA NOT AVAILABLE')
else:
    logging.info('cuda devices: {}'.format(torch.cuda.device_count()))

datasets = OrderedDict({
    'train': DataSet(is_train=True),
    'test': DataSet(is_train=False)
})

x, x_noisy, y = datasets['train'][0]
args.input_size = x.size()

dataloaders = OrderedDict({
    'train': DataLoader(datasets['train'], shuffle=True, batch_size=args.batch_size),
    'test': DataLoader(datasets['test'], shuffle=False, batch_size=args.batch_size)
})
model = Model(args).to(args.device)


# https://pypi.org/project/torch-optimizer/#radam
if args.optimizer == 'radam':
    optimizer = optim.RAdam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    )

def dict_list_append(dict, key, value):
    if key not in dict:
        dict[key] = []
    dict[key].append(value)

metrics_best = {
    'best_test_loss': float('Inf'),
    'best_test_loss_dir': -1
}

count_batches = 0
for epoch in range(1, args.epochs+1):
    logging.info(f'epoch: {epoch}')

    metric_mean = {}
    metrics_list = {}
    for mode, dataloader in dataloaders.items():

        if mode == 'train':
            model = model.train()
            torch.set_grad_enabled(True)
        else:
            model = model.eval()
            torch.set_grad_enabled(False)

        for x, x_noisy, _ in dataloader:

            # plt.subplot(2, 1, 1)
            # plt.imshow(np.transpose(make_grid(x).numpy(), (1, 2, 0)))
            # plt.subplot(2, 1, 2)
            # plt.imshow(np.transpose(make_grid(x_noisy).numpy(), (1, 2, 0)))
            # plt.show()

            if mode == 'train':
                count_batches += 1
            if args.debug_batch_count != 0 and count_batches > args.debug_batch_count: # for debugging
                break

            z_mu, z_sigma, y_prim = model.forward(x_noisy.to(args.device))

            loss_rec = torch.mean((x.to(args.device) - y_prim)**2)

            C = min(args.C_n, (args.C_n - args.C_0) * (count_batches / args.C_interval) + args.C_0)

            kl = z_mu**2 + z_sigma**2 - 1.0 - torch.log(z_sigma**2 + 1e-8)
            kl_means = torch.mean(kl, dim=0) # (32, )
            loss_kl = args.gamma * torch.abs(C - torch.sum(kl_means))

            loss = loss_rec + loss_kl

            loss_scalar = loss.cpu().item()
            loss_rec_scalar = loss_rec.cpu().item()
            loss_kl_scalar = loss_kl.cpu().item()
            if np.isnan(loss_scalar) or np.isinf(loss_scalar):
                logging.error(f'loss_scalar: {loss_scalar} loss_rec_scalar: {loss_rec_scalar} loss_kl_scalar: {loss_kl_scalar}')
                exit()

            dict_list_append(metrics_list, f'{mode}_loss_rec', loss_rec_scalar)
            dict_list_append(metrics_list, f'{mode}_loss_kl', loss_kl_scalar)
            dict_list_append(metrics_list, f'{mode}_loss', loss_scalar)

            if mode == 'train':
                loss.backward()
                optimizer.step()

    for key, value in metrics_list.items():
        value = np.mean(value)
        logging.info(f'{key}: {value}')
        metric_mean[key] = value

        for key_best in metrics_best.keys():
            if f'best_{key}' == key_best:
                direction = metrics_best[f'{key_best}_dir']
                if metrics_best[key_best] * direction < value * direction:
                    torch.save(model.state_dict(), f'{path_artifacts}/best-{key_best}-{args.run_name}.pt')
                    with open(f'{path_artifacts}/best-{key_best}-{args.run_name}.json', 'w') as fp:
                        json.dump(args.__dict__, fp, indent=4)
                    metrics_best[key_best] = value

    summary_writer.add_hparams(
        hparam_dict=args.__dict__,
        metric_dict=metric_mean,
        name=args.run_name,
        global_step=epoch
    )
    summary_writer.flush()
summary_writer.close()




