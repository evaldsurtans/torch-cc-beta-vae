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
parser.add_argument('-debug_batch_count', default=10, type=int) # 0 = release version

parser.add_argument('-loss_huber_delta', default=1, type=float)

parser.add_argument('-test_rollout_steps', default=16, type=int)

args, args_other = parser.parse_known_args()

path_sequence = f'./results/{args.sequence_name}'
args.run_name += ('-' + datetime.utcnow().strftime(f'%y-%m-%d--%H-%M-%S'))
path_run = f'./results/{args.sequence_name}/{args.run_name}'
FileUtils.createDir(path_run)
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

csv_records_run = []

metrics_best = {
    'test_loss': float('Inf'),
    'test_loss_dir': -1
}

for epoch in range(1, args.epochs+1):
    logging.info(f'epoch: {epoch}')

    metric_mean = {}
    for mode, dataloader in dataloaders.items():
        metrics_list = {
            f'{mode}_loss': []
        }

        if mode == 'train':
            model = model.train()
            torch.set_grad_enabled(True)
        else:
            model = model.eval()
            torch.set_grad_enabled(False)

        count_batches = 0
        for x, x_noisy, y in dataloader:

            plt.imshow(np.transpose(make_grid(x).numpy(), (1, 2, 0)))
            plt.show()
            plt.imshow(np.transpose(make_grid(x_noisy).numpy(), (1, 2, 0)))
            plt.show()

            count_batches += 1
            if args.debug_batch_count != 0 and count_batches > args.debug_batch_count: # for debugging
                break

            #TODO

            if mode == 'train':
                #loss.backward()
                optimizer.step()

    for key, value in metrics_list.items():
        value = np.mean(value)
        logging.info(f'{key}: {value}')
        metric_mean[key] = value

        for key_best in metrics_best.keys():
            if key == key_best:
                direction = metrics_best[f'{key_best}_dir']
                if metrics_best[key_best] * direction < value * direction:
                    torch.save(model.state_dict(), f'{path_run}/best-{key_best}-{args.run_name}.pt')
                    with open(f'{path_run}/best-{key_best}-{args.run_name}.json', 'w') as fp:
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




