import argparse
import torch
from unsup3d import setup_runtime, BoDenseTrainer, Unsup3D

## runtime arguments
parser = argparse.ArgumentParser(description='Training configurations.')
parser.add_argument('--config', default=None, type=str, help='Specify a config file path')
parser.add_argument('--gpu', default=None, type=int, help='Specify a GPU device')
parser.add_argument('--num_workers', default=4, type=int, help='Specify the number of worker threads for data loaders')
parser.add_argument('--seed', default=0, type=int, help='Specify a random seed')
parser.add_argument('--train_max_iter', default=5000, type=int, help='')
parser.add_argument('--bo_obj', default='Combined', type=str, choices=['Combined', 'Image_Masked', 'NorErr_masked'])
args = parser.parse_args()

## set up
cfgs = setup_runtime(args)
trainer = BoDenseTrainer(cfgs, Unsup3D)

trainer.train()

# ## run
# if run_train:
#     trainer.train()
# if run_test:
#     trainer.test()
