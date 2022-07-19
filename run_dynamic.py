import argparse
import torch
from unsup3d import setup_runtime, DynamicTrainer, DynamicUnsup3D


## runtime arguments
parser = argparse.ArgumentParser(description='Training configurations.')
parser.add_argument('--config', default=None, type=str, help='Specify a config file path')
parser.add_argument('--gpu', default=None, type=int, help='Specify a GPU device')
parser.add_argument('--num_workers', default=4, type=int, help='Specify the number of worker threads for data loaders')
parser.add_argument('--seed', default=0, type=int, help='Specify a random seed')
parser.add_argument('--dynamic_emb', type=str, default='attention', choices=['attention', 'dense', 'cos'])
parser.add_argument('--dense_order', nargs='+', type=int, help='')
parser.add_argument('--no_dag_loss', action='store_true', default=False)
args = parser.parse_args()

## set up
cfgs = setup_runtime(args)
trainer = DynamicTrainer(cfgs, DynamicUnsup3D, seed=args.seed, dense_order=args.dense_order)
run_train = cfgs.get('run_train', False)
run_test = cfgs.get('run_test', False)
run_viz = cfgs.get('run_viz', False)


## run
if run_train:
    trainer.train()
if run_test:
    trainer.test()
if run_viz:
    trainer.viz()
