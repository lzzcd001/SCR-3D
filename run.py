import argparse
import torch
from unsup3d import setup_runtime, Trainer, Unsup3D
import numpy as np


## runtime arguments
parser = argparse.ArgumentParser(description='Training configurations.')
parser.add_argument('--config', default=None, type=str, help='Specify a config file path')
parser.add_argument('--gpu', default=None, type=int, help='Specify a GPU device')
parser.add_argument('--num_workers', default=4, type=int, help='Specify the number of worker threads for data loaders')
parser.add_argument('--seed', default=0, type=int, help='Specify a random seed')
parser.add_argument('--dense_order', nargs='+', type=int, help='')
parser.add_argument('--adj', nargs='+', type=int, help='')
parser.add_argument('--image_folder', type=str, help='')
args = parser.parse_args()

assert not (args.dense_order is not None and args.adj is not None)

if args.dense_order is not None:
    print("\n\nDense order: ", args.dense_order)
elif args.adj is not None:
    print("\n\nAdjancency")
    print(np.array(args.adj).reshape((4, 4)))
    print("\n\n")

## set up
cfgs = setup_runtime(args)
trainer = Trainer(cfgs, Unsup3D, dense_order=args.dense_order, adjacency=args.adj, seed=args.seed, image_folder=args.image_folder)
run_train = cfgs.get('run_train', False)
run_test = cfgs.get('run_test', False)
run_viz = cfgs.get('run_viz', False)
run_viz_interp = cfgs.get('run_viz_interp', False)

## run
if run_train:
    trainer.train()
if run_test:
    trainer.test()
if run_viz:
    trainer.viz()
if run_viz_interp:
    trainer.viz_interp()

