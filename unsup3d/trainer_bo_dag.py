import os
import glob
from datetime import datetime
import numpy as np
import torch
from . import meters
from . import utils
from .dataloaders import get_data_loaders

import math
import itertools

from . import Trainer


import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

import botorch
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from .kernels import DAGKernel
import random
import itertools


def get_dag(assignment, order):
    dag = torch.zeros(4, 4)
    inv_order = list(reversed(order))
    coord_list = [
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 3),
        (2, 3)
    ]
    for ind, (i, j) in enumerate(coord_list):
        dag[inv_order[i], inv_order[j]] = assignment[ind]
    
    return dag


def get_all_dags(order):
    dags = torch.zeros(2**6, 4, 4)
    inv_order = list(reversed(order))
    coord_list = [
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 3),
        (2, 3)
    ]
    for batch_ind, assignment in enumerate(itertools.product([0, 1], repeat=6)):
        for ind, (i, j) in enumerate(coord_list):
            dags[batch_ind, inv_order[i], inv_order[j]] = assignment[ind]
    
    return dags


# def greedy_optimize(inverse_order, acq_function, gp_model=None, num_order=5):
#     best_score = acq_function(inverse_order)
#     current_inv_order = inverse_order
#     run_flag = True
#     count = 0
#     curr_best = best_score
#     tmp_best = best_score
#     while run_flag:
#         # print(current_inv_order)
#         # print('-------------')
#         tmp_best_inv_order = current_inv_order.clone().detach()
#         perms = list(itertools.combinations(range(num_order), 2))
#         random.shuffle(perms)
#         for i, j in perms:
#             tmp_inv_order = current_inv_order.clone().detach()
#             tmp_inv_order[0, i] = current_inv_order[0, j]
#             tmp_inv_order[0, j] = current_inv_order[0, i]

#             # if existing_list is not None and (torch.abs(tmp_inv_order.view(1, -1) - existing_list).sum(dim=-1) == 0).any():
#             #     continue

#             score = acq_function(tmp_inv_order)
#             if score > tmp_best:
#                 print(tmp_inv_order, tmp_best_inv_order, score.item(), tmp_best.item())
#                 tmp_best = score
#                 tmp_best_inv_order = tmp_inv_order
#         # print(best_score, curr_best)
#         if count != 0 and tmp_best <= curr_best:
#             break
#         else:
#             current_inv_order = tmp_best_inv_order
#             curr_best = tmp_best
#         count += 1

#     return get_inverse_order(current_inv_order[0].cpu().detach().numpy()), curr_best.cpu().detach().numpy().item()


def weight_reset(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()

class BoDAGTrainer():
    def __init__(self, cfgs, model, dense_order):
        self.cfgs = cfgs
        self.model_fn = model
        self.device = cfgs.get('device', 'cpu')
        # self.no_conf = cfgs.get('no_conf', False)

        self.lamda = cfgs.get('lambda', 0.5)
        self.bo_steps = cfgs.get('bo_steps', 24)
        self.bo_obj = cfgs.get('bo_obj', 'Combined')
        self.dense_order = dense_order
        if dense_order is not None:
            dense_order_str = ''.join([str(x) for x in dense_order])
            self.checkpoint_dir = cfgs.get('checkpoint_dir', 'results_bo_dag') + f'/bo_dag_{dense_order_str}'
        else:
            self.checkpoint_dir = cfgs.get('checkpoint_dir', 'results_bo_dag') + f'/bo_dag'

        self.logger = SummaryWriter(os.path.join(self.checkpoint_dir, f'logs', datetime.now().strftime("%Y%m%d-%H%M%S")))
        print(os.path.join(self.checkpoint_dir, f'logs', datetime.now().strftime("%Y%m%d-%H%M%S")))

    def train(self):


        ############################# BO Init ##############################
        # num_factors = 4 if self.no_conf else 5
        dag = torch.zeros(16)
        best_dag = dag
        dag_kernel = DAGKernel(lamda=self.lamda)
        best_score = -1e30
        print('init dag:', dag)
        trainer = None

        dag_candidates = get_all_dags(self.dense_order).to(self.device).view(-1, 16)
        mask = torch.ones(dag_candidates.size(0)).bool()
        mask = torch.logical_and(mask, (dag_candidates.cpu() - dag.view(1, 16)).abs().sum(dim=1).bool())
        assert torch.sum(mask) == dag_candidates.size(0) - 1
        print('size of candidates', mask.sum())




        ############################# BO Steps ##############################

        train_X = None
        train_Y = None
        print("training starts...\n\n")

        t1 = trange(self.bo_steps, desc="best score: N/A, prev score: N/A")
        bo_count = 0
        for ind in t1:
            subexp_name = ''.join([str(int(x)) for x in dag.view(-1).detach().cpu().numpy().tolist()])
            self.logger.add_text('dag', subexp_name, ind)

            if trainer is not None:
                del trainer
                torch.cuda.empty_cache()

            trainer = Trainer(self.cfgs, self.model_fn, adjacency=dag.view(4, 4))
            trainer.use_logger = False ### HACK: Manually set to False
            metric = trainer.train_val()
            loss = metric.get_value(self.bo_obj)
            score = - loss
            if score > best_score:
                best_score = score
                best_dag = dag
            
            print('current dag:', dag)

            if train_X is None:
                train_X = dag.clone().detach().view(1, 16).to(self.device)
                # train_Y = torch.tensor([test_psnr]).to(device).view(1, 1)
                train_Y = torch.tensor([score]).to(self.device).view(1, 1)
            else:
                train_X = torch.cat([train_X.to(self.device), dag.view(1, 16).to(self.device)], dim=0)
                # train_Y = torch.cat([train_Y, torch.tensor([test_psnr]).to(device).unsqueeze(-1)])
                train_Y = torch.cat([train_Y.to(self.device), torch.tensor([score]).to(self.device).unsqueeze(-1)], dim=0)
            
            gp_model = botorch.models.SingleTaskGP(train_X, train_Y, covar_module=dag_kernel)

            # noise_std = 1e-3
            # gp_model = botorch.models.FixedNoiseGP(train_X, train_Y, (train_Y * 0.0 + noise_std).clone().pow(2), covar_module=position_kernel)
            mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
            mll.train()
            if bo_count > 0:
                botorch.optim.fit.fit_gpytorch_torch(mll, optimizer_cls=torch.optim.Adam, options={'lr': 1e-1})
            mll.eval()
            print('lambda', dag_kernel.lamda.item())

            # acq_function = botorch.acquisition.analytic.UpperConfidenceBound(gp_model, beta=1.0)
            # acq_function = botorch.acquisition.analytic.NoisyExpectedImprovement(gp_model, train_X)
            acq_function = botorch.acquisition.analytic.ExpectedImprovement(gp_model, best_score)


            # random_order = np.random.permutation(num_factors)
            # random_inv_order = get_inverse_order(random_order).to(self.device).unsqueeze(0)
            # new_order, ei_score = greedy_optimize(random_inv_order, acq_function)
            # order = new_order.cpu().detach().numpy().astype(np.int32)


            # print(train_X.size(), train_Y.size(), dag_candidates.unsqueeze(-2).size())
            ei_scores = acq_function(dag_candidates[mask].unsqueeze(-2))
            best_index = torch.argmax(ei_scores)
            ei_score = ei_scores[best_index]
            dag = dag_candidates[mask][best_index]

            mask = torch.logical_and(mask, (dag_candidates.cpu() - dag.cpu().view(1, 16)).abs().sum(dim=1).bool())
            assert torch.sum(mask) == dag_candidates.size(0) - 2 - bo_count
            print('size of candidates', mask.sum())






            # t.postfix[1] = best_score.item()
            # t.update()
            t1.set_description('best score: {:f}, prev score: {:f}'.format(best_score, score))
            t1.refresh()
            print('last sample', train_X[-1].cpu().view(16))
            print('best dag', best_dag)
            print('next dag', dag)

            self.logger.add_scalar('score', score, ind)
            self.logger.add_scalar('best_score', best_score, ind)
            self.logger.add_scalar('ei_score', ei_score, ind)

            bo_count += 1