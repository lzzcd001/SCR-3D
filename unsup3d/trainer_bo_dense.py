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
from .kernels import PositionKernel
import random
import itertools

def get_inverse_order(order):
    inverse_order = np.array(order * 0.0)
    for i in range(order.shape[0]):
        inverse_order[int(order[i])] = i
    #     print(order, inverse_order)
    # print(inverse_order)
    # print(torch.tensor(inverse_order))
    # print('**************')
    return torch.tensor(inverse_order)


def greedy_optimize(inverse_order, acq_function, gp_model=None, num_order=5):
    best_score = acq_function(inverse_order)
    current_inv_order = inverse_order
    run_flag = True
    count = 0
    curr_best = best_score
    tmp_best = best_score
    while run_flag:
        # print(current_inv_order)
        # print('-------------')
        tmp_best_inv_order = current_inv_order.clone().detach()
        perms = list(itertools.combinations(range(num_order), 2))
        random.shuffle(perms)
        for i, j in perms:
            tmp_inv_order = current_inv_order.clone().detach()
            tmp_inv_order[0, i] = current_inv_order[0, j]
            tmp_inv_order[0, j] = current_inv_order[0, i]

            # if existing_list is not None and (torch.abs(tmp_inv_order.view(1, -1) - existing_list).sum(dim=-1) == 0).any():
            #     continue

            score = acq_function(tmp_inv_order)
            if score > tmp_best:
                print(tmp_inv_order, tmp_best_inv_order, score.item(), tmp_best.item())
                tmp_best = score
                tmp_best_inv_order = tmp_inv_order
        # print(best_score, curr_best)
        if count != 0 and tmp_best <= curr_best:
            break
        else:
            current_inv_order = tmp_best_inv_order
            curr_best = tmp_best
        count += 1

    return get_inverse_order(current_inv_order[0].cpu().detach().numpy()), curr_best.cpu().detach().numpy().item()


def weight_reset(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()

class BoDenseTrainer():
    def __init__(self, cfgs, model):
        self.cfgs = cfgs
        self.model_fn = model
        self.device = cfgs.get('device', 'cpu')
        # self.no_conf = cfgs.get('no_conf', False)

        self.lamda = cfgs.get('lambda', 0.5)
        self.bo_obj = cfgs.get('bo_obj', 'Combined')
        self.bo_steps = cfgs.get('bo_steps', 64)
        self.checkpoint_dir = cfgs.get('checkpoint_dir', 'results_bo_dense') +  '/bo_dense'
        self.logger = SummaryWriter(os.path.join(self.checkpoint_dir, 'logs', datetime.now().strftime("%Y%m%d-%H%M%S")))
        print(os.path.join(self.checkpoint_dir, 'logs', datetime.now().strftime("%Y%m%d-%H%M%S")))

    def train(self):


        ############################# BO Init ##############################
        # num_factors = 4 if self.no_conf else 5
        num_factors = 4
        order = np.random.permutation(num_factors)
        inv_order = get_inverse_order(order).view(1, -1)
        best_order = order
        position_kernel = PositionKernel(lamda=self.lamda)
        best_score = -1e30
        print('init order:', order)
        trainer = None

        order_candidates = torch.cat([get_inverse_order(np.array(x)).view(1, -1) for x in itertools.permutations(range(num_factors))]).to(self.device)
        mask = torch.ones(order_candidates.size(0)).bool()
        print(order_candidates)
        print((order_candidates.cpu().view(-1, 4) - torch.tensor(inv_order).view(1, 4)).abs().sum(dim=1).bool())
        mask = torch.logical_and(mask, (order_candidates.cpu().view(-1, 4) - torch.tensor(inv_order).view(1, 4)).abs().sum(dim=1).bool())
        assert torch.sum(mask) == order_candidates.size(0) - 1
        print('size of candidates', mask.sum())





        ############################# BO Steps ##############################

        train_X = None
        train_Y = None
        print("training starts...\n\n")

        t1 = trange(self.bo_steps, desc="best score: N/A, prev score: N/A")
        bo_count = 0
        for ind in t1:
            subexp_name = ''.join([str(x) for x in order.tolist()])
            self.logger.add_text('order', subexp_name, ind)

            if trainer is not None:
                del trainer
                torch.cuda.empty_cache()

            trainer = Trainer(self.cfgs, self.model_fn, order)
            trainer.use_logger = False ### HACK: Manually set to False


            metric = trainer.train_val()
            loss = metric.get_value('Combined')
            score = - loss
            if score > best_score:
                best_score = score
                best_order = order
            
            print('current order:', order)

            if train_X is None:
                train_X = get_inverse_order(order).clone().detach().view(1, -1).to(self.device)
                # train_Y = torch.tensor([test_psnr]).to(device).view(1, 1)
                train_Y = torch.tensor([score]).to(self.device).view(1, 1)
            else:
                train_X = torch.cat([train_X.to(self.device), get_inverse_order(order).view(1, -1).to(self.device)], dim=0)
                # train_Y = torch.cat([train_Y, torch.tensor([test_psnr]).to(device).unsqueeze(-1)])
                train_Y = torch.cat([train_Y.to(self.device), torch.tensor([score]).to(self.device).unsqueeze(-1)], dim=0)
            
            gp_model = botorch.models.SingleTaskGP(train_X, train_Y, covar_module=position_kernel)

            # noise_std = 1e-3
            # gp_model = botorch.models.FixedNoiseGP(train_X, train_Y, (train_Y * 0.0 + noise_std).clone().pow(2), covar_module=position_kernel)
            mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
            mll.train()
            if bo_count > 0:
                botorch.optim.fit.fit_gpytorch_torch(mll, optimizer_cls=torch.optim.Adam, options={'lr': 1e-1})
            mll.eval()
            print('lambda', position_kernel.lamda.item())

            # acq_function = botorch.acquisition.analytic.UpperConfidenceBound(gp_model, beta=1.0)
            # acq_function = botorch.acquisition.analytic.NoisyExpectedImprovement(gp_model, train_X)
            acq_function = botorch.acquisition.analytic.ExpectedImprovement(gp_model, best_score)


            # random_order = np.random.permutation(num_factors)
            # random_inv_order = get_inverse_order(random_order).to(self.device).unsqueeze(0)
            # new_order, ei_score = greedy_optimize(random_inv_order, acq_function)
            # order = new_order.cpu().detach().numpy().astype(np.int32)

            # print(train_X.size(), train_Y.size(), order_candidates.unsqueeze(-2).size())
            ei_scores = acq_function(order_candidates[mask].unsqueeze(-2))
            best_index = torch.argmax(ei_scores)
            ei_score = ei_scores[best_index]
            inv_order = order_candidates[mask][best_index]
            order = get_inverse_order(order_candidates[best_index].cpu().detach().numpy().astype(np.int)).cpu().detach().numpy().astype(np.int32)

            mask = torch.logical_and(mask, (order_candidates.cpu() - torch.tensor(inv_order).cpu().view(1, 4)).abs().sum(dim=1).bool())
            assert torch.sum(mask) == order_candidates.size(0) - 2 - bo_count
            print('size of candidates', mask.sum())





            # t.postfix[1] = best_score.item()
            # t.update()
            t1.set_description('best score: {:f}, prev score: {:f}'.format(best_score, score))
            t1.refresh()
            print('last sample', get_inverse_order(train_X[-1].cpu()))
            print('best order', best_order)
            print('next order', order)
            print('size of candidates', order_candidates.size())

            self.logger.add_scalar('score', score, ind)
            self.logger.add_scalar('best_score', best_score, ind)
            self.logger.add_scalar('ei_score', ei_score, ind)

            bo_count += 1