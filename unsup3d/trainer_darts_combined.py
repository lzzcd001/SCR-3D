import os
import glob
from datetime import datetime
import numpy as np
import torch
from . import meters
from . import utils
from .dataloaders import get_data_loaders
import itertools
import math


class DartsCombinedTrainer():
    def __init__(self, cfgs, model, dense_order=None, seed=0):
        self.device = cfgs.get('device', 'cpu')
        self.num_epochs = cfgs.get('num_epochs', 30)
        self.batch_size = cfgs.get('batch_size', 64)
        self.checkpoint_dir = cfgs.get('checkpoint_dir', 'results')
        self.save_checkpoint_freq = cfgs.get('save_checkpoint_freq', 1)
        self.keep_num_checkpoint = cfgs.get('keep_num_checkpoint', 2)  # -1 for keeping all checkpoints
        self.resume = cfgs.get('resume', True)
        self.use_logger = cfgs.get('use_logger', True)
        self.log_freq = cfgs.get('log_freq', 1000)
        self.archive_code = cfgs.get('archive_code', True)
        self.checkpoint_name = cfgs.get('checkpoint_name', None)
        self.test_result_dir = cfgs.get('test_result_dir', None)
        self.cfgs = cfgs
        self.seed = seed

        self.model = model(cfgs, dense_order=dense_order)
        self.model.trainer = self

        if self.model.non_darts:
            if self.model.dynamic_emb:
                self.checkpoint_dir += '_dynemb'
                if self.test_result_dir is not None:
                    self.test_result_dir += '_dynemb'
            
            else:
                self.checkpoint_dir += '_e2e'
                if self.test_result_dir is not None:
                    self.test_result_dir += '_e2e'
            
        else:
            self.checkpoint_dir = f'darts_combined_{self.model.darts_obj}_inner{self.model.inner_steps}_' + self.checkpoint_dir
            if self.test_result_dir is not None:
                self.test_result_dir = f'darts_combined_{self.model.darts_obj}_inner{self.model.inner_steps}_' + self.test_result_dir

            if self.model.lam_adj != self.model.default_lam_adj:
                self.checkpoint_dir += f'_lam{math.log10(self.model.lam_adj)}'
                if self.test_result_dir is not None:
                    self.test_result_dir += f'_lam{math.log10(self.model.lam_adj)}'
        

        if not self.model.use_concat:    
            self.checkpoint_dir += '_additive'
            if self.test_result_dir is not None:
                self.test_result_dir += '_additive'


        if self.model.no_dag_loss:    
            self.checkpoint_dir += '_no_dag_loss'
            if self.test_result_dir is not None:
                self.test_result_dir += '_no_dag_loss'


        if self.model.adj_uv:    
            self.checkpoint_dir += '_adj_uv'
            if self.test_result_dir is not None:
                self.test_result_dir += '_adj_uv'

 
        self.checkpoint_dir += f'_seed{self.seed}'
        if self.test_result_dir is not None:
            self.test_result_dir += f'_seed{self.seed}'

        if dense_order is not None:
            self.checkpoint_dir += '_denseorder' + ''.join([str(x) for x in dense_order])
            if self.test_result_dir is not None:
                self.test_result_dir += '_denseorder' + ''.join([str(x) for x in dense_order])

        self.metrics_trace = meters.MetricsTrace()
        self.make_metrics = lambda m=None: meters.StandardMetrics(m)
        self.train_loader, self.val_loader, self.test_loader = get_data_loaders(cfgs)
        self.darts_train_loader, self.darts_val_loader, _ = get_data_loaders(cfgs)

        self.darts_train_loader = itertools.cycle(self.darts_train_loader)
        self.darts_val_loader = itertools.cycle(self.darts_val_loader)
        self.global_iter = 0

    def load_checkpoint(self, optim=True):
        """Search the specified/latest checkpoint in checkpoint_dir and load the model and optimizer."""
        if self.checkpoint_name is not None:
            checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_name)
        else:
            checkpoints = sorted(glob.glob(os.path.join(self.checkpoint_dir, '*.pth')))
            if len(checkpoints) == 0:
                return 0
            checkpoint_path = checkpoints[-1]
            self.checkpoint_name = os.path.basename(checkpoint_path)
        print(f"Loading checkpoint from {checkpoint_path}")
        cp = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_model_state(cp)
        if optim:
            self.model.load_optimizer_state(cp)
        self.metrics_trace = cp['metrics_trace']
        epoch = cp['epoch']
        return epoch

    def save_checkpoint(self, epoch, optim=True):
        """Save model, optimizer, and metrics state to a checkpoint in checkpoint_dir for the specified epoch."""
        utils.xmkdir(self.checkpoint_dir)
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint{epoch:03}.pth')
        state_dict = self.model.get_model_state()
        if optim:
            optimizer_state = self.model.get_optimizer_state()
            state_dict = {**state_dict, **optimizer_state}
        state_dict['metrics_trace'] = self.metrics_trace
        state_dict['epoch'] = epoch
        print(f"Saving checkpoint to {checkpoint_path}")
        torch.save(state_dict, checkpoint_path)
        if self.keep_num_checkpoint > 0:
            utils.clean_checkpoint(self.checkpoint_dir, keep_num=self.keep_num_checkpoint)

    def save_clean_checkpoint(self, path):
        """Save model state only to specified path."""
        torch.save(self.model.get_model_state(), path)

    def test(self):
        """Perform testing."""
        self.model.to_device(self.device)
        self.current_epoch = self.load_checkpoint(optim=False)
        if self.test_result_dir is None:
            self.test_result_dir = os.path.join(self.checkpoint_dir, f'test_results_{self.checkpoint_name}'.replace('.pth',''))
        print(f"Saving testing results to {self.test_result_dir}")

        with torch.no_grad():
            m = self.run_epoch(self.test_loader, epoch=self.current_epoch, is_test=True)

        score_path = os.path.join(self.test_result_dir, 'eval_scores.txt')
        self.model.save_scores(score_path)

    def train(self):
        """Perform training."""
        ## archive code and configs
        if self.archive_code:
            utils.archive_code(os.path.join(self.checkpoint_dir, 'archived_code.zip'), filetypes=['.py', '.yml'])
        utils.dump_yaml(os.path.join(self.checkpoint_dir, 'configs.yml'), self.cfgs)

        ## initialize
        start_epoch = 0
        self.metrics_trace.reset()
        self.train_iter_per_epoch = len(self.train_loader)
        self.model.to_device(self.device)
        self.model.init_optimizers()

        ## resume from checkpoint
        if self.resume:
            start_epoch = self.load_checkpoint(optim=True)

        ## initialize tensorboardX logger
        if self.use_logger:
            from tensorboardX import SummaryWriter
            self.logger = SummaryWriter(os.path.join(self.checkpoint_dir, 'logs', datetime.now().strftime("%Y%m%d-%H%M%S")))

            ## cache one batch for visualization
            self.viz_input = self.val_loader.__iter__().__next__()

        ## run epochs
        print(f"{self.model.model_name}: optimizing to {self.num_epochs} epochs")
        for epoch in range(start_epoch, self.num_epochs):
            self.current_epoch = epoch
            metrics = self.run_epoch(self.train_loader, epoch)
            self.metrics_trace.append("train", metrics)

            with torch.no_grad():
                metrics = self.run_epoch(self.val_loader, epoch, is_validation=True)
                self.metrics_trace.append("val", metrics)

            if (epoch+1) % self.save_checkpoint_freq == 0:
                self.save_checkpoint(epoch+1, optim=True)
            self.metrics_trace.plot(pdf_path=os.path.join(self.checkpoint_dir, 'metrics.pdf'))
            self.metrics_trace.save(os.path.join(self.checkpoint_dir, 'metrics.json'))

        print(f"Training completed after {epoch+1} epochs.")

    def run_epoch(self, loader, epoch=0, is_validation=False, is_test=False):
        """Run one epoch."""
        is_train = not is_validation and not is_test
        metrics = self.make_metrics()

        if not self.model.dynamic_emb:
            print("\n\nAdjacency")
            adjacency = self.model.get_adj()
            print(adjacency)
            print("\n\n")
            with open(os.path.join(self.checkpoint_dir, 'adj.txt'), 'a') as f:
                f.write(f'epoch: {epoch}\n')
                print(adjacency, file=f)
                f.write("\n\n\n")

        if is_train:
            print(f"Starting training epoch {epoch}")
            self.model.set_train()
        else:
            print(f"Starting validation epoch {epoch}")
            self.model.set_eval()

        for iter, input in enumerate(loader):
            if self.model.dynamic_emb:
                adjacency = None
            else:
                adjacency = self.model.get_adj()

            if self.model.adj_uv:
                m = self.model.forward(input, adjacency)
            else:
                m = self.model.forward(input, adjacency)
            if is_train:
                self.model.backward() 
                self.model.backward_meta(darts_train_loader=self.darts_train_loader, darts_val_loader=self.darts_val_loader) ### Make sure that darts_loader is wrapped by itertools.cycle
            elif is_test:
                self.model.save_results(self.test_result_dir)

            metrics.update(m, self.batch_size)
            print(f"{'T' if is_train else 'V'}{epoch:02}/{iter:05}/{metrics}")

            if (not self.model.dynamic_emb and self.global_iter >= self.model.lam_warmup
                and self.global_iter % self.model.lam_frequency == 0 and self.model.lam_adj < 1e20
            ):
                if self.model.non_darts:
                    self.model.lam_adj *= 10.0
                    self.model.adj_optimizer.param_groups[0]['lr'] = self.model.lr / (math.log10(self.model.lam_adj) + 1e-10)
                else:
                    self.model.lam_adj *= 2.0
                    self.model.adj_optimizer.param_groups[0]['lr'] = self.model.lr / (math.log10(self.model.lam_adj) + 1e-10)

            

            if self.use_logger and is_train:
                total_iter = iter + epoch*self.train_iter_per_epoch
                if total_iter % self.log_freq == 0:
                    if self.model.dynamic_emb:
                        adjacency = None
                    elif self.model.adj_uv:
                        adjacency = torch.sigmoid(self.model.U @ self.model.V.T) * (1.0 - torch.eye(4, 4, device=self.model.U.device))
                    else:
                        adjacency = self.model.get_adj()
                    self.model.forward(self.viz_input, adjacency)
                    self.model.visualize(self.logger, total_iter=total_iter, max_bs=25)

            if iter % 500 == 0 and iter != 0:
                if not self.model.dynamic_emb:
                    print("\n\nAdjacency")
                    if self.model.adj_uv:
                        adjacency = torch.sigmoid(self.model.U @ self.model.V.T) * (1.0 - torch.eye(4, 4, device=self.model.U.device))
                    else:
                        adjacency = self.model.get_adj()
                    print(adjacency)
                    print("\n\n")
                    with open(os.path.join(self.checkpoint_dir, 'adj.txt'), 'a') as f:
                        f.write(f'epoch: {epoch}, iter: {iter}\n')
                        print(adjacency, file=f)
                        f.write("\n\n\n")


            self.global_iter += 1
        return metrics
