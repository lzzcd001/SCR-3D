import os
import glob
from datetime import datetime
import numpy as np
import torch
from . import meters
from . import utils
from .dataloaders import get_data_loaders, get_image_loader


class Trainer():
    def __init__(self, cfgs, model, dense_order=None, seed=None, adjacency=None, image_folder=None):
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
        self.train_max_iter = cfgs.get('train_max_iter', -1)
        self.train_break_flag = False
        self.cfgs = cfgs

        self.model = model(cfgs, dense_order=dense_order, adjacency=adjacency)
        self.model.trainer = self

        self.image_folder = image_folder

        self.train_loader, self.val_loader, self.test_loader = get_data_loaders(cfgs)


        if seed is not None:    
            self.checkpoint_dir += f'seed{seed}_'
            if self.test_result_dir is not None:
                self.test_result_dir += f'seed{seed}_'

        if not self.model.use_concat:    
            self.checkpoint_dir += '_additive'
            if self.test_result_dir is not None:
                self.test_result_dir += '_additive'

        if dense_order is not None:
            self.checkpoint_dir += '_denseorder' + ''.join([str(x) for x in dense_order])
            if self.test_result_dir is not None:
                self.test_result_dir += '_denseorder' + ''.join([str(x) for x in dense_order])
        
        elif adjacency is not None:
            self.checkpoint_dir += '_adj' + ''.join([str(x) for x in adjacency])
            if self.test_result_dir is not None:
                self.test_result_dir += '_adj' + ''.join([str(x) for x in adjacency])
            
            adjacency = torch.tensor(adjacency).view(4,4)
        else:
            self.checkpoint_dir += '_denseorder_baseline'
            if self.test_result_dir is not None:
                self.test_result_dir += '_denseorder_baseline'



        self.metrics_trace = meters.MetricsTrace()
        self.make_metrics = lambda m=None: meters.StandardMetrics(m)
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
            m = self.run_epoch(self.test_loader, epoch=self.current_epoch, is_test=True, no_viz=True)

        if not os.path.exists(self.test_result_dir):
            os.makedirs(self.test_result_dir)
        score_path = os.path.join(self.test_result_dir, 'eval_scores.txt')
        self.model.save_scores(score_path)


    def viz(self):
        """Perform testing."""
        self.model.to_device(self.device)
        self.current_epoch = self.load_checkpoint(optim=False)
        if self.test_result_dir is None:
            self.test_result_dir = os.path.join(self.checkpoint_dir, f'test_results_{self.checkpoint_name}'.replace('.pth',''))
        print(f"Saving testing results to {self.test_result_dir}")


        if self.image_folder is not None:
            with torch.no_grad():
                loader = get_image_loader(self.image_folder, True, batch_size=5, image_size=64)
                for batch in loader:
                    batch = batch.cuda()
                    print(batch.size())
                    # batch = next(loader).cuda()
                    self.run_custom_batch(batch)
                    break
        else:
            with torch.no_grad():
                # print(self.test_loader)
                m = self.run_iter(self.test_loader, epoch=self.current_epoch, is_test=True)

        # from tensorboardX import SummaryWriter
        # logger = SummaryWriter(os.path.join(self.checkpoint_dir, 'logs_viz', datetime.now().strftime("%Y%m%d-%H%M%S")))
        # self.model.gen_viz(logger)



    def viz_interp(self):
        """Perform testing."""
        self.model.to_device(self.device)
        self.current_epoch = self.load_checkpoint(optim=False)
        if self.test_result_dir is None:
            self.test_result_dir = os.path.join(self.checkpoint_dir, f'test_results_{self.checkpoint_name}'.replace('.pth',''))
        print(f"Saving testing results to {self.test_result_dir}")


        with torch.no_grad():
            # print(self.test_loader)
            m = self.run_iter_interp(self.test_loader, epoch=self.current_epoch, is_test=True)

        # from tensorboardX import SummaryWriter
        # logger = SummaryWriter(os.path.join(self.checkpoint_dir, 'logs_viz', datetime.now().strftime("%Y%m%d-%H%M%S")))
        # self.model.gen_viz(logger)

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


    def train_val(self):
        ################# Train #################

        ## initialize
        start_epoch = 0
        self.metrics_trace.reset()
        self.train_iter_per_epoch = len(self.train_loader)
        self.model.to_device(self.device)
        self.model.init_optimizers()

        ## run epochs
        for epoch in range(start_epoch, self.num_epochs):
            self.current_epoch = epoch
            metrics = self.run_epoch(self.train_loader, epoch)
            if self.train_break_flag:
                break

        ##################################

        ### Val
        with torch.no_grad():
            metrics = self.run_epoch(self.val_loader, epoch, is_validation=True)

        return metrics


    def run_epoch(self, loader, epoch=0, is_validation=False, is_test=False, no_viz=False):
        """Run one epoch."""
        is_train = not is_validation and not is_test
        metrics = self.make_metrics()

        if is_train:
            print(f"Starting training epoch {epoch}")
            self.model.set_train()
        else:
            print(f"Starting validation epoch {epoch}")
            self.model.set_eval()

        for iter, input in enumerate(loader):
            m = self.model.forward(input)
            if is_train:
                self.model.backward() 
            elif is_test:
                self.model.save_results(self.test_result_dir, no_viz=no_viz)

            metrics.update(m, self.batch_size)
            print(f"{'T' if is_train else 'V'}{epoch:02}/{iter:05}/{metrics}")

            if self.use_logger and is_train:
                total_iter = iter + epoch*self.train_iter_per_epoch
                if total_iter % self.log_freq == 0:
                    self.model.forward(self.viz_input)
                    self.model.visualize(self.logger, total_iter=total_iter, max_bs=25)

            if is_train:
                total_iter = iter + epoch*self.train_iter_per_epoch
                if self.train_max_iter >= 0 and total_iter >= self.train_max_iter:
                    self.train_break_flag = True
                    break
        return metrics


    # def run_iter(self, loader, epoch=0, is_validation=False, is_test=False, stop_iter=9):
    def run_iter(self, loader, epoch=0, is_validation=False, is_test=False, stop_iter=int(1e6)):
        """Run one iter."""
        is_train = not is_validation and not is_test
        metrics = self.make_metrics()

        if is_train:
            print(f"Starting training epoch {epoch}")
            self.model.set_train()
        else:
            print(f"Starting validation epoch {epoch}")
            self.model.set_eval()

        for iter, input in enumerate(loader):
            m = self.model.forward(input)
            if is_train:
                self.model.backward()
            elif is_test:
                self.model.save_results(self.test_result_dir)
                print('save results')

            metrics.update(m, self.batch_size)
            print(f"{'T' if is_train else 'V'}{epoch:02}/{iter:05}/{metrics}")

            if self.use_logger and is_train:
                total_iter = iter + epoch*self.train_iter_per_epoch
                if total_iter % self.log_freq == 0:
                    self.model.forward(self.viz_input)
                    # self.model.visualize(self.logger, total_iter=total_iter, max_bs=25)
            if iter == stop_iter:
                break
            # break
        return metrics


    def run_iter_interp(self, loader, epoch=0, is_validation=False, is_test=False, stop_iter=10):
        """Run one iter."""
        is_train = not is_validation and not is_test
        metrics = self.make_metrics()

        self.model.set_eval()

        for iter, input in enumerate(loader):
            m = self.model.forward_interp(input)
            self.model.save_results(self.test_result_dir + '_interp')

            if iter == stop_iter:
                break
            # break
        return metrics



    def run_custom_batch(self, batch):
        """Run one iter."""
        self.model.set_eval()
        m = self.model.forward(batch)
        custom_dir = self.test_result_dir + '_custom_input'
        self.model.save_results(custom_dir)
