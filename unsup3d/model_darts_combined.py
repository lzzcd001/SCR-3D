import os
import math
import glob
import torch
import torch.nn as nn
import torchvision
from . import networks
from . import utils
from .renderer import Renderer
import higher
import numpy as np


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")


EPS = 1e-7


class DartsCombinedUnsup3D(nn.Module):
    def __init__(self, cfgs, dense_order=None):
        super(DartsCombinedUnsup3D, self).__init__()
        self.model_name = cfgs.get('model_name', self.__class__.__name__)
        self.device = cfgs.get('device', 'cpu')
        self.image_size = cfgs.get('image_size', 64)
        self.min_depth = cfgs.get('min_depth', 0.9)
        self.max_depth = cfgs.get('max_depth', 1.1)
        self.border_depth = cfgs.get('border_depth', (0.7*self.max_depth + 0.3*self.min_depth))
        self.min_amb_light = cfgs.get('min_amb_light', 0.)
        self.max_amb_light = cfgs.get('max_amb_light', 1.)
        self.min_diff_light = cfgs.get('min_diff_light', 0.)
        self.max_diff_light = cfgs.get('max_diff_light', 1.)
        self.xyz_rotation_range = cfgs.get('xyz_rotation_range', 60)
        self.xy_translation_range = cfgs.get('xy_translation_range', 0.1)
        self.z_translation_range = cfgs.get('z_translation_range', 0.1)
        self.use_conf_map = cfgs.get('use_conf_map', True)
        self.lam_perc = cfgs.get('lam_perc', 1)
        self.lam_flip = cfgs.get('lam_flip', 0.5)
        self.lam_flip_start_epoch = cfgs.get('lam_flip_start_epoch', 0)
        self.lam_depth_sm = cfgs.get('lam_depth_sm', 0)
        self.lr = cfgs.get('lr', 1e-4)
        self.depth_coeff = cfgs.get('depth_coeff', 0.1)
        self.l1_coeff = cfgs.get('l1_coeff', 1.0)
        self.perc_coeff = cfgs.get('perc_coeff', 1.0)
        self.load_gt_depth = cfgs.get('load_gt_depth', False)
        self.use_concat = cfgs.get('use_concat', True)
        self.non_darts = cfgs.get('non_darts', False)
        self.dynamic_emb = cfgs.get('dynamic_emb', False)
        self.adj_uv = cfgs.get('adj_uv', False)
        self.no_dag_loss = cfgs.get('no_dag_loss', False)
        self.darts_obj = cfgs.get('darts_obj', 'NorErr_masked')
        self.renderer = Renderer(cfgs)

        assert not (self.dynamic_emb and not self.non_darts)

        ### darts related
        self.renderer_inner = Renderer(cfgs)
        self.inner_steps = cfgs.get('inner_steps', 5)
        self.default_lam_adj = 10.0
        self.lam_adj = cfgs.get('lam_adj', self.default_lam_adj)
        self.lam_warmup = cfgs.get('lam_warmup', 2500)
        self.lam_frequency = cfgs.get('lam_frequency', 2500)

        ## networks and optimizers
        self.netD_enc = networks.BigEncoder(cin=3, cout=1, nf=64, zdim=256)
        self.netA_enc = networks.BigEncoder(cin=3, cout=3, nf=64, zdim=256)
        self.netL_enc = networks.SmallEncoder(cin=3, cout=4, nf=32, zdim=256)
        self.netV_enc = networks.SmallEncoder(cin=3, cout=6, nf=32, zdim=256)

        self.netD_dec = networks.BigDecoder(cin=3, cout=1, nf=64, zdim=256, activation=None)
        self.netA_dec = networks.BigDecoder(cin=3, cout=3, nf=64, zdim=256)
        self.netL_dec = networks.SmallDecoder(cin=3, cout=4, nf=32, zdim=256)
        self.netV_dec = networks.SmallDecoder(cin=3, cout=6, nf=32, zdim=256)
        if self.use_conf_map:
            self.netC = networks.ConfNet(cin=3, cout=2, nf=64, zdim=128)
        
        if not self.dynamic_emb:
            if self.adj_uv:
                self.U = nn.Parameter((torch.randn(4, 20)) / np.sqrt(20), requires_grad=True)
                self.V = nn.Parameter((torch.randn(4, 20)) / np.sqrt(20), requires_grad=True)
            else:
                self.adjacency = nn.Parameter(1e-1 * torch.randn(4,4) * (torch.ones(4, 4) - torch.eye(4)), requires_grad=True)
        
        if self.use_concat:
            self.factor_mlp_list = []
            for factor in ['depth', 'albedo', 'light', 'view']:
                factor_mlp = networks.FactorMLP(256*4, 256)
                setattr(self, 'factor_mlpnet_' + factor, factor_mlp)
                self.factor_mlp_list.append(factor_mlp)


        if dense_order is not None:
            self.independent = False
            self.dense_order = dense_order
            adjacency = torch.zeros(4, 4).float()
            for i, indx in enumerate(self.dense_order):
                if i == 0:
                    continue
                for j in range(i):
                    adjacency[indx, self.dense_order[j]] = 1.0
            self.adjacency_mask = nn.Parameter(adjacency, requires_grad=False)
        else:
            self.adjacency_mask = None
            

        # self.network_names = [k for k in vars(self) if 'net' in k]
        self.network_names = [k for k in self._modules if 'net' in k]
        self.make_optimizer = lambda model: torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)

        ## other parameters
        self.PerceptualLoss = networks.PerceptualLoss(requires_grad=False)
        self.other_param_names = ['PerceptualLoss']

        ## depth rescaler: -1~1 -> min_deph~max_deph
        self.depth_rescaler = lambda d : (1+d)/2 *self.max_depth + (1-d)/2 *self.min_depth
        self.amb_light_rescaler = lambda x : (1+x)/2 *self.max_amb_light + (1-x)/2 *self.min_amb_light
        self.diff_light_rescaler = lambda x : (1+x)/2 *self.max_diff_light + (1-x)/2 *self.min_diff_light

    def init_optimizers(self):

        param_list = []
        for net_name in self.network_names:
            param_list.extend(list(getattr(self, net_name).parameters()))
        self.global_optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, param_list),
                lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
        if not self.dynamic_emb:
            if self.adj_uv:
                self.adj_optimizer = torch.optim.Adam([self.U, self.V], lr=self.lr)
            else:
                self.adj_optimizer = torch.optim.Adam([self.adjacency], lr=self.lr)
            self.optimizer_names = ['global_optimizer', 'adj_optimizer']
        else:
            self.optimizer_names = ['global_optimizer']


    def load_model_state(self, cp):
        for k in cp:
            if k and k in self.network_names:
                getattr(self, k).load_state_dict(cp[k])

    def load_optimizer_state(self, cp):
        for k in cp:
            if k and k in self.optimizer_names:
                getattr(self, k).load_state_dict(cp[k])

    def get_model_state(self):
        states = {}
        for net_name in self.network_names:
            states[net_name] = getattr(self, net_name).state_dict()
        return states

    def get_optimizer_state(self):
        states = {}
        for optim_name in self.optimizer_names:
            states[optim_name] = getattr(self, optim_name).state_dict()
        return states

    def to_device(self, device):
        self.device = device
        for net_name in self.network_names:
            setattr(self, net_name, getattr(self, net_name).to(device))
        if self.other_param_names:
            for param_name in self.other_param_names:
                setattr(self, param_name, getattr(self, param_name).to(device))

    def set_train(self):
        for net_name in self.network_names:
            getattr(self, net_name).train()

    def set_eval(self):
        for net_name in self.network_names:
            getattr(self, net_name).eval()

    def photometric_loss(self, im1, im2, mask=None, conf_sigma=None, reduce=True):
        loss = (im1-im2).abs()
        if conf_sigma is not None:
            loss = loss *2**0.5 / (conf_sigma +EPS) + (conf_sigma +EPS).log()
        if reduce:
            if mask is not None:
                mask = mask.expand_as(loss)
                loss = (loss * mask).sum() / mask.sum()
            else:
                loss = loss.mean()
        else:
            if mask is not None:
                mask = mask.expand_as(loss)
                loss = loss * mask
        return loss

    def backward(self):
        self.global_optimizer.zero_grad()
        if not self.dynamic_emb:
            self.adj_optimizer.zero_grad()
        if self.non_darts and not self.dynamic_emb and not self.no_dag_loss:
            loss_adj = self.DAGLoss(self.get_adj().abs())
            (self.loss_total + self.lam_adj * loss_adj).backward()
        else:
            self.loss_total.backward()
        self.global_optimizer.step()
        if self.non_darts and not self.dynamic_emb:
            self.adj_optimizer.step()
    
    def get_adj(self):
        if self.adj_uv:
            adjacency = torch.sigmoid(self.U @ self.V.T) * (1.0 - torch.eye(4, 4, device=self.U.device))

            # U = self.U / torch.sqrt(self.U.pow(2).sum(dim=1, keepdim=True) + 1e-8)
            # V = self.V / torch.sqrt(self.V.pow(2).sum(dim=1, keepdim=True) + 1e-8)
            # adjacency = torch.sigmoid(U @ V.T) * (1.0 - torch.eye(4, 4, device=self.U.device))


            # adjacency = torch.abs(self.U @ self.V.T) * (1.0 - torch.eye(4, 4, device=self.U.device))


            # adjacency = torch.tanh(self.U @ self.V.T).abs() * (1.0 - torch.eye(4, 4, device=self.U.device))
        else:
            # adjacency = torch.sigmoid(self.adjacency) * (1.0 - torch.eye(4, 4, device=self.adjacency.device))
            adjacency = torch.abs(self.adjacency) * (1.0 - torch.eye(4, 4, device=self.adjacency.device))
        
        if self.adjacency_mask is not None:
            adjacency = self.adjacency_mask * adjacency
        return adjacency

    def backward_meta(self, darts_train_loader, darts_val_loader):
        if self.non_darts:
            return
        self.adj_optimizer.zero_grad()
        self.global_optimizer.zero_grad()

        with higher.innerloop_ctx(self, getattr(self, 'global_optimizer')) as (fmodel, diffopt):
            adjacency = self.get_adj()
            for _ in range(self.inner_steps):
                if self.load_gt_depth:
                    darts_input, _ = next(darts_train_loader)
                else:
                    darts_input = next(darts_train_loader)
                loss = fmodel.forward_inner(darts_input, adjacency)['loss_total']
                diffopt.step(loss)
            
            if self.load_gt_depth:
                darts_input, gt_depth = next(darts_val_loader)
            else:
                darts_input = next(darts_val_loader)
                gt_depth = None
            meta_loss_dict = fmodel.forward_inner(darts_input, adjacency, depth_gt=gt_depth)
            meta_loss = (
                self.l1_coeff * meta_loss_dict['Image_masked'] 
                + self.perc_coeff * meta_loss_dict['Perc_masked'] 
                + self.depth_coeff * meta_loss_dict[self.darts_obj]
            )

            # loss_adj = self.DAGLoss(adjacency.abs())
            # loss_adj_coeff = 1.0
            # loss_adj_coeff = self.lam_adj
            # meta_loss = self.depth_coeff * meta_loss + loss_adj_coeff * loss_adj
            # meta_loss.backward()
    
            meta_loss.backward(retain_graph=True)
        
        # print(self.adjacency.grad)
        if not self.no_dag_loss:
            loss_adj = self.DAGLoss(adjacency.abs())
            (self.lam_adj * loss_adj).backward()
            print(self.lam_adj * loss_adj)
        self.adj_optimizer.step()

    def DAGLoss(self, adj):
        m = 4
        loss = (torch.trace(torch.matrix_power(torch.eye(m).to(adj.device) + adj.pow(2) / m, m)) - m).pow(2)
        return loss

    def forward_inner(self, input, adjacency, depth_gt=None):
        input_im = input.to(self.device) *2.-1.
        b, c, h, w = input_im.shape


        ## Run networks
        # canon_depth_pre = self.netD_dec(self.netD_enc(self.input_im))
        # cacon_albedo_pre = self.netA_dec(self.netA_enc(self.input_im))
        # canon_light_pre = self.netL_dec(self.netL_enc(self.input_im))
        # view_pre = self.netV_dec(self.netV_enc(self.input_im))

        depth_emb = self.netD_enc(input_im)
        albedo_emb = self.netA_enc(input_im)
        light_emb = self.netL_enc(input_im)
        view_emb = self.netV_enc(input_im)

        emb_dict = {
            'depth': depth_emb,
            'albedo': albedo_emb,
            'light': light_emb,
            'view': view_emb
        }

        concat_emb = torch.cat([
            depth_emb.unsqueeze_(2), 
            albedo_emb.unsqueeze_(2),
            light_emb.unsqueeze_(2),
            view_emb.unsqueeze_(2),
        ], dim=2)
        bs, ch, _, hz, wz = concat_emb.size()
        if self.dynamic_emb:
            normed_concat_emb = concat_emb.permute(0, 2, 1, 3, 4) / np.sqrt(ch * hz * wz)
            adjacency = torch.sigmoid(torch.einsum('nxacd, nyacd->nxy', normed_concat_emb, normed_concat_emb))
            adjacency_aug = adjacency.to(concat_emb.device).abs() + torch.eye(4, device=concat_emb.device)
        else:
            adjacency_aug = adjacency.to(concat_emb.device).abs() + torch.eye(4, device=concat_emb.device)

        if self.dynamic_emb:
            if self.use_concat:
                for i, factor in enumerate(['depth', 'albedo', 'light', 'view']):
                    concat_input = concat_emb * adjacency_aug[:, i][:, None, :, None, None].abs()
                    emb_dict[factor] = self.factor_mlp_list(concat_input.view(bs, -1, 1, 1))
            else:
                for i, factor in enumerate(['depth', 'albedo', 'light', 'view']):
                    emb_dict[factor] = (concat_emb * adjacency_aug[:, i][:, None, :, None, None].abs()).sum(2)

        else:
            if self.use_concat:
                for i, factor in enumerate(['depth', 'albedo', 'light', 'view']):
                    concat_input = concat_emb * adjacency_aug[i][None, None, :, None, None].abs()
                    emb_dict[factor] = self.factor_mlp_list[i](concat_input.view(bs, -1, 1, 1))
            else:
                for i, factor in enumerate(['depth', 'albedo', 'light', 'view']):
                    emb_dict[factor] = (concat_emb * adjacency_aug[i][None, None, :, None, None].abs()).sum(2)

        
        # for i, factor in enumerate(['depth', 'albedo', 'light', 'view']):
        #     concat_input = concat_emb * adjacency_aug[i][None, None, :, None, None].abs()
        #     emb_dict[factor] = self.factor_mlp_list[i](concat_input.view(bs, -1, 1, 1))
        
        depth_pre = self.netD_dec(emb_dict['depth'])
        albedo_pre = self.netA_dec(emb_dict['albedo'])
        light_pre = self.netL_dec(emb_dict['light'])
        view_pre = self.netV_dec(emb_dict['view'])


        ## predict canonical depth
        canon_depth_raw = depth_pre.squeeze(1)  # BxHxW
        canon_depth = canon_depth_raw - canon_depth_raw.view(b,-1).mean(1).view(b,1,1)
        canon_depth = canon_depth.tanh()
        canon_depth = self.depth_rescaler(canon_depth)

        ## optional depth smoothness loss (only used in synthetic car experiments)
        loss_depth_sm = ((canon_depth[:,:-1,:] - canon_depth[:,1:,:]) /(self.max_depth-self.min_depth)).abs().mean()
        loss_depth_sm += ((canon_depth[:,:,:-1] - canon_depth[:,:,1:]) /(self.max_depth-self.min_depth)).abs().mean()

        ## clamp border depth
        depth_border = torch.zeros(1,h,w-4).to(input_im.device)
        depth_border = nn.functional.pad(depth_border, (2,2), mode='constant', value=1)
        canon_depth = canon_depth*(1-depth_border) + depth_border *self.border_depth
        canon_depth = torch.cat([canon_depth, canon_depth.flip(2)], 0)  # flip

        ## predict canonical albedo
        canon_albedo = albedo_pre  # Bx3xHxW
        canon_albedo = torch.cat([canon_albedo, canon_albedo.flip(3)], 0)  # flip

        ## predict confidence map
        if self.use_conf_map:
            conf_sigma_l1_raw, conf_sigma_percl_raw = self.netC(input_im)  # Bx2xHxW
            conf_sigma_l1 = conf_sigma_l1_raw[:,:1]
            conf_sigma_l1_flip = conf_sigma_l1_raw[:,1:]
            conf_sigma_percl = conf_sigma_percl_raw[:,:1]
            conf_sigma_percl_flip = conf_sigma_percl_raw[:,1:]
        else:
            conf_sigma_l1 = None
            conf_sigma_l1_flip = None
            conf_sigma_percl = None
            conf_sigma_percl_flip = None

        ## predict lighting
        canon_light = light_pre.repeat(2,1)  # Bx4
        canon_light_a = self.amb_light_rescaler(canon_light[:,:1])  # ambience term
        canon_light_b = self.diff_light_rescaler(canon_light[:,1:2])  # diffuse term
        canon_light_dxy = canon_light[:,2:]
        canon_light_d = torch.cat([canon_light_dxy, torch.ones(b*2,1).to(input_im.device)], 1)
        canon_light_d = canon_light_d / ((canon_light_d**2).sum(1, keepdim=True))**0.5  # diffuse light direction

        ## shading
        canon_normal = self.renderer_inner.get_normal_from_depth(canon_depth)
        canon_diffuse_shading = (canon_normal * canon_light_d.view(-1,1,1,3)).sum(3).clamp(min=0).unsqueeze(1)
        canon_shading = canon_light_a.view(-1,1,1,1) + canon_light_b.view(-1,1,1,1) * canon_diffuse_shading
        canon_im = (canon_albedo/2+0.5) * canon_shading *2-1

        ## predict viewpoint transformation
        view = view_pre.repeat(2,1)
        view = torch.cat([
            view[:,:3] * math.pi/180 * self.xyz_rotation_range,
            view[:,3:5] * self.xy_translation_range,
            view[:,5:] * self.z_translation_range], 1)

        ## reconstruct input view
        self.renderer_inner.set_transform_matrices(view)
        recon_depth = self.renderer_inner.warp_canon_depth(canon_depth)
        recon_normal = self.renderer_inner.get_normal_from_depth(recon_depth)
        grid_2d_from_canon = self.renderer_inner.get_inv_warped_2d_grid(recon_depth)
        recon_im = nn.functional.grid_sample(canon_im, grid_2d_from_canon, mode='bilinear')

        margin = (self.max_depth - self.min_depth) /2
        recon_im_mask = (recon_depth < self.max_depth+margin).float()  # invalid border pixels have been clamped at max_depth+margin
        recon_im_mask_both = recon_im_mask[:b] * recon_im_mask[b:]  # both original and flip reconstruction
        recon_im_mask_both = recon_im_mask_both.repeat(2,1,1).unsqueeze(1).detach()
        recon_im = recon_im * recon_im_mask_both

        ## loss function
        loss_l1_im = self.photometric_loss(recon_im[:b], input_im, mask=recon_im_mask_both[:b], conf_sigma=conf_sigma_l1)
        loss_l1_im_flip = self.photometric_loss(recon_im[b:], input_im, mask=recon_im_mask_both[b:], conf_sigma=conf_sigma_l1_flip)
        loss_perc_im = self.PerceptualLoss(recon_im[:b], input_im, mask=recon_im_mask_both[:b], conf_sigma=conf_sigma_percl)
        loss_perc_im_flip = self.PerceptualLoss(recon_im[b:], input_im, mask=recon_im_mask_both[b:], conf_sigma=conf_sigma_percl_flip)
        lam_flip = 1 if self.trainer.current_epoch < self.lam_flip_start_epoch else self.lam_flip
        loss_total = loss_l1_im + lam_flip * loss_l1_im_flip + self.lam_perc * (loss_perc_im + lam_flip * loss_perc_im_flip) + self.lam_depth_sm * loss_depth_sm

        loss_dict = {'loss_total': loss_total}
        ## compute accuracy if gt depth is available
        if depth_gt is not None:
            depth_gt = depth_gt[:,0,:,:].to(input_im.device)
            depth_gt = (1-depth_gt)*2-1
            depth_gt = self.depth_rescaler(depth_gt)
            normal_gt = self.renderer_inner.get_normal_from_depth(depth_gt)


            # mask out background
            mask_gt = (depth_gt<depth_gt.max()).float()
            mask_gt = (nn.functional.avg_pool2d(mask_gt.unsqueeze(1), 3, stride=1, padding=1).squeeze(1) > 0.99).float()  # erode by 1 pixel
            mask_pred = (nn.functional.avg_pool2d(recon_im_mask[:b].unsqueeze(1), 3, stride=1, padding=1).squeeze(1) > 0.99).float()  # erode by 1 pixel
            mask = mask_gt * mask_pred

            acc_mae_masked = ((recon_depth[:b] - depth_gt[:b]).abs() *mask).view(b,-1).sum(1) / mask.view(b,-1).sum(1)
            acc_mse_masked = (((recon_depth[:b] - depth_gt[:b])**2) *mask).view(b,-1).sum(1) / mask.view(b,-1).sum(1)
            loss_l1_map = self.photometric_loss(recon_im[:b], input_im[:b], mask=mask[:b].unsqueeze(1), reduce=False)
            loss_l1_masked = loss_l1_map.view(b,-1).sum(1) / mask.view(b,-1).sum(1)
            loss_perc_im_masked = self.PerceptualLoss(recon_im[:b], input_im[:b], mask=mask[:b].unsqueeze(1))

            sie_map_masked = utils.compute_sc_inv_err(recon_depth[:b].log(), depth_gt[:b].log(), mask=mask)
            acc_sie_masked = (sie_map_masked.view(b,-1).sum(1) / mask.view(b,-1).sum(1))**0.5

            norm_err_map_masked = utils.compute_angular_distance(recon_normal[:b], normal_gt[:b], mask=mask)
            acc_normal_masked = norm_err_map_masked.view(b,-1).sum(1) / mask.view(b,-1).sum(1)

            loss_dict['Image_masked'] = loss_l1_masked.sum()
            loss_dict['Perc_masked'] = loss_perc_im_masked.mean()
            loss_dict['MAE_masked'] = acc_mae_masked.mean()
            loss_dict['MSE_masked'] = acc_mse_masked.mean()
            loss_dict['SIE_masked'] = acc_sie_masked.mean()
            loss_dict['NorErr_masked'] = acc_normal_masked.mean()

        return loss_dict

    def forward(self, input, adjacency):
        """Feedforward once."""
        if self.load_gt_depth:
            input, depth_gt = input
        self.input_im = input.to(self.device) *2.-1.
        b, c, h, w = self.input_im.shape


        ## Run networks
        # canon_depth_pre = self.netD_dec(self.netD_enc(self.input_im))
        # cacon_albedo_pre = self.netA_dec(self.netA_enc(self.input_im))
        # canon_light_pre = self.netL_dec(self.netL_enc(self.input_im))
        # view_pre = self.netV_dec(self.netV_enc(self.input_im))

        depth_emb = self.netD_enc(self.input_im)
        albedo_emb = self.netA_enc(self.input_im)
        light_emb = self.netL_enc(self.input_im)
        view_emb = self.netV_enc(self.input_im)

        emb_dict = {
            'depth': depth_emb,
            'albedo': albedo_emb,
            'light': light_emb,
            'view': view_emb
        }

        concat_emb = torch.cat([
            depth_emb.unsqueeze_(2), 
            albedo_emb.unsqueeze_(2),
            light_emb.unsqueeze_(2),
            view_emb.unsqueeze_(2),
        ], dim=2)
        bs, ch, _, hz, wz = concat_emb.size()
        if self.dynamic_emb:
            normed_concat_emb = concat_emb.permute(0, 2, 1, 3, 4) / np.sqrt(ch * hz * wz)
            adjacency = torch.sigmoid(torch.einsum('nxacd, nyacd->nxy', normed_concat_emb, normed_concat_emb))
            adjacency_aug = adjacency.to(concat_emb.device).abs() + torch.eye(4, device=concat_emb.device)
        else:
            adjacency_aug = adjacency.to(concat_emb.device).abs() + torch.eye(4, device=concat_emb.device)


        if self.dynamic_emb:
            if self.use_concat:
                for i, factor in enumerate(['depth', 'albedo', 'light', 'view']):
                    concat_input = concat_emb * adjacency_aug[:, i][:, None, :, None, None].abs()
                    emb_dict[factor] = self.factor_mlp_list(concat_input.view(bs, -1, 1, 1))
            else:
                for i, factor in enumerate(['depth', 'albedo', 'light', 'view']):
                    emb_dict[factor] = (concat_emb * adjacency_aug[:, i][:, None, :, None, None].abs()).sum(2)

        else:
            if self.use_concat:
                for i, factor in enumerate(['depth', 'albedo', 'light', 'view']):
                    concat_input = concat_emb * adjacency_aug[i][None, None, :, None, None].abs()
                    emb_dict[factor] = self.factor_mlp_list[i](concat_input.view(bs, -1, 1, 1))
            else:
                for i, factor in enumerate(['depth', 'albedo', 'light', 'view']):
                    emb_dict[factor] = (concat_emb * adjacency_aug[i][None, None, :, None, None].abs()).sum(2)

        
        # for i, factor in enumerate(['depth', 'albedo', 'light', 'view']):
        #     concat_input = concat_emb * adjacency_aug[i][None, None, :, None, None].abs()
        #     emb_dict[factor] = self.factor_mlp_list[i](concat_input.view(bs, -1, 1, 1))
        
        depth_pre = self.netD_dec(emb_dict['depth'])
        albedo_pre = self.netA_dec(emb_dict['albedo'])
        light_pre = self.netL_dec(emb_dict['light'])
        view_pre = self.netV_dec(emb_dict['view'])


        ## predict canonical depth
        self.canon_depth_raw = depth_pre.squeeze(1)  # BxHxW
        self.canon_depth = self.canon_depth_raw - self.canon_depth_raw.view(b,-1).mean(1).view(b,1,1)
        self.canon_depth = self.canon_depth.tanh()
        self.canon_depth = self.depth_rescaler(self.canon_depth)

        ## optional depth smoothness loss (only used in synthetic car experiments)
        self.loss_depth_sm = ((self.canon_depth[:,:-1,:] - self.canon_depth[:,1:,:]) /(self.max_depth-self.min_depth)).abs().mean()
        self.loss_depth_sm += ((self.canon_depth[:,:,:-1] - self.canon_depth[:,:,1:]) /(self.max_depth-self.min_depth)).abs().mean()

        ## clamp border depth
        depth_border = torch.zeros(1,h,w-4).to(self.input_im.device)
        depth_border = nn.functional.pad(depth_border, (2,2), mode='constant', value=1)
        self.canon_depth = self.canon_depth*(1-depth_border) + depth_border *self.border_depth
        self.canon_depth = torch.cat([self.canon_depth, self.canon_depth.flip(2)], 0)  # flip

        ## predict canonical albedo
        self.canon_albedo = albedo_pre  # Bx3xHxW
        self.canon_albedo = torch.cat([self.canon_albedo, self.canon_albedo.flip(3)], 0)  # flip

        ## predict confidence map
        if self.use_conf_map:
            conf_sigma_l1, conf_sigma_percl = self.netC(self.input_im)  # Bx2xHxW
            self.conf_sigma_l1 = conf_sigma_l1[:,:1]
            self.conf_sigma_l1_flip = conf_sigma_l1[:,1:]
            self.conf_sigma_percl = conf_sigma_percl[:,:1]
            self.conf_sigma_percl_flip = conf_sigma_percl[:,1:]
        else:
            self.conf_sigma_l1 = None
            self.conf_sigma_l1_flip = None
            self.conf_sigma_percl = None
            self.conf_sigma_percl_flip = None

        ## predict lighting
        canon_light = light_pre.repeat(2,1)  # Bx4
        self.canon_light_a = self.amb_light_rescaler(canon_light[:,:1])  # ambience term
        self.canon_light_b = self.diff_light_rescaler(canon_light[:,1:2])  # diffuse term
        canon_light_dxy = canon_light[:,2:]
        self.canon_light_d = torch.cat([canon_light_dxy, torch.ones(b*2,1).to(self.input_im.device)], 1)
        self.canon_light_d = self.canon_light_d / ((self.canon_light_d**2).sum(1, keepdim=True))**0.5  # diffuse light direction

        ## shading
        self.canon_normal = self.renderer.get_normal_from_depth(self.canon_depth)
        self.canon_diffuse_shading = (self.canon_normal * self.canon_light_d.view(-1,1,1,3)).sum(3).clamp(min=0).unsqueeze(1)
        canon_shading = self.canon_light_a.view(-1,1,1,1) + self.canon_light_b.view(-1,1,1,1)*self.canon_diffuse_shading
        self.canon_im = (self.canon_albedo/2+0.5) * canon_shading *2-1

        ## predict viewpoint transformation
        self.view = view_pre.repeat(2,1)
        self.view = torch.cat([
            self.view[:,:3] *math.pi/180 *self.xyz_rotation_range,
            self.view[:,3:5] *self.xy_translation_range,
            self.view[:,5:] *self.z_translation_range], 1)

        ## reconstruct input view
        self.renderer.set_transform_matrices(self.view)
        self.recon_depth = self.renderer.warp_canon_depth(self.canon_depth)
        self.recon_normal = self.renderer.get_normal_from_depth(self.recon_depth)
        grid_2d_from_canon = self.renderer.get_inv_warped_2d_grid(self.recon_depth)
        self.recon_im = nn.functional.grid_sample(self.canon_im, grid_2d_from_canon, mode='bilinear')

        margin = (self.max_depth - self.min_depth) /2
        recon_im_mask = (self.recon_depth < self.max_depth+margin).float()  # invalid border pixels have been clamped at max_depth+margin
        recon_im_mask_both = recon_im_mask[:b] * recon_im_mask[b:]  # both original and flip reconstruction
        recon_im_mask_both = recon_im_mask_both.repeat(2,1,1).unsqueeze(1).detach()
        self.recon_im = self.recon_im * recon_im_mask_both

        ## render symmetry axis
        canon_sym_axis = torch.zeros(h, w).to(self.input_im.device)
        canon_sym_axis[:, w//2-1:w//2+1] = 1
        self.recon_sym_axis = nn.functional.grid_sample(canon_sym_axis.repeat(b*2,1,1,1), grid_2d_from_canon, mode='bilinear')
        self.recon_sym_axis = self.recon_sym_axis * recon_im_mask_both
        green = torch.FloatTensor([-1,1,-1]).to(self.input_im.device).view(1,3,1,1)
        self.input_im_symline = (0.5*self.recon_sym_axis) *green + (1-0.5*self.recon_sym_axis) *self.input_im.repeat(2,1,1,1)

        ## loss function
        self.loss_l1_im = self.photometric_loss(self.recon_im[:b], self.input_im, mask=recon_im_mask_both[:b], conf_sigma=self.conf_sigma_l1)
        self.loss_l1_im_flip = self.photometric_loss(self.recon_im[b:], self.input_im, mask=recon_im_mask_both[b:], conf_sigma=self.conf_sigma_l1_flip)
        self.loss_perc_im = self.PerceptualLoss(self.recon_im[:b], self.input_im, mask=recon_im_mask_both[:b], conf_sigma=self.conf_sigma_percl)
        self.loss_perc_im_flip = self.PerceptualLoss(self.recon_im[b:], self.input_im, mask=recon_im_mask_both[b:], conf_sigma=self.conf_sigma_percl_flip)
        lam_flip = 1 if self.trainer.current_epoch < self.lam_flip_start_epoch else self.lam_flip
        self.loss_total = self.loss_l1_im + lam_flip*self.loss_l1_im_flip + self.lam_perc*(self.loss_perc_im + lam_flip*self.loss_perc_im_flip) + self.lam_depth_sm*self.loss_depth_sm

        if self.dynamic_emb:
            metrics = {'loss': self.loss_total}
        else:
            dag_loss = self.DAGLoss(adjacency.abs())
            metrics = {'loss': self.loss_total, 'loss_dag': dag_loss}

        ## compute accuracy if gt depth is available
        if self.load_gt_depth:
            self.depth_gt = depth_gt[:,0,:,:].to(self.input_im.device)
            self.depth_gt = (1-self.depth_gt)*2-1
            self.depth_gt = self.depth_rescaler(self.depth_gt)
            self.normal_gt = self.renderer.get_normal_from_depth(self.depth_gt)

            # mask out background
            mask_gt = (self.depth_gt<self.depth_gt.max()).float()
            mask_gt = (nn.functional.avg_pool2d(mask_gt.unsqueeze(1), 3, stride=1, padding=1).squeeze(1) > 0.99).float()  # erode by 1 pixel
            mask_pred = (nn.functional.avg_pool2d(recon_im_mask[:b].unsqueeze(1), 3, stride=1, padding=1).squeeze(1) > 0.99).float()  # erode by 1 pixel
            mask = mask_gt * mask_pred
            self.acc_mae_masked = ((self.recon_depth[:b] - self.depth_gt[:b]).abs() *mask).view(b,-1).sum(1) / mask.view(b,-1).sum(1)
            self.acc_mse_masked = (((self.recon_depth[:b] - self.depth_gt[:b])**2) *mask).view(b,-1).sum(1) / mask.view(b,-1).sum(1)
            self.loss_l1_map = self.photometric_loss(self.recon_im[:b], self.input_im[:b], mask=mask[:b].unsqueeze(1), reduce=False)
            self.loss_l1_masked = self.loss_l1_map.view(b,-1).sum(1) / mask.view(b,-1).sum(1)
            self.loss_perc_im_masked = self.PerceptualLoss(self.recon_im[:b], self.input_im[:b], mask=mask[:b].unsqueeze(1))
            self.sie_map_masked = utils.compute_sc_inv_err(self.recon_depth[:b].log(), self.depth_gt[:b].log(), mask=mask)
            self.acc_sie_masked = (self.sie_map_masked.view(b,-1).sum(1) / mask.view(b,-1).sum(1))**0.5
            self.norm_err_map_masked = utils.compute_angular_distance(self.recon_normal[:b], self.normal_gt[:b], mask=mask)
            self.acc_normal_masked = self.norm_err_map_masked.view(b,-1).sum(1) / mask.view(b,-1).sum(1)

            metrics['Image_masked'] = self.loss_l1_masked.sum()
            metrics['Perc_masked'] = self.loss_perc_im_masked.mean()
            metrics['MAE_masked'] = self.acc_mae_masked.mean()
            metrics['MSE_masked'] = self.acc_mse_masked.mean()
            metrics['SIE_masked'] = self.acc_sie_masked.mean()
            metrics['NorErr_masked'] = self.acc_normal_masked.mean()

        return metrics

    def visualize(self, logger, total_iter, max_bs=25):
        b, c, h, w = self.input_im.shape
        b0 = min(max_bs, b)

        ## render rotations
        with torch.no_grad():
            v0 = torch.FloatTensor([-0.1*math.pi/180*60,0,0,0,0,0]).to(self.input_im.device).repeat(b0,1)
            canon_im_rotate = self.renderer.render_yaw(self.canon_im[:b0], self.canon_depth[:b0], v_before=v0, maxr=90).detach().cpu() /2.+0.5  # (B,T,C,H,W)
            canon_normal_rotate = self.renderer.render_yaw(self.canon_normal[:b0].permute(0,3,1,2), self.canon_depth[:b0], v_before=v0, maxr=90).detach().cpu() /2.+0.5  # (B,T,C,H,W)

        input_im = self.input_im[:b0].detach().cpu().numpy() /2+0.5
        input_im_symline = self.input_im_symline[:b0].detach().cpu() /2.+0.5
        canon_albedo = self.canon_albedo[:b0].detach().cpu() /2.+0.5
        canon_im = self.canon_im[:b0].detach().cpu() /2.+0.5
        recon_im = self.recon_im[:b0].detach().cpu() /2.+0.5
        recon_im_flip = self.recon_im[b:b+b0].detach().cpu() /2.+0.5
        canon_depth_raw_hist = self.canon_depth_raw.detach().unsqueeze(1).cpu()
        canon_depth_raw = self.canon_depth_raw[:b0].detach().unsqueeze(1).cpu() /2.+0.5
        canon_depth = ((self.canon_depth[:b0] -self.min_depth)/(self.max_depth-self.min_depth)).detach().cpu().unsqueeze(1)
        recon_depth = ((self.recon_depth[:b0] -self.min_depth)/(self.max_depth-self.min_depth)).detach().cpu().unsqueeze(1)
        canon_diffuse_shading = self.canon_diffuse_shading[:b0].detach().cpu()
        canon_normal = self.canon_normal.permute(0,3,1,2)[:b0].detach().cpu() /2+0.5
        recon_normal = self.recon_normal.permute(0,3,1,2)[:b0].detach().cpu() /2+0.5
        if self.use_conf_map:
            conf_map_l1 = 1/(1+self.conf_sigma_l1[:b0].detach().cpu()+EPS)
            conf_map_l1_flip = 1/(1+self.conf_sigma_l1_flip[:b0].detach().cpu()+EPS)
            conf_map_percl = 1/(1+self.conf_sigma_percl[:b0].detach().cpu()+EPS)
            conf_map_percl_flip = 1/(1+self.conf_sigma_percl_flip[:b0].detach().cpu()+EPS)

        canon_im_rotate_grid = [torchvision.utils.make_grid(img, nrow=int(math.ceil(b0**0.5))) for img in torch.unbind(canon_im_rotate, 1)]  # [(C,H,W)]*T
        canon_im_rotate_grid = torch.stack(canon_im_rotate_grid, 0).unsqueeze(0)  # (1,T,C,H,W)
        canon_normal_rotate_grid = [torchvision.utils.make_grid(img, nrow=int(math.ceil(b0**0.5))) for img in torch.unbind(canon_normal_rotate, 1)]  # [(C,H,W)]*T
        canon_normal_rotate_grid = torch.stack(canon_normal_rotate_grid, 0).unsqueeze(0)  # (1,T,C,H,W)

        ## write summary
        logger.add_scalar('Loss/loss_total', self.loss_total, total_iter)
        logger.add_scalar('Loss/loss_l1_im', self.loss_l1_im, total_iter)
        logger.add_scalar('Loss/loss_l1_im_flip', self.loss_l1_im_flip, total_iter)
        logger.add_scalar('Loss/loss_perc_im', self.loss_perc_im, total_iter)
        logger.add_scalar('Loss/loss_perc_im_flip', self.loss_perc_im_flip, total_iter)
        logger.add_scalar('Loss/loss_depth_sm', self.loss_depth_sm, total_iter)

        logger.add_histogram('Depth/canon_depth_raw_hist', canon_depth_raw_hist, total_iter)
        vlist = ['view_rx', 'view_ry', 'view_rz', 'view_tx', 'view_ty', 'view_tz']
        for i in range(self.view.shape[1]):
            logger.add_histogram('View/'+vlist[i], self.view[:,i], total_iter)
        logger.add_histogram('Light/canon_light_a', self.canon_light_a, total_iter)
        logger.add_histogram('Light/canon_light_b', self.canon_light_b, total_iter)
        llist = ['canon_light_dx', 'canon_light_dy', 'canon_light_dz']
        for i in range(self.canon_light_d.shape[1]):
            logger.add_histogram('Light/'+llist[i], self.canon_light_d[:,i], total_iter)

        def log_grid_image(label, im, nrow=int(math.ceil(b0**0.5)), iter=total_iter):
            im_grid = torchvision.utils.make_grid(im, nrow=nrow)
            logger.add_image(label, im_grid, iter)

        log_grid_image('Image/input_image_symline', input_im_symline)
        log_grid_image('Image/canonical_albedo', canon_albedo)
        log_grid_image('Image/canonical_image', canon_im)
        log_grid_image('Image/recon_image', recon_im)
        log_grid_image('Image/recon_image_flip', recon_im_flip)
        log_grid_image('Image/recon_side', canon_im_rotate[:,0,:,:,:])

        log_grid_image('Depth/canonical_depth_raw', canon_depth_raw)
        log_grid_image('Depth/canonical_depth', canon_depth)
        log_grid_image('Depth/recon_depth', recon_depth)
        log_grid_image('Depth/canonical_diffuse_shading', canon_diffuse_shading)
        log_grid_image('Depth/canonical_normal', canon_normal)
        log_grid_image('Depth/recon_normal', recon_normal)

        logger.add_histogram('Image/canonical_albedo_hist', canon_albedo, total_iter)
        logger.add_histogram('Image/canonical_diffuse_shading_hist', canon_diffuse_shading, total_iter)

        if self.use_conf_map:
            log_grid_image('Conf/conf_map_l1', conf_map_l1)
            logger.add_histogram('Conf/conf_sigma_l1_hist', self.conf_sigma_l1, total_iter)
            log_grid_image('Conf/conf_map_l1_flip', conf_map_l1_flip)
            logger.add_histogram('Conf/conf_sigma_l1_flip_hist', self.conf_sigma_l1_flip, total_iter)
            log_grid_image('Conf/conf_map_percl', conf_map_percl)
            logger.add_histogram('Conf/conf_sigma_percl_hist', self.conf_sigma_percl, total_iter)
            log_grid_image('Conf/conf_map_percl_flip', conf_map_percl_flip)
            logger.add_histogram('Conf/conf_sigma_percl_flip_hist', self.conf_sigma_percl_flip, total_iter)

        logger.add_video('Image_rotate/recon_rotate', canon_im_rotate_grid, total_iter, fps=4)
        logger.add_video('Image_rotate/canon_normal_rotate', canon_normal_rotate_grid, total_iter, fps=4)

        # visualize images and accuracy if gt is loaded
        if self.load_gt_depth:
            depth_gt = ((self.depth_gt[:b0] -self.min_depth)/(self.max_depth-self.min_depth)).detach().cpu().unsqueeze(1)
            normal_gt = self.normal_gt.permute(0,3,1,2)[:b0].detach().cpu() /2+0.5
            sie_map_masked = self.sie_map_masked[:b0].detach().unsqueeze(1).cpu() *1000
            norm_err_map_masked = self.norm_err_map_masked[:b0].detach().unsqueeze(1).cpu() /100

            logger.add_scalar('Acc_masked/Image_masked', self.loss_l1_masked.sum(), total_iter)
            logger.add_scalar('Acc_masked/Perc_masked', self.loss_perc_im_masked.mean(), total_iter)
            logger.add_scalar('Acc_masked/MAE_masked', self.acc_mae_masked.mean(), total_iter)
            logger.add_scalar('Acc_masked/MSE_masked', self.acc_mse_masked.mean(), total_iter)
            logger.add_scalar('Acc_masked/SIE_masked', self.acc_sie_masked.mean(), total_iter)
            logger.add_scalar('Acc_masked/NorErr_masked', self.acc_normal_masked.mean(), total_iter)

            log_grid_image('Depth_gt/depth_gt', depth_gt)
            log_grid_image('Depth_gt/normal_gt', normal_gt)
            log_grid_image('Depth_gt/sie_map_masked', sie_map_masked)
            log_grid_image('Depth_gt/norm_err_map_masked', norm_err_map_masked)
            log_grid_image('Depth_gt/l1_map_masked', self.loss_l1_map)

    def save_results(self, save_dir):
        b, c, h, w = self.input_im.shape

        with torch.no_grad():
            v0 = torch.FloatTensor([-0.1*math.pi/180*60,0,0,0,0,0]).to(self.input_im.device).repeat(b,1)
            canon_im_rotate = self.renderer.render_yaw(self.canon_im[:b], self.canon_depth[:b], v_before=v0, maxr=90, nsample=15)  # (B,T,C,H,W)
            canon_im_rotate = canon_im_rotate.clamp(-1,1).detach().cpu() /2+0.5
            canon_normal_rotate = self.renderer.render_yaw(self.canon_normal[:b].permute(0,3,1,2), self.canon_depth[:b], v_before=v0, maxr=90, nsample=15)  # (B,T,C,H,W)
            canon_normal_rotate = canon_normal_rotate.clamp(-1,1).detach().cpu() /2+0.5

        input_im = self.input_im[:b].detach().cpu().numpy() /2+0.5
        input_im_symline = self.input_im_symline.detach().cpu().numpy() /2.+0.5
        canon_albedo = self.canon_albedo[:b].detach().cpu().numpy() /2+0.5
        canon_im = self.canon_im[:b].clamp(-1,1).detach().cpu().numpy() /2+0.5
        recon_im = self.recon_im[:b].clamp(-1,1).detach().cpu().numpy() /2+0.5
        recon_im_flip = self.recon_im[b:].clamp(-1,1).detach().cpu().numpy() /2+0.5
        canon_depth = ((self.canon_depth[:b] -self.min_depth)/(self.max_depth-self.min_depth)).clamp(0,1).detach().cpu().unsqueeze(1).numpy()
        recon_depth = ((self.recon_depth[:b] -self.min_depth)/(self.max_depth-self.min_depth)).clamp(0,1).detach().cpu().unsqueeze(1).numpy()
        canon_diffuse_shading = self.canon_diffuse_shading[:b].detach().cpu().numpy()
        canon_normal = self.canon_normal[:b].permute(0,3,1,2).detach().cpu().numpy() /2+0.5
        recon_normal = self.recon_normal[:b].permute(0,3,1,2).detach().cpu().numpy() /2+0.5
        if self.use_conf_map:
            conf_map_l1 = 1/(1+self.conf_sigma_l1[:b].detach().cpu().numpy()+EPS)
            conf_map_l1_flip = 1/(1+self.conf_sigma_l1_flip[:b].detach().cpu().numpy()+EPS)
            conf_map_percl = 1/(1+self.conf_sigma_percl[:b].detach().cpu().numpy()+EPS)
            conf_map_percl_flip = 1/(1+self.conf_sigma_percl_flip[:b].detach().cpu().numpy()+EPS)
        canon_light = torch.cat([self.canon_light_a, self.canon_light_b, self.canon_light_d], 1)[:b].detach().cpu().numpy()
        view = self.view[:b].detach().cpu().numpy()

        canon_im_rotate_grid = [torchvision.utils.make_grid(img, nrow=int(math.ceil(b**0.5))) for img in torch.unbind(canon_im_rotate,1)]  # [(C,H,W)]*T
        canon_im_rotate_grid = torch.stack(canon_im_rotate_grid, 0).unsqueeze(0).numpy()  # (1,T,C,H,W)
        canon_normal_rotate_grid = [torchvision.utils.make_grid(img, nrow=int(math.ceil(b**0.5))) for img in torch.unbind(canon_normal_rotate,1)]  # [(C,H,W)]*T
        canon_normal_rotate_grid = torch.stack(canon_normal_rotate_grid, 0).unsqueeze(0).numpy()  # (1,T,C,H,W)

        sep_folder = True
        utils.save_images(save_dir, input_im, suffix='input_image', sep_folder=sep_folder)
        utils.save_images(save_dir, input_im_symline, suffix='input_image_symline', sep_folder=sep_folder)
        utils.save_images(save_dir, canon_albedo, suffix='canonical_albedo', sep_folder=sep_folder)
        utils.save_images(save_dir, canon_im, suffix='canonical_image', sep_folder=sep_folder)
        utils.save_images(save_dir, recon_im, suffix='recon_image', sep_folder=sep_folder)
        utils.save_images(save_dir, recon_im_flip, suffix='recon_image_flip', sep_folder=sep_folder)
        utils.save_images(save_dir, canon_depth, suffix='canonical_depth', sep_folder=sep_folder)
        utils.save_images(save_dir, recon_depth, suffix='recon_depth', sep_folder=sep_folder)
        utils.save_images(save_dir, canon_diffuse_shading, suffix='canonical_diffuse_shading', sep_folder=sep_folder)
        utils.save_images(save_dir, canon_normal, suffix='canonical_normal', sep_folder=sep_folder)
        utils.save_images(save_dir, recon_normal, suffix='recon_normal', sep_folder=sep_folder)
        if self.use_conf_map:
            utils.save_images(save_dir, conf_map_l1, suffix='conf_map_l1', sep_folder=sep_folder)
            utils.save_images(save_dir, conf_map_l1_flip, suffix='conf_map_l1_flip', sep_folder=sep_folder)
            utils.save_images(save_dir, conf_map_percl, suffix='conf_map_percl', sep_folder=sep_folder)
            utils.save_images(save_dir, conf_map_percl_flip, suffix='conf_map_percl_flip', sep_folder=sep_folder)
        utils.save_txt(save_dir, canon_light, suffix='canonical_light', sep_folder=sep_folder)
        utils.save_txt(save_dir, view, suffix='viewpoint', sep_folder=sep_folder)

        utils.save_videos(save_dir, canon_im_rotate_grid, suffix='image_video', sep_folder=sep_folder, cycle=True)
        utils.save_videos(save_dir, canon_normal_rotate_grid, suffix='normal_video', sep_folder=sep_folder, cycle=True)

        # save scores if gt is loaded
        if self.load_gt_depth:
            depth_gt = ((self.depth_gt[:b] -self.min_depth)/(self.max_depth-self.min_depth)).clamp(0,1).detach().cpu().unsqueeze(1).numpy()
            normal_gt = self.normal_gt[:b].permute(0,3,1,2).detach().cpu().numpy() /2+0.5
            utils.save_images(save_dir, depth_gt, suffix='depth_gt', sep_folder=sep_folder)
            utils.save_images(save_dir, normal_gt, suffix='normal_gt', sep_folder=sep_folder)

            all_scores = torch.stack([
                self.acc_mae_masked.detach().cpu(),
                self.acc_mse_masked.detach().cpu(),
                self.acc_sie_masked.detach().cpu(),
                self.acc_normal_masked.detach().cpu(),
                self.loss_l1_masked.detach().cpu(),
                self.loss_perc_im_masked.detach().cpu(),
            ], 1)
            if not hasattr(self, 'all_scores'):
                self.all_scores = torch.FloatTensor()
            self.all_scores = torch.cat([self.all_scores, all_scores], 0)

    def save_scores(self, path):
        # save scores if gt is loaded
        if self.load_gt_depth:
            header = 'MAE_masked, \
                      MSE_masked, \
                      SIE_masked, \
                      NorErr_masked,  \
                      Image_masked, \
                      Perc_masked'
            mean = self.all_scores.mean(0)
            std = self.all_scores.std(0)
            header = header + '\nMean: ' + ',\t'.join(['%.8f'%x for x in mean])
            header = header + '\nStd: ' + ',\t'.join(['%.8f'%x for x in std])
            utils.save_scores(path, self.all_scores, header=header)