"""
model.py - warpper and utility functions for network training
Compute loss, back-prop, update parameters, logging, etc.
"""


import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from model.network import STCN
from model.losses import LossComputer, iou_hooks_mo, iou_hooks_so
from util.log_integrator import Integrator
from util.image_saver import pool_pairs
import torch.nn.functional as F
import numpy as np
from PIL import Image
from model.accuracy import object_level_Accuracy

import wandb


class STCNModel:
    def __init__(self, para, logger=None, save_path="model_saves", local_rank=0, world_size=1):
        self.para = para
        self.single_object = para['single_object']
        self.local_rank = local_rank

        # self.STCN = nn.parallel.DistributedDataParallel(
        #     STCN(self.single_object).cuda(), 
        #     device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

        self.STCN = STCN(self.single_object).cuda()

        # Setup logger when local_rank=0
        self.logger = logger
        self.save_path = save_path
        if logger is not None:
            self.last_time = time.time()
        self.train_integrator = Integrator(self.logger, distributed=True, local_rank=local_rank, world_size=world_size)
        if self.single_object:
            self.train_integrator.add_hook(iou_hooks_so)
        else:
            self.train_integrator.add_hook(iou_hooks_mo)
        self.loss_computer = LossComputer(para)

        self.train()
        self.optimizer = optim.Adam(filter(
            lambda p: p.requires_grad, self.STCN.parameters()), lr=para['lr'], weight_decay=1e-7)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, para['steps'], para['gamma'])
        if para['amp']:
            self.scaler = torch.cuda.amp.GradScaler()

        # Logging info
        self.report_interval = 100
        self.save_im_interval = 800
        self.save_model_interval = 50000
        if para['debug']:
            self.report_interval = self.save_im_interval = 1
        
        self.losses = []

        self.acc_metric = object_level_Accuracy()

    def val_pass(self, data, it=0):
        for k, v in data.items():
            if type(v) != list and type(v) != dict and type(v) != int:
                data[k] = v.cuda(non_blocking=True)

        out = {}
        Fs = data['instance_seg'] #used for Key and Value [16, 1, 608, 800]
        Qs = data['gaze_heatmap'] #used for Query [16, 1, 608, 800]
        Ms = data['label'] #Label mask [16, 1, 608, 800]
        inst_metric = data['inst_metrics']

        with torch.cuda.amp.autocast(enabled=self.para['amp']):
            # key features never change, compute once
            k16, kf16= self.STCN('encode_key', Fs)  # [16, 64, 38, 50], [16, 256, 38, 50]

            v16, vf16 = self.STCN('encode_value', Fs)

            q16, qf16= self.STCN('encode_query', Qs)

            #TODO: finish this once query encoder is implemented
            # would also need to modify the segementation part of the network

            # segment args: query-key, query-value, query-f8, query-f4, memory-key, memory-value
            # k16, v16, kf8, kf4, q16, qf16 ???

            logits, mask = self.STCN('segment', k16, v16, q16, qf16)

            out['logits'] = logits
            out['mask'] = mask
            
            #object level accuracy
            acc, preds, gts, raw_preds, obj_ids = self.acc_metric.forward(mask, Ms, inst_metric)
            #wandb.log({'val_object_level_accuracy': acc})

            # log the ground truth label masks and the predicted logits and mask for each sample in the batch as images into wandb
            for b in range(data['label'].shape[0]):
                # get rid of the batch dimension
                gt_mask = Ms[b].cpu().squeeze()
                gt_fin  = np.zeros([gt_mask.shape[0], gt_mask.shape[1], 3])
                gt_fin[gt_mask == 0] = np.array([0, 255, 0])
                gt_fin[gt_mask == 1] = np.array([255, 0, 0])
                gt_fin[gt_mask == 2] = np.array([0, 0, 0])

                pr_mask = mask[b].cpu().squeeze().detach().numpy()
                pr_mask = np.argmax(pr_mask, axis=0)
                #print(np.unique(pr_mask))
                pr_fin = np.zeros([pr_mask.shape[0], pr_mask.shape[1], 3])                
                pr_fin[pr_mask == 0] = np.array([0, 255, 0]) # green for aware
                pr_fin[pr_mask == 1] = np.array([255, 0, 0]) # red for aware
                #mask_b = mask[b].squeeze()
                gt_image = Image.fromarray(np.uint8(gt_fin))
                #logits_image = F.to_pil_image(logits_b)
                mask_image = Image.fromarray(np.uint8(pr_fin))

                heatmap_image = Qs[b].cpu().squeeze()
                wandb.log({'val_gt_image': wandb.Image(gt_image, caption='Val Ground Truth Label Mask'), 'val_mask_image': wandb.Image(mask_image, caption='Val Predicted Mask'), 'val_heatmap_image': wandb.Image(heatmap_image, caption='Val Gaze Heatmap')})

            losses = self.loss_computer.compute({**data, **out}, it)
            #val_losses = {'val_loss': losses['loss'], 'val_p': losses['p'], 'val_total_loss': losses['total_loss'], 'val_hide_iou/i': losses['hide_iou/i'], 'val_hide_iou/u': losses['hide_iou/u']}
            val_losses = losses['loss']
            # wandb.log(val_losses)
            return val_losses, acc
            
    def do_pass(self, data, it=0):
        # No need to store the gradient outside training
        torch.set_grad_enabled(self._is_train)

        for k, v in data.items():
            if type(v) != list and type(v) != dict and type(v) != int:
                data[k] = v.cuda(non_blocking=True)

        out = {}
        Fs = data['instance_seg'] #used for Key and Value [16, 1, 608, 800]
        Qs = data['gaze_heatmap'] #used for Query [16, 1, 608, 800]
        Ms = data['label'] #Label mask [16, 1, 608, 800]
        inst_metric = data['inst_metrics']

        with torch.cuda.amp.autocast(enabled=self.para['amp']):
            # key features never change, compute once
            k16, kf16= self.STCN('encode_key', Fs)  # [16, 64, 38, 50], [16, 256, 38, 50]

            v16, vf16 = self.STCN('encode_value', Fs)

            q16, qf16= self.STCN('encode_query', Qs)

            logits, mask = self.STCN('segment', k16, v16, q16, qf16)

            out['logits'] = logits
            out['mask'] = mask

            #object level accuracy
            acc, preds, gts, raw_preds, obj_ids = self.acc_metric.forward(mask, Ms, inst_metric)
            wandb.log({'object_level_accuracy': acc})


            # log the ground truth label masks and the predicted logits and mask for each sample in the batch as images into wandb
            for b in range(data['label'].shape[0]):
                # get rid of the batch dimension
                gt_mask = Ms[b].cpu().squeeze()
                gt_fin  = np.zeros([gt_mask.shape[0], gt_mask.shape[1], 3])
                gt_fin[gt_mask == 0] = np.array([0, 255, 0])
                gt_fin[gt_mask == 1] = np.array([255, 0, 0])
                gt_fin[gt_mask == 2] = np.array([0, 0, 0])
                logits_b = logits[b].squeeze()

                pr_mask = mask[b].cpu().squeeze().detach().numpy()
                pr_mask = np.argmax(pr_mask, axis=0)
                #print(np.unique(pr_mask))
                pr_fin = np.zeros([pr_mask.shape[0], pr_mask.shape[1], 3])                
                pr_fin[pr_mask == 0] = np.array([0, 255, 0]) # green for aware
                pr_fin[pr_mask == 1] = np.array([255, 0, 0]) # red for aware
                #mask_b = mask[b].squeeze()
                gt_image = Image.fromarray(np.uint8(gt_fin))
                #logits_image = F.to_pil_image(logits_b)
                mask_image = Image.fromarray(np.uint8(pr_fin))

                heatmap_image = Qs[b].cpu().squeeze()
                wandb.log({'gt_image': wandb.Image(gt_image, caption='Ground Truth Label Mask'), 'mask_image': wandb.Image(mask_image, caption='Predicted Mask'), 'heatmap_image': wandb.Image(heatmap_image, caption='Gaze Heatmap')})

            if self._do_log or self._is_train:
                losses = self.loss_computer.compute({**data, **out}, it)
                self.losses.append(losses['total_loss'])
                wandb.log(losses)

            # Backward pass
            # This should be done outside autocast
            # but I trained it like this and it worked fine
            # so I am keeping it this way for reference
            self.optimizer.zero_grad(set_to_none=True)
            if self.para['amp']:
                self.scaler.scale(losses['total_loss']).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses['total_loss'].backward() 
                self.optimizer.step()
            self.scheduler.step()

    def save(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        model_path = self.save_path + ('_%s.pth' % it)
        torch.save(self.STCN.module.state_dict(), model_path)
        print('Model saved to %s.' % model_path)

        self.save_checkpoint(it)

    def save_checkpoint(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        checkpoint_path = self.save_path + '_checkpoint.pth'
        checkpoint = { 
            'it': it,
            'network': self.STCN.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()}
        torch.save(checkpoint, checkpoint_path)

        print('Checkpoint saved to %s.' % checkpoint_path)

    def load_model(self, path):
        # This method loads everything and should be used to resume training
        map_location = 'cuda:%d' % self.local_rank
        checkpoint = torch.load(path, map_location={'cuda:0': map_location})

        it = checkpoint['it']
        network = checkpoint['network']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']

        map_location = 'cuda:%d' % self.local_rank
        self.STCN.module.load_state_dict(network)
        self.optimizer.load_state_dict(optimizer)
        self.scheduler.load_state_dict(scheduler)

        print('Model loaded.')

        return it

    def load_network(self, path):
        # This method loads only the network weight and should be used to load a pretrained model
        map_location = 'cuda:%d' % self.local_rank
        src_dict = torch.load(path, map_location={'cuda:0': map_location})

        # Maps SO weight (without other_mask) to MO weight (with other_mask)
        for k in list(src_dict.keys()):
            if k == 'value_encoder.conv1.weight':
                if src_dict[k].shape[1] == 4:
                    pads = torch.zeros((64,1,7,7), device=src_dict[k].device)
                    nn.init.orthogonal_(pads)
                    src_dict[k] = torch.cat([src_dict[k], pads], 1)

        self.STCN.module.load_state_dict(src_dict)
        print('Network weight loaded:', path)

    def train(self):
        self._is_train = True
        self._do_log = True
        self.integrator = self.train_integrator
        # Shall be in eval() mode to freeze BN parameters
        self.STCN.eval()
        return self

    def val(self):
        self._is_train = False
        self._do_log = True
        self.STCN.eval()
        return self

    def test(self):
        self._is_train = False
        self._do_log = False
        self.STCN.eval()
        return self

