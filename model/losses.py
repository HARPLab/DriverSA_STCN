import torch
import torch.nn as nn
import torch.nn.functional as F
from util.tensor_util import compute_tensor_iu

from collections import defaultdict


def get_iou_hook(values):
    return 'iou/iou', (values['hide_iou/i']+1)/(values['hide_iou/u']+1)

def get_sec_iou_hook(values):
    return 'iou/sec_iou', (values['hide_iou/sec_i']+1)/(values['hide_iou/sec_u']+1)

iou_hooks_so = [
    get_iou_hook,
]

iou_hooks_mo = [
    get_iou_hook,
    get_sec_iou_hook,
]


# https://stackoverflow.com/questions/63735255/how-do-i-compute-bootstrapped-cross-entropy-loss-in-pytorch
class BootstrappedCE(nn.Module):
    def __init__(self, start_warm=20000, end_warm=70000, top_p=0.15):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p

    def forward(self, input, target, it):
        # convert input from (N, C, H, W) to (N, C, H*W)
        input = input.flatten(2)
        # target has shape (N, C, H, W) -- convert to (N, H*W, C)
        target = target.permute(0, 2, 3, 1)
        target = target.reshape(target.shape[0], -1, target.shape[3])
        # exclude class dimension
        target = target.squeeze(-1)

        if it < self.start_warm:
            return F.cross_entropy(input, target, weight=torch.tensor([1, 1, 10**(-2)]).to("cuda:0")), 1.0

        raw_loss = F.cross_entropy(input, target, reduction='none').view(-1)
        num_pixels = raw_loss.numel()

        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1-self.top_p)*((self.end_warm-it)/(self.end_warm-self.start_warm))
        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        return loss.mean(), this_p


class LossComputer:
    def __init__(self, para):
        super().__init__()
        self.para = para
        self.bce = BootstrappedCE()

    def compute(self, data, it):
        losses = defaultdict(int)

        b, _, _, _ = data['label'].shape
        #selector = data.get('selector', None)

        # for i in range(1, s):
        #     # Have to do it in a for-loop like this since not every entry has the second object
        #     # Well it's not a lot of iterations anyway
        for j in range(b):
            # print(data['logits'].shape)
            # print(data['label'].shape)
            # print(data['mask'].shape)
            loss, p =  self.bce(data['logits'][j:j+1], data['label'][j:j+1], it)
            # if selector is not None and selector[j][1] > 0.5:
            #     loss, p = self.bce(data['logits_%d'%i][j:j+1], data['cls_gt'][j:j+1,i], it)
            # else:
            #     loss, p = self.bce(data['logits_%d'%i][j:j+1,:2], data['cls_gt'][j:j+1,i], it)

            losses['loss'] += loss / b
            losses['p'] += p / b

            losses['total_loss'] += losses['loss']

            new_total_i, new_total_u = compute_tensor_iu(data['mask']>0.5, data['label']>0.5)
            losses['hide_iou/i'] += new_total_i
            losses['hide_iou/u'] += new_total_u

            # if selector is not None:
            #     new_total_i, new_total_u = compute_tensor_iu(data['sec_mask_%d'%i]>0.5, data['sec_gt'][:,i]>0.5)
            #     losses['hide_iou/sec_i'] += new_total_i
            #     losses['hide_iou/sec_u'] += new_total_u

        return losses
