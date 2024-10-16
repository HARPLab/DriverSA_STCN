"""
network.py - The core of the neural network
Defines the structure and memory operations
Modifed from STM: https://github.com/seoungwugoh/STM

The trailing number of a variable usually denote the stride
e.g. f16 -> encoded features with stride 16
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules import *


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress = ResBlock(1024, 512)
        self.up_16_8 = UpsampleBlock(512, 512, 256) # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(256, 256, 256) # 1/8 -> 1/4

        self.pred = nn.Conv2d(256, 1, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, f16, f8, f4):
        x = self.compress(f16)
        x = self.up_16_8(f8, x)
        x = self.up_8_4(f4, x)

        x = self.pred(F.relu(x))
        
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return x


class MemoryReader(nn.Module):
    def __init__(self):
        super().__init__()
 
    def get_affinity(self, mk, qk):
        B, CK, H, W = mk.shape
        mk = mk.flatten(start_dim=2)
        qk = qk.flatten(start_dim=2)

        # See supplementary material
        a_sq = mk.pow(2).sum(1).unsqueeze(2)
        ab = mk @ qk

        affinity = (2*ab-a_sq) / math.sqrt(CK)   # B, THW, HW
        
        # softmax operation; aligned the evaluation style
        maxes = torch.max(affinity, dim=1, keepdim=True)[0]
        x_exp = torch.exp(affinity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        affinity = x_exp / x_exp_sum 

        return affinity

    def readout(self, affinity, mv, qv):
        B, CV, H, W = mv.shape

        mo = mv.view(B, CV, H*W) 
        mem = torch.bmm(mo, affinity) # Weighted-sum B, CV, HW
        mem = mem.view(B, CV, H, W)

        mem_out = torch.cat([mem, qv], dim=1)

        return mem_out


class STCN(nn.Module):
    def __init__(self, single_object):
        super().__init__()
        self.single_object = single_object

        self.key_encoder = KeyEncoder()
        self.value_encoder = ValueEncoder() 
        self.query_encoder = QueryEncoder()

        # Projection from f16 feature space to key space
        self.key_proj = KeyProjection(1024, keydim=64)

        # Compress f16 a bit to use in decoding later on
        self.key_comp = nn.Conv2d(1024, 512, kernel_size=3, padding=1)

        self.memory = MemoryReader()
        self.decoder = Decoder()

    def aggregate(self, prob):
        new_prob = torch.cat([
            torch.prod(1-prob, dim=1, keepdim=True),
            prob
        ], 1).clamp(1e-7, 1-1e-7)
        logits = torch.log((new_prob /(1-new_prob)))
        return logits

    def encode_key(self, frame): 
        #Vos dataset frame shape: [b, 3, 3, 384, 384]
        #awareness dataset frame shape: [b, 1, 608, 800]
        # input: b*t*c*h*w
        b = frame.shape[:1]

        f16, f8, f4 = self.key_encoder(frame.flatten(start_dim=0, end_dim=1))
        k16 = self.key_proj(f16)
        f16_thin = self.key_comp(f16)

        # B*C*T*H*W
        k16 = k16.view(b, *k16.shape[-3:]).contiguous()

        # B*T*C*H*W
        f16_thin = f16_thin.view(b, *f16_thin.shape[-3:])
        f16 = f16.view(b, *f16.shape[-3:])
        f8 = f8.view(b, *f8.shape[-3:])
        f4 = f4.view(b, *f4.shape[-3:])

        return k16, f16_thin, f16, f8, f4

    def encode_value(self, frame): 
        b = frame.shape[:1]

        f16, f8, f4 = self.key_encoder(frame.flatten(start_dim=0, end_dim=1))

        v16 = self.key_proj(f16)
        f16_thin = self.key_comp(f16)

        # B*C*T*H*W
        v16 = v16.view(b, *v16.shape[-3:]).contiguous()

        # B*T*C*H*W
        f16_thin = f16_thin.view(b, *f16_thin.shape[-3:])
        f16 = f16.view(b, *f16.shape[-3:])
        f16 = f16.view(b, *f16.shape[-3:])
        f8 = f8.view(b, *f8.shape[-3:])
        f4 = f4.view(b, *f4.shape[-3:])

        return v16, f16_thin, f16, f8, f4

    def encode_query(self, gaze_heatmap):
        # gaze heatmap image encoding

        # input: b*t*c*h*w
        b = gaze_heatmap.shape[:1]

        f16, f8, f4 = self.key_encoder(gaze_heatmap.flatten(start_dim=0, end_dim=1))
        q16 = self.key_proj(f16)
        f16_thin = self.key_comp(f16)

        # B*C*T*H*W
        q16 = q16.view(b, *q16.shape[-3:]).contiguous()

        # B*T*C*H*W
        f16_thin = f16_thin.view(b, *f16_thin.shape[-3:])
        f16 = f16.view(b, *f16.shape[-3:])
        f8 = f8.view(b, *f8.shape[-3:])
        f4 = f4.view(b, *f4.shape[-3:])

        return q16, f16_thin, f16, f8, f4

    def self_attention(self, k16, q16, v16):
        # self attention between key instance segmentation and query gaze heatmap
        # k16, q16, v16 shape: [b, c, h, w]
        similarity  = self.memory.get_affinity(k16, q16) # TODO: change affinitiy calculation
        scaled = similarity / math.sqrt(k16.shape[1])
        attention = F.softmax(scaled, dim=-1)
        out = attention @ v16
        return out


    def segment(self, qk16, qv16, qf8, qf4, mk16, mv16, selector=None): 
        # k16, v16, kf8, kf4, q16, qf16 ???
        # q - query, m - memory
        # qv16 is f16_thin above
        #affinity = self.memory.get_affinity(mk16, qk16)

        attention = self.self_attention(qk16, mk16, qv16)
        
        if self.single_object:
            logits = self.decoder(self.memory.readout(affinity, mv16, qv16), qf8, qf4)
            prob = torch.sigmoid(logits)
        else:
            logits = torch.cat([
                self.decoder(self.memory.readout(affinity, mv16[:,0], qv16), qf8, qf4),
                self.decoder(self.memory.readout(affinity, mv16[:,1], qv16), qf8, qf4),
            ], 1)

            prob = torch.sigmoid(logits)
            prob = prob * selector.unsqueeze(2).unsqueeze(2)

        logits = self.aggregate(prob)
        prob = F.softmax(logits, dim=1)[:, 1:]

        return logits, prob

    def forward(self, mode, *args, **kwargs):
        if mode == 'encode_key':
            return self.encode_key(*args, **kwargs)
        elif mode == 'encode_value':
            return self.encode_value(*args, **kwargs)
        elif mode == 'encode_query':
            return self.encode_query(*args, **kwargs)
        elif mode == 'segment':
            return self.segment(*args, **kwargs)
        else:
            raise NotImplementedError


