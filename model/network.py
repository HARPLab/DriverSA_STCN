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
        self.compress = ResBlock(64, 256)
        #self.up_16_8 = UpsampleBlock(512, 512, 256) # 1/16 -> 1/8
        #self.up_8_4 = UpsampleBlock(256, 256, 256) # 1/8 -> 1/4

        self.pred = nn.Conv2d(256, 3, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, f16):
        x = self.compress(f16)
        #x = self.up_16_8(f8, x)
        #x = self.up_8_4(f4, x)

        x = self.pred(F.relu(x))
        
        x = F.interpolate(x, size=(600, 800), mode='bilinear', align_corners=False)
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


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
   
    def forward(self, k, q, v):
        batch_size, time_steps, channels, height, width = k.shape

        # Spatial Attention
        k_flat = k.view(batch_size*time_steps, channels, height*width).transpose(-2, -1)
        q_flat = q.view(batch_size*time_steps, channels, height*width)

        similarity = k_flat @ q_flat
        scaled = similarity / math.sqrt(self.input_dim)
        attention = F.softmax(scaled, dim=-1)
        v_flat = v.view(batch_size*time_steps, channels, height*width).transpose(-2, -1)
        spatial_out = attention @ v_flat 

        #Temporal Attention
        spatial_out = spatial_out.view(batch_size, time_steps, channels, height, width)
        k_flat_temp = spatial_out.transpose(1, 2).reshape(batch_size, channels, time_steps, -1).transpose(-2, -1)
        q_flat_temp = q.transpose(1, 2).reshape(batch_size, channels, time_steps, -1)

        similarity_temp = k_flat_temp @ q_flat_temp
        scaled_temp = similarity_temp / math.sqrt(self.input_dim)
        attention_temp = F.softmax(scaled_temp, dim=-1)
        v_flat_temp = spatial_out.transpose(1, 2).reshape(batch_size, channels, time_steps, -1).transpose(-2, -1)
        temp_out = attention_temp @ v_flat_temp

        out = temp_out.transpose(-2, -1).view(batch_size, channels, time_steps, height, width)

        out = out.mean(dim=2)
        # k = k.transpose(-2, -1)
        # similarity = k @ q
        # scaled = similarity / math.sqrt(self.input_dim)
        # attention = F.softmax(scaled, dim=-1)
        # v = v.transpose(-2, -1)
        # out = attention @ v
        # out = out.transpose(-2, -1)
        return out



class STCN(nn.Module):
    def __init__(self, single_object):
        super().__init__()
        self.single_object = single_object

        self.key_encoder = KeyEncoder()
        self.value_encoder = ValueEncoder() 
        self.query_encoder = QueryEncoder()

        # Projection from f16 feature space to key space
        self.key_proj = KeyProjection(256, keydim=64)

        # Compress f16 a bit to use in decoding later on
        self.key_comp = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        self.memory = MemoryReader()
        self.decoder = Decoder()

    def aggregate(self, prob):
        # new_prob = torch.cat([
        #     torch.prod(1-prob, dim=1, keepdim=True),
        #     prob
        # ], 1).clamp(1e-7, 1-1e-7)
        # logits = torch.log((new_prob /(1-new_prob)))
        return prob.clamp(-50, 50)
        return logits

    def encode_key(self, frame): 
        #Vos dataset frame shape: [b, 3, 3, 384, 384]
        #awareness dataset frame shape: [b, 12, 1, 600, 800]
        # input: b*t*c*h*w
        batch_size, time_steps, channels, height, width = frame.shape

        #NOTE: batch size of 24 is too big -- 16 might work
        
        encoding_outputs = []
        projection_outputs = []
        for i in range(time_steps):
            f16 = self.key_encoder(frame[:,i, :, :, :]) # [b, 256, 38, 50]
            k16 = self.key_proj(f16) 
            k16 = k16.contiguous() #[b, 64, 38, 50]
            encoding_outputs.append(f16)
            projection_outputs.append(k16)


        k16_tensor = torch.stack(projection_outputs, dim=1) #[16, 12, 64, 38, 50]
        f16_tensor = torch.stack(encoding_outputs, dim=1)


        # f16 = self.key_encoder(frame)
        # print("f16 shape: ", f16.shape)
        # k16 = self.key_proj(f16)
        # print("k16 shape: ", k16.shape)
        # f16_thin = self.key_comp(f16)

        # # B*C*H*W
        # #k16 = k16.view(b, *k16.shape[-3:]).contiguous()
        # k16_tensor = k16.contiguous()
        # #print("k16 shape after view: ", k16.shape)

        # # B*CHW
        # #f16_thin = f16_thin.view(b, *f16_thin.shape[-3:])
        # #f16 = f16.view(b, *f16.shape[-3:])
        # f16_tensor = f16
        # #f8 = f8.view(b, *f8.shape[-3:])
        # #f4 = f4.view(b, *f4.shape[-3:])

        return k16_tensor, f16_tensor

    def encode_value(self, frame): 
        batch_size, time_steps, channels, height, width = frame.shape

        #NOTE: batch size of 24 is too big -- 16 might work
        
        encoding_outputs = []
        projection_outputs = []
        for i in range(time_steps):
            f16 = self.value_encoder(frame[:,i, :, :, :]) # [b, 256, 38, 50]
            v16 = self.key_proj(f16) 
            v16 = v16.contiguous() #[b, 64, 38, 50]
            encoding_outputs.append(f16)
            projection_outputs.append(v16)
        

        v16_tensor = torch.stack(projection_outputs, dim=1)
        f16_tensor = torch.stack(encoding_outputs, dim=1)
        # b = frame.shape[:1]

        # f16 = self.value_encoder(frame)

        # v16 = self.key_proj(f16)
        # f16_thin = self.key_comp(f16)

        # # B*C*T*H*W
        # #v16 = v16.view(b, *v16.shape[-3:]).contiguous()
        # v16 = v16.contiguous()

        # # B*T*C*H*W
        # #f16_thin = f16_thin.view(b, *f16_thin.shape[-3:])
        # #f16 = f16.view(b, *f16.shape[-3:])
        # f16 = f16
        # # f8 = f8.view(b, *f8.shape[-3:])
        # # f4 = f4.view(b, *f4.shape[-3:])

        return v16_tensor, f16_tensor

    def encode_query(self, gaze_heatmap):
        # gaze heatmap image encoding

        batch_size, time_steps, channels, height, width = gaze_heatmap.shape

        #NOTE: batch size of 24 is too big -- 16 might work
        
        encoding_outputs = []
        projection_outputs = []
        for i in range(time_steps):
            f16 = self.query_encoder(gaze_heatmap[:,i, :, :, :]) # [b, 256, 38, 50]
            q16 = self.key_proj(f16) 
            q16 = q16.contiguous() #[b, 64, 38, 50]
            encoding_outputs.append(f16)
            projection_outputs.append(q16)
        

        q16_tensor = torch.stack(projection_outputs, dim=1)
        f16_tensor = torch.stack(encoding_outputs, dim=1)

        # input: b*t*c*h*w
        # b = gaze_heatmap.shape[:1]

        # f16 = self.query_encoder(gaze_heatmap)
        # q16 = self.key_proj(f16)
        # f16_thin = self.key_comp(f16)

        # # B*C*T*H*W
        # #q16 = q16.view(b, *q16.shape[-3:]).contiguous()
        # q16 = q16.contiguous()

        # # B*T*C*H*W
        # #f16_thin = f16_thin.view(b, *f16_thin.shape[-3:])
        # #f16 = f16.view(b, *f16.shape[-3:])
        # f16 = f16
        # # f8 = f8.view(b, *f8.shape[-3:])
        # # f4 = f4.view(b, *f4.shape[-3:])

        return q16_tensor, f16_tensor

    def self_attention_op(self, k16, q16, v16):
        # self attention between key instance segmentation and query gaze heatmap
        # k16, q16, v16 shape: [b, c, h, w]
        similarity  = k16 @ q16
        scaled = similarity / math.sqrt(k16.shape[1])
        attention = F.softmax(scaled, dim=-1)
        out = attention @ v16
        return out


    def segment(self, qk16, qv16, mk16, mv16, selector=None): 
        # input shapes
        # qk16 [16, 12, 64, 38, 50]
        # qv16 [16, 12, 64, 38, 50]
        # mk16 [16, 12, 64, 38, 50]
        
        #affinity = self.memory.get_affinity(mk16, qk16)

        #attention = self.self_attention(qk16, mk16, qv16)
        attention_module = SelfAttention(qk16.shape[1])
        attention = attention_module(qk16, mk16, qv16) # [16, 64, 38, 50]

        logits = self.decoder(attention) #[16, 1, 600, 800] -- now with decoder change [16, 3, 600, 800]
        #prob = torch.sigmoid(logits) #[16, 1, 600, 800]
        
        # if self.single_object:
        #     logits = self.decoder(self.memory.readout(affinity, mv16, qv16), qf8, qf4)
        #     prob = torch.sigmoid(logits)
        # else:
        #     logits = torch.cat([
        #         self.decoder(self.memory.readout(affinity, mv16[:,0], qv16), qf8, qf4),
        #         self.decoder(self.memory.readout(affinity, mv16[:,1], qv16), qf8, qf4),
        #     ], 1)

        #     prob = torch.sigmoid(logits)
        #     prob = prob * selector.unsqueeze(2).unsqueeze(2)

        logits = self.aggregate(logits) #now [16, 3, 600, 800]
        #prob = F.softmax(logits, dim=1)[:, 1:]
        prob = F.softmax(logits, dim=1)

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


