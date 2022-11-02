import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.block.strided_encoder import clones,LayerNorm,attention
import numpy as np
import math
import os
import copy

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h 
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1,model_sel=1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        if model_sel==1:
            print("using silu")
            self.gelu = nn.SiLU()
        else:
            self.gelu=nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.gelu(self.w_1(x))))

class Van_Transformer(nn.Module):
    def __init__(self, n_layers=3, d_model=256, d_ff=512, h=8, dropout=0.1, length=27,model_sel=1):
        super(Van_Transformer, self).__init__()

        self.pos_embedding = nn.Parameter(torch.randn(1, length, d_model))
        self.model = self.make_model(N=n_layers, d_model=d_model, d_ff=d_ff, h=h, dropout=dropout,model_sel=model_sel)
                
    def forward(self, x, mask=None):
        x += self.pos_embedding

        x = self.model(x, mask)

        return x

    def make_model(self, N=3, d_model=256, d_ff=512, h=8, dropout=0.1,model_sel=1):
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout,model_sel=model_sel)
        model = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
        return model






