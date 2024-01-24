import torch
from torch import nn
import torch.nn as nn
from torch.nn import functional as F
import math
import einops
from einops import rearrange


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features   
        self.out_features = out_features   
        self.dropout = dropout   
        self.alpha = alpha    
        self.concat = concat   

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))  
        nn.init.xavier_uniform_(self.W.data, gain=1.414) 
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, inp):
        h = torch.matmul(inp, self.W)   
        batch_size = h.size()[0]
        N = h.size()[1]   

        a_input = torch.cat([h.repeat(1,1, N).view(batch_size,N*N, -1), h.repeat(1,N, 1)], dim=1).view(batch_size,N, -1, 2*self.out_features)
        attention = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        attention = F.softmax(attention, dim=1)   
        attention = F.dropout(attention, self.dropout, training=self.training)   
        h_prime = torch.matmul(attention, h) 
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime 
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
 

class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, dropout, alpha, n_heads):
        super(GAT, self).__init__()
        self.dropout = dropout 
        
        self.attentions = [GraphAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention) 
        self.out_att = GraphAttentionLayer(n_hid * n_heads, n_class, dropout=dropout,alpha=alpha, concat=False)
    
    def forward(self, x):

        x = F.dropout(x, self.dropout, training=self.training)   
        x = torch.cat([att(x) for att in self.attentions], dim=-1)  
        x = F.dropout(x, self.dropout, training=self.training)   
        x = F.elu(self.out_att(x))   
        return F.log_softmax(x, dim=1) 
    
class KL_atten(nn.Module):
    def __init__(self, in_features):
        super(KL_atten, self).__init__()
        self.W = nn.Parameter(torch.zeros(size=(in_features, 1))) 
        self.scale_factor = in_features ** -0.5
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  
    def forward(self, x):
        attention = torch.matmul(x, self.W).squeeze(-1) 
        attention = F.softmax(attention, dim=1).unsqueeze(-1)
        h = x * attention
        h = torch.sum(h,dim=-2)
        return h 
    
class Mul_atten(nn.Module):
    def __init__(self, in_features, hidden_dim=256):
        super(Mul_atten, self).__init__()
        self.W_k_sensor = nn.Parameter(torch.zeros(size=(in_features, hidden_dim))) 
        self.W_q_video = nn.Parameter(torch.zeros(size=(in_features, hidden_dim))) 
        self.W_v_video = nn.Parameter(torch.zeros(size=(in_features, hidden_dim))) 
        nn.init.xavier_uniform_(self.W_k_sensor.data, gain=1.414)  
        nn.init.xavier_uniform_(self.W_q_video.data, gain=1.414)  
        nn.init.xavier_uniform_(self.W_v_video.data, gain=1.414)  
    def forward(self, sensor_base, video_base, dropout=None):
        sensor_base = sensor_base.unsqueeze(1)
        sensor_K = torch.matmul(sensor_base, self.W_k_sensor)
        video_Q = torch.matmul(video_base, self.W_q_video)
        video_V = torch.matmul(video_base, self.W_v_video)
        d_k = video_Q.size(-1)
        scores = torch.matmul(video_Q, sensor_K.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        scores = torch.matmul(sensor_K, video_Q.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        p_attn = F.softmax(scores, dim = -1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        z = p_attn.transpose(-2, -1)*video_V
        return z 


class Fastformer(nn.Module):
    def __init__(self, dim = 3, decode_dim = 16):
        super(Fastformer, self).__init__()
        self.to_qkv = nn.Linear(dim, decode_dim * 3, bias = False)
        self.weight_q = nn.Linear(dim, decode_dim, bias = False)
        self.weight_k = nn.Linear(dim, decode_dim, bias = False)
        self.weight_v = nn.Linear(dim, decode_dim, bias = False)
        self.weight_r = nn.Linear(decode_dim, decode_dim, bias = False)
        self.weight_alpha = nn.Parameter(torch.randn(decode_dim))
        self.weight_beta = nn.Parameter(torch.randn(decode_dim))
        self.scale_factor = decode_dim ** -0.5

    def forward(self, x, mask = None):
        query = self.weight_q(x)
        key = self.weight_k(x)
        value = self.weight_v(x)
        b, n, d = query.shape

        mask_value = -torch.finfo(x.dtype).max
        mask = rearrange(mask, 'b n -> b () n')

        # Caculate the global query
        alpha_weight = (torch.mul(query, self.weight_alpha) * self.scale_factor).masked_fill(~mask, mask_value)
        alpha_weight = torch.softmax(alpha_weight, dim = -1)
        global_query = query * alpha_weight
        global_query = torch.einsum('b n d -> b d', global_query)

        # Model the interaction between global query vector and the key vector
        repeat_global_query = einops.repeat(global_query, 'b d -> b copy d', copy = n)
        p = repeat_global_query * key
        beta_weight = (torch.mul(p, self.weight_beta) * self.scale_factor).masked_fill(~mask, mask_value)
        beta_weight = torch.softmax(beta_weight, dim = -1)
        global_key = p * beta_weight
        global_key = torch.einsum('b n d -> b d', global_key)

        # key-value
        key_value_interaction = torch.einsum('b j, b n j -> b n j', global_key, value)
        key_value_interaction_out = self.weight_r(key_value_interaction)
        result = key_value_interaction_out + query
        return result
