import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision.models import resnet50
import numpy as np
import math
import copy
import random
import time
# initial way
# use the next 3 functions to initial a model, eg at the third function
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)

    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)

    elif classname.find('LSTM') != -1:
        for name, param in m.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.orthogonal_(param)

        # Initialize biases for LSTM’s forget gate to 1 to remember more by default. Similarly, initialize biases for GRU’s reset gate to -1.
        for names in m._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(m, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

    elif classname.find('GRU') != -1:
        for name, param in m.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.orthogonal(param)


def initial_model_weight(layers):
    for layer in layers:
        if list(layer.children()) == []:
            weights_init(layer)
            # print('weight initial finished!')
        else:
            for sub_layer in list(layer.children()):
                initial_model_weight([sub_layer])

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    # print('key.shape:',key.shape,'key.transpose(-2,-1).shape',key.transpose(-2,-1).shape)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # print("atention-score_shape",scores.shape,scores)
    if mask is not None:
        scores = scores.masked_fill(mask == 1, -1e9)
        
    p_attn = scores.softmax(dim=-1)
    # print(p_attn,"v",value)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # print(torch.matmul(p_attn, value))
    return torch.matmul(p_attn, value), p_attn

class Keyless_atten(nn.Module):
    def __init__(self,feature_dim,dp=0.2,batchnorm=True) -> None:
        super(Keyless_atten, self).__init__()
        self.atten_weight = nn.Linear(feature_dim,1,bias=False)
        initial_model_weight(layers=list(self.children()))
        if batchnorm:
            self.norm=nn.Sequential(
                nn.BatchNorm1d(num_features=feature_dim),
                nn.ReLU(),
                nn.Dropout(dp)
            )
        else:
            self.norm=nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dp)
            )
    def forward(self, x):
        # x_size: batch_size, segment_num, feature_dim
        alpha_weight = self.atten_weight(x)
        alpha_weight = torch.softmax(alpha_weight, dim = -2)
        out = torch.mul(x,alpha_weight)
        out = torch.sum(out, dim = -2)
        out = self.norm(out)
        return out


class HCN_Inertial_ENC(nn.Module):
    '''
    Input shape:
    Input shape should be (N, C, T, S)
    where N is the number of samples,
          C is the number of input channels,
          T is the length of the each segment,
          S is the number of segments.
    '''
    def __init__(self,
                 in_channel=3, # x y z axis
                 slide_size = 5,
                 window_size=5,
                 out_channel=64,
                 feature_dim=256,
                #  segment_num = 32,
                 ):
        super(HCN_Inertial_ENC, self).__init__()
        
        self.seg_size = window_size
        self.sli_size = slide_size
        self.out_channel = out_channel

        # point_level 1 x 1 x 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
        )
        # self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=window_size, kernel_size=(3,1), stride=1, padding=(1,0))

        # global_level 3 x 3 x 32
        self.conv2 = nn.Sequential(
            # 3 x 3 x (64 / 2)
            nn.Conv2d(in_channels=1, out_channels=out_channel//2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2)
            )

        # print(f"{window_size} {out_channel} {window_size//2} {out_channel //2}")
        self.fc3= nn.Sequential(
            nn.Linear((out_channel // 2)*(window_size//2)*out_channel//2,feature_dim), # 4*4 for window=64; 8*8 for window=128
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5))
        # self.fc8 = nn.Linear(256*2,num_class)

        # temporal feature
        self.lstm = nn.Sequential(
            nn.LSTM(input_size=feature_dim, hidden_size=feature_dim, num_layers=2,batch_first=True)
        )   
        # self.lstm_after = nn.Sequential(
        #     # BNqd -- no use in shallow
        #     nn.ReLU(),
        #     nn.Dropout(0.2)
        # )
        self.atten = Keyless_atten(feature_dim)

        # initial weight
        initial_model_weight(layers = list(self.children()))

    def forward(self, x):
        # x: batch size , total length, channel (3)
        # N, C, T, S = x.size()  # N0, C1, T2, S3
        # N C T V M
            # position
            # N0,C1,T2,V3 point-level
        B = x.size(0)
        
        # sample_len = x.size(1)
        # slices = [range(i,i+self.seg_size) if i+5 <= sample_len else range(i,sample_len) for i in range(0,sample_len,self.sli_size)]
        # if len(slices[-1]) != self.seg_size:
        #     slices.pop()
        # seg_num = len(slices)
        # # x: batch size, 3, seg_num, window_length
        # x = x[:,slices,:].permute(0,3,1,2).contiguous()
        x = x.permute(0,3,1,2).contiguous()
        seg_num = x.size(2)
        # self.conv2[0] = nn.Conv2d(in_channels=seg_num, out_channels=self.out_channel//2 * seg_num, kernel_size=3, stride=1, padding=1,groups=seg_num)

        out = self.conv1(x)

        # x: batch size, seg_num, 1, window_length, 64
        out = out.permute(0,2,3,1).contiguous().view(B*seg_num, self.seg_size, -1).unsqueeze(1)

        out = self.conv2(out)

        # N0,S1,T2,C3, global level
        out = out.view(B,seg_num, -1)


        out = self.fc3(out)
        

        t = out
        assert not ((t != t).any())# find out nan in tensor
        assert not (t.abs().sum() == 0) # find out 0 tensor

        out,_ = self.lstm(out)
        # out = self.lstm_after(out)
        out = self.atten(out)
        return out
class CNNLSTM_RGB_ENC(nn.Module):
    def __init__(self, pretrained=True,special_fd=256,temporal_fd = 256, stride_size = 3):
        super(CNNLSTM_RGB_ENC, self).__init__()
        self.stride_size = stride_size
        self.resnet = resnet50(pretrained=pretrained)
        self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, special_fd))
        self.resnet_after = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.lstm = nn.Sequential(
            nn.LSTM(input_size=special_fd, hidden_size=temporal_fd, num_layers=2,batch_first=True)
        )
        # self.lstm_after = nn.Sequential(
        #     # BNqd -- no use in shallow
        #     nn.ReLU(),
        #     nn.Dropout(0.2)
        # )
        self.atten = Keyless_atten(temporal_fd)
        initial_model_weight(layers=list(self.children()))
        # self.fc1 = nn.Linear(256, 128)
        # self.fc2 = nn.Linear(128, num_classes)       
    def forward(self, x_3d):
        # x: batch, total_num, 3, height, width
        # sample_len = x_3d.size(1)
        # slices = [i for i in range(0,sample_len,self.stride_size)]
        # # x: batch, seg_num, 3, height, width
        # x_3d = x_3d[:,slices,...]
        B, N, C, H, W=x_3d.size()
        with torch.no_grad():
            x = self.resnet(x_3d.view(B*N,C,H,W)).view(B,N,-1)
        out,_ = self.lstm(x)
        # out = self.lstm_after(out)
        out = self.atten(out)

        return out

class UMF_ENC(nn.Module):
    def __init__(self,mod_list):
        super().__init__()
        self.ENC=nn.ModuleDict()
        for mod in mod_list:
            if mod == "RGB":
                self.ENC[mod] = CNNLSTM_RGB_ENC(pretrained=True,special_fd=256,temporal_fd=256,)
            elif "IMU" in mod:
                self.ENC[mod] = HCN_Inertial_ENC(in_channel=3,out_channel=64,feature_dim=256)
            else:
                raise NotImplementedError
    def forward(self,x):
        uni_features=[]
        assert len(x) == len(self.ENC)
        for item in x.items():
            item_out = self.ENC[item[0]](item[1])
            item_out = item_out.unsqueeze(-2) # B, 1 ,fd
            uni_features.append(item_out)
        uni_features = torch.cat(uni_features,dim=-2) # B, M, fd
        return uni_features

class MultiHeadedAttention(nn.Module):
    def __init__(self, h=1, d_model=256, dropout=0.1,div_two=False):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.div_two = div_two

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(0)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        if self.div_two:
            x = x/2
        x = self.linears[-1](x)
        del query
        del key
        del value
        # B Seg_n dim
        return x


class Expert(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu2 = nn.ReLU()
        

    def forward(self, x):
        # B, fd
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        # B, fd
        return self.relu2(out)

class MMoE_Gate(nn.Module):
    def __init__(self,head=2,feature_dim = 256):
        super(MMoE_Gate,self).__init__()
        self.h = head
        self.d_model = head*feature_dim
        self.multimodal_context = Keyless_atten(feature_dim)
        self.MoE = MultiHeadedAttention(1,256)
    def forward(self,E_a,X_a):
        # E_a (B, expert_num, fd), X_a : B M fd
        E_c = self.multimodal_context(X_a)
        out = self.MoE(E_c.unsqueeze(-2),E_a,E_a)
        # B 1 fd
        return out

class Self_MoEAT(nn.Module):
    def __init__(self,modal_num,expert_num=2,input_size=256,output_size=256,hidden_size=512,head_intra=1):
        super(Self_MoEAT, self).__init__()
        self.intra_head = head_intra
        self.num_experts = modal_num * expert_num
        self.expert_num = expert_num
        self.modal_num = modal_num
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.experts = nn.ModuleList([Expert(self.input_size, self.output_size, self.hidden_size) for i in range(self.num_experts)])
        # intramodal
        self.intraModal_atten =  nn.ModuleList([MultiHeadedAttention(h = self.intra_head, d_model= output_size*self.intra_head) for i in range(self.modal_num)])


    def forward(self, x):
        # B, M, feature_dim
        x = torch.chunk(x,x.size(1),1) # tuple: (B, 1, fd) * M
        out_list = [] # [(B, expert_num, fd)] * M 
        for i in range(len(x)):
            # x[i] = x[i].squeeze(1)
            # for j in range(self.expert_num)
            # print(x[i].shape)

            uni_list = [self.experts[self.expert_num*i + j](x[i].squeeze(1)) for j in range(self.expert_num)]
            # uni_list = torch.cat(uni_list,dim=1)
            uni_list = torch.stack(uni_list, dim = 1)
            out_list.append(uni_list)
        

        # below: [(B, expert_num, fd)] * M : out_list

        # out_list: [(B, expert_num, fd)] * M
        for i in range(len(out_list)):
            out_list[i] = self.intraModal_atten[i](out_list[i],out_list[i],out_list[i])

        # 

        return out_list

class Multi_MoE(nn.Module):
    def __init__(self,modal_num,expert_num=2,input_size=256,output_size=256,hidden_size=512,head_intra=1):
        super(Multi_MoE, self).__init__()
        self.self_moeat = Self_MoEAT(modal_num,expert_num,input_size,output_size,hidden_size,head_intra)
        # self.multimodal_context = Keyless_atten(input_size,0.5,True)
        self.mmoe_gate = nn.ModuleList([MMoE_Gate(expert_num, feature_dim=input_size) for i in range(modal_num)])

        

    def forward(self,x):
        # x: B M fd

        # out: [B exp_num fd] * M
        out_moeat = self.self_moeat(x) 
        # E_c: B fd
        # E_c = self.multimodal_context(x)
        
        for i in range(len(out_moeat)):
            out_moeat[i] = self.mmoe_gate[i](out_moeat[i],x)

        # [B 1 fd] * M
        
       
        
        out_moeat = torch.cat(out_moeat,dim=-2) # [B,M,fd]
        # E^u_m: [B,M,fd]
        x = out_moeat+x
        return x

class Cross_GAT(nn.Module):
    def __init__(self,modal_num,num_class,feature_dim=256,dropout=0.5):
        super(Cross_GAT,self).__init__()
        self.linears = clones(nn.Linear(feature_dim, feature_dim), 4 * modal_num)
        
        self.gat = MultiHeadedAttention(1,feature_dim,div_two=True)
        self.norm = nn.Sequential(
                nn.BatchNorm1d(num_features=modal_num),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        self.proj = nn.Sequential(
                nn.Linear(modal_num*feature_dim,feature_dim),
                nn.BatchNorm1d(num_features=feature_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(feature_dim,num_class)
    def forward(self, X): # [B,M,fd]
        print(len(self.linears))
        querys=[]
        keys=[]
        values=[]
        msgs = []
        x = torch.chunk(X,X.size(1),1) # # [B,1,fd] * M
        for i, x_m in  enumerate(x):
            querys.append(self.linears[0*i](x_m)) 
            keys.append(self.linears[1*i](x_m))
            values.append(self.linears[2*i](x_m))
        query = torch.cat(querys,dim=1)
        key = torch.cat(keys,dim=1)
        value = torch.cat(values,dim=1)
        atten_out,_ = attention(
            query, key, value, mask=torch.eye(query.size(1))
        )
        atten_out = atten_out / 2
        atten_out = torch.chunk(atten_out,atten_out.size(1),1) 
        for i, atten_m in enumerate(atten_out):
            msgs.append(self.linears[3*i](atten_m))
        msg = torch.cat(msgs,dim=1)
        out = msg+X
        out = self.norm(out)
        
        out = self.proj(out.view(msg.size(0), -1))

        return self.classifier(out)
        
class Multi_GAT(nn.Module):
    def __init__(self,mod_list,num_class,expert_num = 2, feature_dim = 256):
        super().__init__()
        self.UFE = UMF_ENC(mod_list)
        self.multi_moe = Multi_MoE(len(mod_list),expert_num,feature_dim,feature_dim,2*feature_dim)
        self.cross_gat = Cross_GAT(len(mod_list), num_class)
        initial_model_weight(layers=list(self.children()))
    def forward(self,x):
        x = self.UFE(x)
        x = self.multi_moe(x)
        x = self.cross_gat(x)
        return x


class Synthetic_dataset(data.Dataset):
    def __init__(self,length = 1000, modal_list = ["RGB","accIMU","gyroIMU"],RGB_length = 100, IMU_length = 200, group_num = 3, class_num = 37) -> None:
        super(Synthetic_dataset,self).__init__()
        self.len = length
        self.modal_list = modal_list
        self.RGB_length = RGB_length
        self.IMU_length = IMU_length
        self.group_num = group_num
        self.class_num = class_num
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        out = {}
        for i in self.modal_list:
            if "RGB" in i:
                out[i]=torch.rand(self.RGB_length,3,224,224)
            elif "IMU" in i:
                out[i]=torch.rand(self.RGB_length,3)
            else:
                raise NotImplementedError
        return out, random.randint(0, self.group_num-1), random.randint(0, self.class_num-1)

if __name__ == '__main__':

    syn_dataset = Synthetic_dataset()
    Synthetic_dataloader = data.DataLoader(syn_dataset, batch_size=1, shuffle=False,drop_last=True)

    slide_size = 5
    window_size = 5
    stride_size = 3
    model = Multi_GAT(["RGB","accIMU","gyroIMU"],37)
    model.eval()
    for i, (input, coarse_target, fine_target) in enumerate(Synthetic_dataloader):

        for um in input.items():
            
            if "IMU" in um[0]:
                
                B = um[1].size(0)
                sample_len = um[1].size(1)
                slices = [range(i,i+window_size) if i+5 <= sample_len else range(i,sample_len) for i in range(0,sample_len,slide_size)]
                if len(slices[-1]) != window_size:
                    slices.pop()
                seg_num = len(slices)
                # x: batch size, 3, seg_num, window_length
                
                input[um[0]] = um[1][:,slices,:]
                
            if "RGB" in um[0]:
                
                B = um[1].size(0)
                sample_len = um[1].size(1)
                slices = [i for i in range(0,sample_len,stride_size)]
                # x: batch, seg_num, 3, height, width
                
                input[um[0]] = um[1][:,slices,...]
                
        end = time.time()
        out_p = model(input)
        print(time.time()-end)