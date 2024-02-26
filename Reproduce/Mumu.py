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

class Keyless_atten(nn.Module):
    def __init__(self,feature_dim) -> None:
        super(Keyless_atten, self).__init__()
        self.atten_weight = nn.Linear(feature_dim,1,bias=False)
        initial_model_weight(layers=list(self.children()))
        self.norm=nn.Sequential(
            nn.BatchNorm1d(num_features=feature_dim),
            nn.ReLU(),
            nn.Dropout(0.4)
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
                 feature_dim=128,
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
            nn.Dropout(p=0.4))
        # self.fc8 = nn.Linear(256*2,num_class)

        # temporal feature
        self.lstm = nn.Sequential(
            nn.LSTM(input_size=feature_dim, hidden_size=feature_dim, num_layers=2,batch_first=True)
        )   
        self.lstm_after = nn.Sequential(
            # BNqd -- no use in shallow
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.atten = Keyless_atten(feature_dim)

        # initial weight
        initial_model_weight(layers = list(self.children()))
        print('weight initial finished!')

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
        out = self.lstm_after(out)
        out = self.atten(out)
        return out
class CNNLSTM_RGB_ENC(nn.Module):
    def __init__(self, pretrained=True,special_fd=128,temporal_fd = 128, stride_size = 3):
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
        self.lstm_after = nn.Sequential(
            # BNqd -- no use in shallow
            nn.ReLU(),
            nn.Dropout(0.2)
        )
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
        out = self.lstm_after(out)
        out = self.atten(out)

        return out

class Mumu_ENC(nn.Module):
    def __init__(self,mod_list):
        super().__init__()
        self.ENC=nn.ModuleDict()
        for mod in mod_list:
            if mod == "RGB":
                self.ENC[mod] = CNNLSTM_RGB_ENC(pretrained=True,special_fd=128,temporal_fd=128,)
            elif "IMU" in mod:
                self.ENC[mod] = HCN_Inertial_ENC(in_channel=3,out_channel=64,feature_dim=128)
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


class ATL_Module(nn.Module):
    def __init__(self,feature_dim, class_num) -> None:
        super(ATL_Module,self).__init__()
        self.SM_Fusion = Keyless_atten(feature_dim=feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim,64),
            nn.ReLU(),
            nn.Linear(64,class_num)
        )
        initial_model_weight(layers=list(self.children()))
    def forward(self, x):
        # x: Batch size, M, fd
        X_out = self.SM_Fusion(x)
        out = self.classifier(X_out)
        group_out = torch.softmax(out, dim = -1)
        return X_out, group_out


class TTL_Module(nn.Module):
    def __init__(self, class_num, modal_num, feature_dim=128,guide_dim=128,hidden_dim=128,dropout=0.4):
        super(TTL_Module,self).__init__()
        assert feature_dim == guide_dim
        self.modal_size = modal_num
        self.linears = clones(nn.Linear(feature_dim,hidden_dim),2)
        self.proj = nn.Linear(modal_num,1)
        self.guide_linear = nn.Linear(guide_dim,hidden_dim)
        self.norm = nn.Sequential(
            nn.BatchNorm1d(num_features=hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        self.classifier = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, class_num)
        )


    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        query = query.unsqueeze(-2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = scores.softmax(dim=-1).transpose(-2, -1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.mul(p_attn, value), p_attn

    def forward(self,x,x_guide):
        
        q = self.guide_linear(x_guide) # B, hidden_dim
        k,v = [
            lin(x_in) for lin, x_in in zip(self.linears, (x,x)) 
        ] # B, M, hidden_dim
        # x: B, M, hidden_dim
        x, self.attn = self.attention(q,k,v)
        # x: B, hidden_dim,
        x = self.proj(x.transpose(-2,-1)).squeeze(-1)
        x = self.norm(x)



        out = torch.cat([x,x_guide],dim=-1)
        out = self.classifier(out)
        return out

class Mumu(nn.Module):
    def __init__(self,mod_list,feature_dim=128,group_class_n = 3,class_n=37):
        super().__init__()
        self.UFE = Mumu_ENC(mod_list=mod_list)
        self.SM_Fusion = ATL_Module(feature_dim=feature_dim,class_num=group_class_n)
        self.GM_Fusion = TTL_Module(class_num=class_n,modal_num=len(mod_list),feature_dim=feature_dim,guide_dim=128,hidden_dim=128,dropout=0.4)
    def forward(self, x):
        x = self.UFE(x)
        X_out, group_out = self.SM_Fusion(x)
        class_out = self.GM_Fusion(x,X_out)
        return group_out, class_out

class Synthetic_dataset(data.Dataset):
    def __init__(self,length = 1000, modal_list = ["RGB","accIMU","gyroIMU"],RGB_length = 100, IMU_length = 200, group_num = 3, class_num = 17) -> None:
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
    model = Mumu(["RGB","accIMU","gyroIMU"])
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
        out_g, out_c = model(input)
        print(time.time()-end)
        