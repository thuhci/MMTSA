import torch
from torch import nn
from torch.nn.init import normal_, constant_
from attention_fusion import GraphAttentionLayer, GAT, KL_atten, Mul_atten
from ops.basic_ops import ConsensusModule
import copy


class Fusion_Classification_Network(nn.Module):

    def __init__(self, feature_dim, modality, midfusion, num_class,
                 consensus_type, before_softmax, dropout, num_segments):
        super().__init__()
        self.num_class = num_class
        self.modality = modality
        self.midfusion = midfusion
        self.reshape = True
        self.consensus = ConsensusModule(consensus_type)
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.num_segments = num_segments

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        if len(self.modality) > 1:  # Fusion

            if self.midfusion == 'concat':
                self._add_sensorvision_fc_layer(len(self.modality) * feature_dim, 512)
                self._add_classification_layer(512)
            elif self.midfusion == 'concat_atten':
                self.atten = KL_atten(len(self.modality) * feature_dim)
                self._add_sensorvision_fc_layer(len(self.modality) * feature_dim, 512)
                self._add_classification_layer(512)
            elif self.midfusion == 'double_atten':
                self.intra_segmentAtt = KL_atten(feature_dim)
                self.intra_modalAtt = nn.ModuleList([copy.deepcopy(KL_atten(feature_dim)) for _ in range(self.num_segments)])
                self._add_sensorvision_fc_layer(feature_dim, 512)
                self._add_classification_layer(512)
            elif self.midfusion == 'final_atten':
                self.intra_segmentAtt = nn.ModuleList([copy.deepcopy(KL_atten(feature_dim)) for _ in range(len(self.modality))])
                self.intra_modalAtt = GraphAttentionLayer(feature_dim,256,0.5,0.5)
                self._add_sensorvision_fc_layer(len(self.modality) * 256, 256)
                self._add_classification_layer(256)
            elif self.midfusion == 'mul_atten':
                self.sen_atten = KL_atten(feature_dim)
                self.mul_atten = Mul_atten(feature_dim)
                if self.dropout > 0:
                    self.dropout_layer = nn.Dropout(p=self.dropout)
                self._add_classification_layer(256)
            elif self.midfusion == 'graph_attention':
                self.attention_layer = GraphAttentionLayer(feature_dim,256,0.2,0.2)
                if self.dropout > 0:
                    self.dropout_layer = nn.Dropout(p=self.dropout)
                self._add_classification_layer(256 * len(self.modality))
            elif self.midfusion == 'gat':
                self.gat = GAT(feature_dim,256,256,0.2,0.2,8)
                if self.dropout > 0:
                    self.dropout_layer = nn.Dropout(p=self.dropout)
                self._add_classification_layer(256 * len(self.modality))

        else:  # Single modality
            if self.dropout > 0:
                self.dropout_layer = nn.Dropout(p=self.dropout)

            self._add_classification_layer(feature_dim)

    def _add_classification_layer(self, input_dim):

        std = 0.001
        self.fc_action = nn.Linear(input_dim, self.num_class)
        normal_(self.fc_action.weight, 0, std)
        constant_(self.fc_action.bias, 0)
    def _add_sensorvision_fc_layer(self, input_dim, output_dim):

        self.fc1 = nn.Linear(input_dim, output_dim)
        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(p=self.dropout)

        std = 0.001
        normal_(self.fc1.weight, 0, std)
        constant_(self.fc1.bias, 0)
        self.relu = nn.ReLU()


    def forward(self, inputs):
        # inputs: list , inputs[i] = [32 * segment_num, feature_dim]
        if len(self.modality) > 1:  # Fusion
            if self.midfusion == 'concat_atten':
                base_out = torch.cat(inputs, dim=1) # [24, 2048] or [32 * segment_num, 1024 * len_modal]
                base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:]) # [batch_size, segment_num, 1024 * len_modal]
                base_out = self.atten(base_out) # [batch_size, 1024 * len_modal]
                base_out = self.fc1(base_out)
                base_out = self.relu(base_out)
            elif self.midfusion == 'double_atten':
                intra_modalOut = []
                for i in range(len(self.modality)):
                    inputs[i] = inputs[i].unsqueeze(1)
                modal_out = torch.cat(inputs, dim=1) # torch.Size([batch_size x seg_num, modal_num, 1024])
                modal_out = modal_out.view((-1,self.num_segments)+modal_out.size()[1:])# torch.Size([batch_size , seg_num, modal_num, 1024])
                
                
                modal_out = torch.transpose(modal_out,1,0) # torch.Size([seg_num, batch_size, modal_num, 1024])
                for i, modal_per_seg in enumerate(modal_out):
                    modal_seg_out = self.intra_modalAtt[i](modal_per_seg) # [batch_size, 1024]
                    intra_modalOut.append(modal_seg_out)
                for i in range(self.num_segments):
                    intra_modalOut[i] = intra_modalOut[i].unsqueeze(1)
                seg_out = torch.cat(intra_modalOut, dim=1) # torch.Size([batch_size, seg_num, feature_dim = 1024])
                base_out = self.intra_segmentAtt(seg_out) # torch.Size([batch_size, feature_dim = 1024])
                base_out = self.fc1(base_out)
                base_out = self.relu(base_out)
                
            elif self.midfusion == 'final_atten': # 层次
                intra_segmentOut = []
                for i, input_modal in enumerate(inputs):
                    # input_modal: [32 * segment_num, 1024]
                    segment_out = input_modal.view((-1, self.num_segments) + input_modal.size()[1:]) # [32 , segment_num, feature_dim = 1024]
                    segment_out = self.intra_segmentAtt[i](segment_out) # [32, feature_dim = 1024]
                    intra_segmentOut.append(segment_out)
                # intra_modalOut : [] : len_modal x [32, feature_dim = 1024]
#                 segment_out = torch.cat(intra_modalOut, dim=1) # [32, feature_dim = 1024]
                for i in range(len(self.modality)):
                    intra_segmentOut[i] = intra_segmentOut[i].unsqueeze(1)
                modal_out = torch.cat(intra_segmentOut, dim=1) # torch.Size([batch_size, len_modal, feature_dim = 1024])
                modal_out = self.intra_modalAtt(modal_out) # torch.Size([batch_size, len_modal, out_feature_dim = 512])
                base_out = modal_out.view(modal_out.shape[0],-1) # torch.Size([24, 512 * modal_num])
                base_out = self.fc1(base_out)
                base_out = self.relu(base_out)
            elif self.midfusion == 'concat':
#                 with torch.no_grad():
#                     print(torch.mean (torch.cosine_similarity(inputs[0], inputs[1], dim=1)))
                base_out = torch.cat(inputs, dim=1) # [24, 2048]            
                base_out = self.fc1(base_out)
                base_out = self.relu(base_out)
#                 print(base_out.shape)
            elif self.midfusion == 'mul_atten':
                # inputs[0] rgb, inputs[1] sensor, 
                sensor_base_out = inputs[1].view((-1, self.num_segments) + inputs[1].size()[1:])
                rgb_base_out = inputs[0].view((-1, self.num_segments) + inputs[0].size()[1:])
                sensor_base_out = self.sen_atten(sensor_base_out) # [8, 1024]
                base_out = self.mul_atten(sensor_base_out, rgb_base_out) # [8, 3, 256]

            elif self.midfusion == 'attention':
                # input_i: [32 * segment_num, feature_dim] -- modal_i
                for i in range(len(self.modality)):
                    inputs[i] = inputs[i].unsqueeze(1) # input_i: [32 * segment_num, 1 ,feature_dim] -- modal_i
                base_out = torch.cat(inputs, dim=1) # torch.Size([32 * segment_num, modal_num ,feature_dim])
                base_out = self.attention_layer(base_out) # torch.Size([32 * segment_num, modal_num, 512])
                base_out = base_out.reshape(base_out.shape[0],-1) #  torch.Size([24, 512 * modal_num])
            elif self.midfusion == 'gat':
                # multi head
                for i in range(len(self.modality)):
                    inputs[i] = inputs[i].unsqueeze(1)
                base_out = torch.cat(inputs, dim=1) # torch.Size([24, 2, 1024])
                base_out = self.gat( base_out )
                base_out = torch.cat([base_out[:,i] for i in range(base_out.shape[1])],dim=1)
#                 print(base_out.shape)
                
        else:  # Single modality
            base_out = inputs[0]

        if self.dropout > 0 :
            base_out = self.dropout_layer(base_out)

        if self.midfusion != 'concat_atten' and self.midfusion != 'mul_atten' and self.midfusion != 'final_atten' and self.midfusion != 'double_atten':
            base_out = self.fc_action(base_out)
            if not self.before_softmax:
                base_out = self.softmax(base_out)
            if self.reshape:
                base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
            output = self.consensus(base_out)
            output = output.squeeze(1)
        elif self.midfusion == 'concat_atten' or self.midfusion == 'final_atten' or self.midfusion == 'double_atten':
            output = self.fc_action(base_out)
            if not self.before_softmax:
                output = self.softmax(output)
        elif self.midfusion == 'mul_atten':
            base_out = self.fc_action(base_out) # [8,3,20]
            if not self.before_softmax:
                base_out = self.softmax(base_out)
            output = self.consensus(base_out)
            output = output.squeeze(1)
        return output
