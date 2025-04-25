import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from transformers import AutoModel

class MultiTaskModel(nn.Module):
    def __init__(self, input_feature, hidden_feature, clip_length, seq_length, p_dropout, num_stage):
        super(MultiTaskModel, self).__init__()
        self.clip_length = clip_length
        self.seq_length = seq_length
        
        self.video_cnn = GCN(input_feature, hidden_feature, clip_length, p_dropout, num_stage)
        self.text_model = AutoModel.from_pretrained("google/canine-c") # (bs, seq_len, 768)
        self.text_proj = nn.Linear(768, hidden_feature)
        
        self.video_clf = nn.Linear(2*hidden_feature, 3)
        self.text_clf = nn.Linear(2*hidden_feature, 2)
        self.video_span = nn.Linear(hidden_feature, 2)
        self.text_span = nn.Linear(hidden_feature, 2)

    def forward(self, video_data, text_data, attention_mask, task):
        y_video = self.video_cnn(video_data) # (batch, input_length, hidden_feature)
        y_text = self.text_model(text_data, attention_mask=attention_mask).last_hidden_state # (batch, seq_len, 768)
        y_text = self.text_proj(y_text)
        if 'clf' in task:
            y = torch.cat((torch.mean(y_video, dim=1), torch.mean(y_text, dim=1)), dim=1) # (batch, hidden_feature*2)
        else:
            y = torch.cat((y_text, y_video), dim=1) # (batch, seq_len+clip_len, hidden_feature)
        if task == 'video_clf':
            output = self.video_clf(y)
        elif task == 'text_clf':
            output = self.text_clf(y)
        elif task == 'video_span':
            output = self.video_span(y)[:, self.seq_length:, :]
        elif task == 'text_span':
            output = self.text_span(y)[:, :self.seq_length, :] 
        # print(f"{task} output shape: ", output.shape)
        return output

class GCN(nn.Module):
    # From https://github.com/dxli94/WLASL
    def __init__(self, input_feature, hidden_feature, input_length, p_dropout, num_stage=1, is_resi=True, nhead=2, num_layers=1):
        super(GCN, self).__init__()
        self.num_stage = num_stage
        self.input_length = input_length

        self.gc1 = GraphConvolution_att(input_feature, hidden_feature, input_length)
        self.bn1 = nn.BatchNorm1d(input_length * hidden_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, num_keypoints=input_length, p_dropout=p_dropout, is_resi=is_resi))

        self.gcbs = nn.ModuleList(self.gcbs)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()


    def forward(self, x):
        # x.shape: (batch, input_length, input_feature)
        y = self.gc1(x)        
        b, n, f = y.shape # (batch, input_length, hidden_feature)
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)
    
        return y # (batch, input_length, hidden_feature)

class GraphConvolution_att(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, num_keypoints, bias=True, init_A=0):
        super(GraphConvolution_att, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(num_keypoints, num_keypoints))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        # AHW
        support = torch.matmul(input, self.weight)  # HW
        output = torch.matmul(self.att, support)  # g
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GC_Block(nn.Module):

    def __init__(self, in_features, num_keypoints, p_dropout, bias=True, is_resi=True):
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        self.is_resi = is_resi

        self.gc1 = GraphConvolution_att(in_features, in_features, num_keypoints)
        self.bn1 = nn.BatchNorm1d(num_keypoints * in_features)

        self.gc2 = GraphConvolution_att(in_features, in_features, num_keypoints)
        self.bn2 = nn.BatchNorm1d(num_keypoints * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)
        if self.is_resi:
            return y + x
        else:
            return y

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


