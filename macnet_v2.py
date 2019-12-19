import copy
import json
import ipdb
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import numpy as np
import math
from torch.nn import init
from torch.nn.utils import rnn

class Controller(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim
        self.memory_down = nn.Linear(self.dim * 2, self.dim)
        self.update_gate = nn.Linear(self.dim, self.dim)
        self.hidden_state = nn.Linear(self.dim*2, self.dim)

    def forward(self, step, context, question, control, ques_mask, mem_2d):
        memory = mem_2d.view(mem_2d.size(0), -1)
        memory = self.memory_down(memory)
        gate = nn.functional.sigmoid(self.update_gate(question * memory))
        candidate = nn.functional.tanh(self.hidden_state(torch.cat([question, memory], -1)))
        next_control = gate * candidate + (1 - gate) * control

        return next_control

class WriteHead(nn.Module):
    def __init__(self, dim, self_attention=False, memory_gate=False):
        super().__init__()

        self.dim = dim
        self.update_gate = nn.Linear(self.dim, self.dim)
        self.hidden_state = nn.Linear(self.dim*1, self.dim)

        self.mha = MultiHeadedAttention(8, dim, dropout=None)

    def forward(self, retrieved, controls, prev_mem):
        mha_in = torch.cat([prev_mem, torch.stack([retrieved for _ in range(prev_mem.shape[1])], 1)], 2)
        mha_out = self.mha(prev_mem, mha_in, mha_in)

        control_stack = controls[-1].unsqueeze(1).expand(controls[-1].size()[0], prev_mem.size(1), controls[-1].size()[1])
        gate = F.sigmoid(self.update_gate(prev_mem * control_stack))
        next_mem = gate * prev_mem + (1 - gate) * self.hidden_state(mha_out)
        return next_mem


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model * 2, d_model)
        self.linear_v = nn.Linear(d_model * 2, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        
        self.attn = None
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        batchsize = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query = self.linear_q(query).view(batchsize, -1, self.h, self.d_k).transpose(1, 2)
        key = self.linear_k(key).view(batchsize, -1, self.h, self.d_k).transpose(1, 2)
        value = self.linear_v(value).view(batchsize, -1, self.h, self.d_k).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous().view(batchsize, -1, self.h * self.d_k)
        
        return self.linear_out(x)

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MACv2(nn.Module):
    def __init__(self, input_dim, hidden, max_steps):
        super(MACv2, self).__init__()
        self.input_dim = input_dim
        self.hidden = hidden
        self.max_steps = max_steps

        # Define Write Head & Controller
        self.write_head = WriteHead(self.input_dim, self_attention=True, memory_gate=True)
        self.controller = Controller(self.input_dim)

        # Initialize initial memory and control vectors
        self.mem_init = nn.Parameter(torch.zeros(1, self.hidden))
        self.control_init = nn.Parameter(torch.zeros(1, self.hidden))
        torch.nn.init.normal(self.mem_init)
        torch.nn.init.normal(self.control_init)
        self.mem_2d_init = torch.nn.init.normal(nn.Parameter(torch.zeros(2, self.hidden))) # control the 2d memory size


    def forward(self, ques_output, ques_hidden, ques_mask, sp_output_with_end, sent_mask, sent_mask_neg, sp_labels_mod, is_train, teacher_forcing, pointing):

        batchsize = ques_output.shape[0]
        control = self.control_init.expand(batchsize, self.hidden)
        memory = self.mem_init.expand(batchsize, self.hidden)
        mem_2d = self.mem_2d_init.expand(batchsize, self.mem_2d_init.size(0), self.mem_2d_init.size(1)).contiguous()
        controls = [control]
        reads = []

        att_logits = []
        argmax_ids = []
        att_softmaxes = []
        att_softmaxes_post = []
        for ii in range(self.max_steps):
            control = self.controller(ii, ques_output, ques_hidden, control, ques_mask, mem_2d)
            controls.append(control)

            sp_att_logits = torch.bmm(sp_output_with_end, control.unsqueeze(-1)) * sent_mask + sent_mask_neg
            att_logits.append(sp_att_logits)
            sp_onehot = torch.FloatTensor(batchsize, sp_att_logits.squeeze(-1).shape[-1])
            sp_onehot.zero_()
            sp_att_distrib = nn.functional.softmax(sp_att_logits, 1).squeeze(-1)
            att_softmaxes.append(sp_att_distrib)

            if is_train: # Training mode
                if teacher_forcing: # Feed labels as next inputs
                    sp_att_input = Variable(sp_onehot.cuda()).scatter(1, sp_labels_mod[:,ii].unsqueeze(1), 1) 
                else: # Next inputs are calculated
                    if pointing:
                        sp_att_input = Variable(sp_onehot.cuda()).scatter(1, torch.max(sp_att_distrib, 1)[1].unsqueeze(1), 1) # Pointing to argmax
                    else:
                        sp_att_input = sp_att_distrib
            else: # Inference mode
                if pointing:
                    sp_att_input = Variable(sp_onehot.cuda()).scatter(1, torch.max(sp_att_distrib, 1)[1].unsqueeze(1), 1) # Pointing to argmax
                else:
                    sp_att_input = sp_att_distrib

            argmax = torch.max(sp_att_input, 1)[1] == 1
            argmax_ids.append(argmax)

            att_softmaxes_post.append(sp_att_input)
            sp_att_input = sp_att_input.unsqueeze(-1)        
            read = torch.sum(sp_att_input*sp_output_with_end, 1)
            mem_2d = self.write_head(read, controls, mem_2d)

            reads.append(read)
 
        reduced_memories = None
        output_h, output_m = None, None
        return att_logits, output_h, output_m, att_softmaxes, att_softmaxes_post, reduced_memories


