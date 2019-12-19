import json
import ipdb, pdb
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import numpy as np
import math
from torch.nn import init
from torch.nn.utils import rnn
from util import Translator

from macnet_v2 import MACv2

DEBUG = False
if DEBUG:
    TR = Translator('./idx2word.json')

class SPModel(nn.Module):
    def __init__(self, config, word_mat, char_mat):
        super().__init__()
        self.config = config
        if config.bert: # Uses Bert
            if config.bert_with_glove: # Bert and Glove
                self.word_dim = config.bert_dim + config.glove_dim
                self.word_emb = nn.Embedding(len(word_mat), len(word_mat[0]), padding_idx=0)
                self.word_emb.weight.data.copy_(torch.from_numpy(word_mat))
                self.word_emb.weight.requires_grad = False
            else: # Bert only
                self.word_dim = config.bert_dim
        else: # Don't use Bert (Glove only)
            self.word_dim = config.glove_dim
            self.word_emb = nn.Embedding(len(word_mat), len(word_mat[0]), padding_idx=0)
            self.word_emb.weight.data.copy_(torch.from_numpy(word_mat))
            self.word_emb.weight.requires_grad = False
        self.char_emb = nn.Embedding(len(char_mat), len(char_mat[0]), padding_idx=0)
        self.char_emb.weight.data.copy_(torch.from_numpy(char_mat))

        self.end_mem = torch.nn.Parameter(torch.FloatTensor(config.hidden*2).cuda()) # A learnable end vector for support memory
        torch.nn.init.normal(self.end_mem)
        self.null_mem = torch.nn.Parameter(torch.FloatTensor(config.hidden*2).cuda(), requires_grad=False) # Not learned. We filter it out by requires_grad in the optimizer.
        torch.nn.init.constant(self.null_mem, 0)

        self.char_cnn = nn.Conv1d(config.char_dim, config.char_hidden, 5)
        self.char_hidden = config.char_hidden
        self.hidden = config.hidden

        self.rnn = EncoderRNN(config.char_hidden+self.word_dim, config.hidden, 1, True, True, 1-config.keep_prob, True)

        self.qc_att = BiAttention(config.hidden*2, 1-config.keep_prob)
        self.linear_1 = nn.Sequential(
                nn.Linear(config.hidden*8, config.hidden),
                nn.ReLU()
            )

        self.rnn_2 = EncoderRNN(config.hidden, config.hidden, 1, False, True, 1-config.keep_prob, False)

        self.att_layer1 = TransformerLayer1(config.hidden*2, config.hidden*2, config.keep_prob)
        self.att_layer2 = TransformerLayer1(config.hidden*2, config.hidden*2, config.keep_prob)
        self.att_layer3 = TransformerLayer1(config.hidden*2, config.hidden*2, config.keep_prob)
        self.att_layer4 = BiAttention(config.hidden*2, 1-config.keep_prob)
        self.linear_al4 = nn.Sequential(
                nn.Linear(config.hidden*8, config.hidden*2),
                nn.ReLU()
            )

        self.linear_custom = nn.Sequential(
                nn.Linear(config.hidden*2, config.hidden),
                nn.ReLU()
            )

        self.summ_ques = SummarySelfAttention(config.hidden*2, 1, config.keep_prob)

        if self.config.reasoning_module == 'QRN':
            self.qrn = QRN(config.hidden*2, config.hidden*2, self.config.reasoning_steps)
        if self.config.reasoning_module == 'QRNv2':
            self.qrn = QRNv2(config.hidden*2, config.hidden*2, self.config.reasoning_steps)
        if self.config.reasoning_module == 'RMNv2':
            self.rmn = RMNv2(config.hidden*2, config.hidden*2, self.config.reasoning_steps)
        if self.config.reasoning_module == 'MAC':
            self.mac = MAC(config.hidden*2, self.config.reasoning_steps, dropout=0.0)
        if self.config.reasoning_module == 'MACv2':
            self.mac = MACv2(config.hidden*2, config.hidden*2, self.config.reasoning_steps)

        self.rnn_sp = EncoderRNN(config.hidden, config.hidden, 1, False, True, 1-config.keep_prob, False)

        self.rnn_start = EncoderRNN(config.hidden*3, config.hidden, 1, False, True, 1-config.keep_prob, False)
        self.linear_start = nn.Linear(config.hidden*2, 1)

        self.rnn_end = EncoderRNN(config.hidden*3, config.hidden, 1, False, True, 1-config.keep_prob, False)
        self.linear_end = nn.Linear(config.hidden*2, 1)

        self.rnn_type = EncoderRNN(config.hidden*3, config.hidden, 1, False, True, 1-config.keep_prob, False)
        self.linear_type = nn.Linear(config.hidden*2, 3)

        self.cache_S = 0

    def get_output_mask(self, outer):
        S = outer.size(1)
        if S <= self.cache_S:
            return Variable(self.cache_mask[:S, :S], requires_grad=False)
        self.cache_S = S
        np_mask = np.tril(np.triu(np.ones((S, S)), 0), 15)
        self.cache_mask = outer.data.new(S, S).copy_(torch.from_numpy(np_mask))
        return Variable(self.cache_mask, requires_grad=False)

    def forward(self, context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, return_yp=False, support_labels=None, is_train=False, sp_labels_mod=None, bert=False, bert_context=None, bert_ques=None):
        para_size, ques_size, char_size, bsz = context_idxs.size(1), ques_idxs.size(1), context_char_idxs.size(2), context_idxs.size(0)

        context_mask = (context_idxs > 0).float()
        ques_mask = (ques_idxs > 0).float()

        context_ch = self.char_emb(context_char_idxs.contiguous().view(-1, char_size)).view(bsz * para_size, char_size, -1)
        ques_ch = self.char_emb(ques_char_idxs.contiguous().view(-1, char_size)).view(bsz * ques_size, char_size, -1)

        context_ch = self.char_cnn(context_ch.permute(0, 2, 1).contiguous()).max(dim=-1)[0].view(bsz, para_size, -1)
        ques_ch = self.char_cnn(ques_ch.permute(0, 2, 1).contiguous()).max(dim=-1)[0].view(bsz, ques_size, -1)

        if self.config.bert: # Uses Bert
            if self.config.bert_with_glove: # Bert and Glove
               glove_context_word = self.word_emb(context_idxs)
               glove_ques_word = self.word_emb(ques_idxs)

               context_word = torch.cat([glove_context_word, bert_context], dim=2)
               ques_word = torch.cat([glove_ques_word, bert_ques], dim=2)              
            else: # Bert only
                context_word = bert_context
                ques_word = bert_ques
        else: # Don't use Bert
            context_word = self.word_emb(context_idxs)
            ques_word = self.word_emb(ques_idxs)

        context_output = torch.cat([context_word, context_ch], dim=2)
        ques_output = torch.cat([ques_word, ques_ch], dim=2)

        context_output, context_hidden = self.rnn(context_output, context_lens) # Returns last hidden state too
        ques_output, ques_hidden = self.rnn(ques_output) # Returns last hidden state too

        output = self.qc_att(context_output, ques_output, ques_mask)
        output = self.linear_1(output)

        output_t = self.rnn_2(output, context_lens)
        output_t = self.att_layer1(output_t, context_mask)
        output_t = self.att_layer2(output_t, context_mask)
        output_t = self.att_layer3(output_t, context_mask)
        output = self.linear_custom(output_t)
        sp_output = self.rnn_sp(output, context_lens)

        start_output = torch.matmul(start_mapping.permute(0, 2, 1).contiguous(), sp_output[:,:,self.hidden:])
        end_output = torch.matmul(end_mapping.permute(0, 2, 1).contiguous(), sp_output[:,:,:self.hidden])
        sp_output = torch.cat([start_output, end_output], dim=-1)
        sp_output_with_end = torch.cat([self.null_mem.expand(sp_output.shape[0], 1, len(self.null_mem)),
                                        self.end_mem.expand(sp_output.shape[0], 1, len(self.end_mem)),
                                        sp_output], 1)

        # Reasoning Module
        ques_v = self.summ_ques(ques_output, ques_mask)
        sent_mask = torch.sum(all_mapping,1) > 0
        sent_mask = sent_mask.unsqueeze(-1).float()
        # A column of zeros then a column of ones is concatenated to the front because we add the null sentence and end sentence: [s1, s2, s3, empty, empty], mask: [1, 1, 1, 0, 0] -> [null, end, s1, s2, s3, empty, empty], mask: [0, 1, 1, 1, 1, 0, 0]
        sent_mask = torch.cat([torch.zeros_like(sent_mask)[:,0,:], torch.ones_like(sent_mask)[:,0,:], sent_mask], 1)
        sent_mask_neg = (1 - sent_mask ) * -10**10

        if self.config.reasoning_module == 'QRN':
            att_logits, reduced_query, att_softmaxes, reduced_queries = self.qrn(ques_v, sp_output_with_end, sent_mask, sent_mask_neg, sp_labels_mod, is_train, self.config.teacher_forcing, self.config.pointing)
        if self.config.reasoning_module == 'QRNv2':
            att_logits, reduced_query = self.qrn(ques_v, sp_output_with_end, sent_mask, sent_mask_neg, sp_labels_mod, is_train, self.config.teacher_forcing, self.config.pointing)
        if self.config.reasoning_module == 'RMNv2':
            att_logits, reduced_query = self.rmn(ques_v, sp_output_with_end, sent_mask, sent_mask_neg, sp_labels_mod, is_train, self.config.teacher_forcing, self.config.pointing)
        if self.config.reasoning_module == 'MAC':
            att_logits, reduced_query = self.mac(ques_output, ques_hidden, sp_output_with_end, sent_mask, sent_mask_neg, sp_labels_mod, is_train, self.config.teacher_forcing, self.config.pointing)
        if self.config.reasoning_module == 'MACv2':
            att_logits, output_h, output_m, att_softmaxes, att_softmaxes_post, grn_memories = self.mac(ques_output, ques_v, ques_mask, sp_output_with_end, sent_mask, sent_mask_neg, sp_labels_mod, is_train, self.config.teacher_forcing, self.config.pointing)

        sp_att4_logits = None
        if DEBUG:
            ipdb.set_trace()
            TR.register_batch(context_idxs, ques_idxs, all_mapping)
            TR.register_sp_att(att_softmaxes)
            ipdb.set_trace()
            TR.top_att(0, 1, 1)

        GRN_SP_PRED = True# temporary flag
        if GRN_SP_PRED:
            sp_att1 = att_logits[0].squeeze(-1)
            sp_att2 = att_logits[1].squeeze(-1)
            sp_att3 = att_logits[2].squeeze(-1)
            if self.config.reasoning_steps == 4:
                sp_att4 = att_logits[3].squeeze(-1)

            predict_support_grn = torch.stack([torch.zeros_like(sp_att1), torch.zeros_like(sp_att1)], -1).float() 
            sp_ids1 = sp_att1.topk(1, 1)[1].squeeze(-1).squeeze(-1).float()
            sp_ids2 = sp_att2.topk(1, 1)[1].squeeze(-1).squeeze(-1).float()
            sp_ids3 = sp_att3.topk(1, 1)[1].squeeze(-1).squeeze(-1).float()
            if self.config.reasoning_steps == 4:
                sp_ids4 = sp_att4.topk(1, 1)[1].squeeze(-1).squeeze(-1).float()
            for bid in range(len(sp_ids1)):
                predict_support_grn[bid, int(sp_ids1[bid].data.cpu()[0]), 1] = 1
                predict_support_grn[bid, int(sp_ids2[bid].data.cpu()[0]), 1] = 1
                predict_support_grn[bid, int(sp_ids3[bid].data.cpu()[0]), 1] = 1
                if self.config.reasoning_steps == 4:
                    predict_support_grn[bid, int(sp_ids4[bid].data.cpu()[0]), 1] = 1

            predict_support = predict_support_grn[:,2:,:]
            sp_output_t = predict_support[:,:,1]
            if is_train:
                coarse_att = torch.matmul(all_mapping, (sp_output_t>=0.5).unsqueeze(-1).float())
            else:
                coarse_att = torch.matmul(all_mapping, (sp_output_t>=0.5).unsqueeze(-1).float())

            sp_output = torch.matmul(all_mapping, sp_output)         
            sp_output = self.att_layer4(sp_output, context_output, coarse_att.squeeze(-1)) # New addition
            sp_output = self.linear_al4(sp_output)

        else:
            sp_output = reduced_query.unsqueeze(1) + sp_output # Add final reduced query to coattention encoding
            sp_output_t = self.linear_sp(sp_output)
            sp_output_aux = Variable(sp_output_t.data.new(sp_output_t.size(0), sp_output_t.size(1), 1).zero_())
            predict_support = torch.cat([sp_output_aux, sp_output_t], dim=-1).contiguous()

            if is_train:
                coarse_att = torch.matmul(all_mapping, (support_labels>0).unsqueeze(-1).float())
            else:
                coarse_att = torch.matmul(all_mapping, (sp_output_t>=0.5).float())

            sp_output = torch.matmul(all_mapping, sp_output)
            
            sp_output = self.att_layer4(sp_output, context_output, coarse_att.squeeze(-1)) # New addition
            sp_output = self.linear_al4(sp_output)

        output_start = torch.cat([output, sp_output], dim=-1)

        output_start = self.rnn_start(output_start, context_lens)
        logit1 = self.linear_start(output_start).squeeze(2) - 1e30 * (1 - context_mask)
        grn_logit1 = None
        output_end = torch.cat([output, output_start], dim=2)
        output_end = self.rnn_end(output_end, context_lens)
        logit2 = self.linear_end(output_end).squeeze(2) - 1e30 * (1 - context_mask)
        grn_logit2 = None

        output_type = torch.cat([output, output_end], dim=2)
        output_type = torch.max(self.rnn_type(output_type, context_lens), 1)[0]
        predict_type = self.linear_type(output_type)

        if not return_yp: return logit1, logit2, grn_logit1, grn_logit2, coarse_att, predict_type, predict_support, att_logits


        outer = logit1[:,:,None] + logit2[:,None]
        outer_mask = self.get_output_mask(outer)
        outer = outer - 1e30 * (1 - outer_mask[None].expand_as(outer))
        yp1 = outer.max(dim=2)[0].max(dim=1)[1]
        yp2 = outer.max(dim=1)[0].max(dim=1)[1]
        return logit1, logit2, grn_logit1, grn_logit2, coarse_att, predict_type, predict_support, yp1, yp2

class LockedDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        dropout = self.dropout
        if not self.training:
            return x
        m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m.div_(1 - dropout), requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x

class EncoderRNN(nn.Module):
    def __init__(self, input_size, num_units, nlayers, concat, bidir, dropout, return_last):
        super().__init__()
        self.rnns = []
        for i in range(nlayers):
            if i == 0:
                input_size_ = input_size
                output_size_ = num_units
            else:
                input_size_ = num_units if not bidir else num_units * 2
                output_size_ = num_units
            self.rnns.append(nn.GRU(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
        self.rnns = nn.ModuleList(self.rnns)
        self.init_hidden = nn.ParameterList([nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])
        self.dropout = LockedDropout(dropout)
        self.concat = concat
        self.nlayers = nlayers
        self.return_last = return_last

        # self.reset_parameters()

    def reset_parameters(self):
        for rnn in self.rnns:
            for name, p in rnn.named_parameters():
                if 'weight' in name:
                    p.data.normal_(std=0.1)
                else:
                    p.data.zero_()

    def get_init(self, bsz, i):
        return self.init_hidden[i].expand(-1, bsz, -1).contiguous()

    def forward(self, input, input_lengths=None):
        bsz, slen = input.size(0), input.size(1)
        output = input
        outputs = []
        hiddens = []
        if input_lengths is not None:
            lens = input_lengths.data.cpu().numpy()
        for i in range(self.nlayers):
            hidden = self.get_init(bsz, i)
            output = self.dropout(output)
            if input_lengths is not None:
                output = rnn.pack_padded_sequence(output, lens, batch_first=True)
            output, hidden = self.rnns[i](output, hidden)
            if input_lengths is not None:
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
                if output.size(1) < slen: # used for parallel
                    padding = Variable(output.data.new(1, 1, 1).zero_())
                    output = torch.cat([output, padding.expand(output.size(0), slen-output.size(1), output.size(2))], dim=1)
            if self.return_last:
                hiddens.append(hidden.permute(1, 0, 2).contiguous().view(bsz, -1))
            #else:
            outputs.append(output)
        if self.concat:
            if self.return_last:
                return torch.cat(outputs, dim=2), hiddens[-1] # TODO improve
            else:
                return torch.cat(outputs, dim=2)
        return outputs[-1]

class BiAttention(nn.Module):
    def __init__(self, input_size, dropout):
        super().__init__()
        self.dropout = LockedDropout(dropout)
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        self.memory_linear = nn.Linear(input_size, 1, bias=False)

        self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(1.0 / (input_size ** 0.5)))

    def forward(self, input, memory, mask):
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)
        input = self.dropout(input)
        memory = self.dropout(memory)

        input_dot = self.input_linear(input)
        memory_dot = self.memory_linear(memory).view(bsz, 1, memory_len)
        cross_dot = torch.bmm(input * self.dot_scale, memory.permute(0, 2, 1).contiguous())
        att = input_dot + memory_dot + cross_dot
        att = att - 1e30 * (1 - mask[:,None])

        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)
        weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len)
        output_two = torch.bmm(weight_two, input)

        return torch.cat([input, output_one, input*output_one, output_two*output_one], dim=-1)

class TransformerLayer1(nn.Module):
    def __init__(self, d_in, d_out, keep_prob):
        super(TransformerLayer1, self).__init__()
        # Transformer layer: Self-attention -> Feed forward -> Relu -> Residual
        self.attn = BiAttention(d_in, 1-keep_prob) # output dim will be d_in*4
        self.ff = nn.Sequential(nn.Linear(d_in*4, d_out),nn.ReLU())

    def forward(self, input, context_mask):
        output = self.attn(input, input, context_mask)
        output = self.ff(output)
        output = input + output
        return output

class GateLayer(nn.Module):
    def __init__(self, d_input, d_output):
        super(GateLayer, self).__init__()
        self.linear = nn.Linear(d_input, d_output)
        self.gate = nn.Linear(d_input, d_output)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        return self.linear(input) * self.sigmoid(self.gate(input))

class GateLayerTanh(nn.Module):
    def __init__(self, d_input, d_output):
        super(GateLayer, self).__init__()
        self.linear = nn.Linear(d_input, d_output)
        self.gate = nn.Linear(d_input, d_output)
        self.tanh = nn.Tanh()

    def forward(self, input):
        return self.linear(input) * self.tanh(self.gate(input))

class SummarySelfAttention(nn.Module):
    """Position-wise attention weighted summary
    """
    def __init__(self, d_input, d_output, dropout):
        super(SummarySelfAttention, self).__init__()
        self.linear1 = nn.Linear(d_input, d_input)
        self.linear2 = nn.Linear(d_input, 1)
        self.softmax = nn.functional.softmax
        self.tanh = nn.Tanh()

    def forward(self, input, mask):
        """Args
        input: batch x len x d_input
        """
        layer1 = self.tanh(self.linear1(input))
        layer2 = self.tanh(self.linear2(layer1))
        input = input * mask.unsqueeze(-1).expand(-1, -1, layer1.shape[-1])
        weighted_sum = torch.sum(self.softmax(layer2*mask.unsqueeze(-1), 1) * input, 1)
        return weighted_sum
