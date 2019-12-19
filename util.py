import h5py
import ipdb
import torch
import numpy as np
import re
from collections import Counter
import string
import pickle
import random
from torch.autograd import Variable
import copy
import ujson as json
import traceback

IGNORE_INDEX = -100

RE_D = re.compile('\d')
def has_digit(string):
    return RE_D.search(string)

def prepro(token):
    return token if not has_digit(token) else 'N'

class DataIterator(object):
    def __init__(self, buckets, bsz, para_limit, ques_limit, char_limit, shuffle, sent_limit, bert=False, bert_buckets=None, new_spans=False, ques_type=None):
        self.buckets = buckets
        self.bert = bert
        self.bert_buckets = bert_buckets
        self.update_span=False
        self.ques_type = ques_type
        if new_spans:
            self.update_span = True
            self.new_spans = json.load(open('train_new_spans_v2.json','r')) # Manually set the new spans
        if self.ques_type:
            self.type_and_id = json.load(open(self.ques_type,'r'))
            self.type_and_id = {val:key for key, val in self.type_and_id}

        self.bsz = bsz
        if para_limit is not None and ques_limit is not None:
            self.para_limit = para_limit
            self.ques_limit = ques_limit
        else:
            para_limit, ques_limit = 0, 0
            for bucket in buckets:
                for dp in bucket:
                    para_limit = max(para_limit, dp['context_idxs'].size(0))
                    ques_limit = max(ques_limit, dp['ques_idxs'].size(0))
            self.para_limit, self.ques_limit = para_limit, ques_limit
        self.char_limit = char_limit
        self.sent_limit = sent_limit

        self.num_buckets = len(self.buckets)
        self.bkt_pool = [i for i in range(self.num_buckets) if len(self.buckets[i]) > 0]
        if shuffle:
            for i in range(self.num_buckets):
                random.shuffle(self.buckets[i])
        self.bkt_ptrs = [0 for i in range(self.num_buckets)]
        self.shuffle = shuffle

    def __iter__(self):
        context_idxs = torch.LongTensor(self.bsz, self.para_limit).cuda()
        ques_idxs = torch.LongTensor(self.bsz, self.ques_limit).cuda()
        context_char_idxs = torch.LongTensor(self.bsz, self.para_limit, self.char_limit).cuda()
        ques_char_idxs = torch.LongTensor(self.bsz, self.ques_limit, self.char_limit).cuda()
        y1 = torch.LongTensor(self.bsz).cuda()
        y2 = torch.LongTensor(self.bsz).cuda()
        q_type = torch.LongTensor(self.bsz).cuda()
        if self.ques_type:
            ques_type = torch.LongTensor(self.bsz).cuda()
        start_mapping = torch.Tensor(self.bsz, self.para_limit, self.sent_limit).cuda()
        end_mapping = torch.Tensor(self.bsz, self.para_limit, self.sent_limit).cuda()
        all_mapping = torch.Tensor(self.bsz, self.para_limit, self.sent_limit).cuda()
        is_support = torch.LongTensor(self.bsz, self.sent_limit).cuda()

        is_y1 = torch.LongTensor(self.bsz, self.para_limit).cuda()
        is_y2 = torch.LongTensor(self.bsz, self.para_limit).cuda()

        while True:
            if self.bert:
                bert_context = torch.FloatTensor(self.bsz, self.para_limit, 768).zero_().cuda()
                bert_ques = torch.FloatTensor(self.bsz, self.ques_limit, 768).zero_().cuda()

            if len(self.bkt_pool) == 0: break
            bkt_id = random.choice(self.bkt_pool) if self.shuffle else self.bkt_pool[0]
            start_id = self.bkt_ptrs[bkt_id]
            cur_bucket = self.buckets[bkt_id]
            cur_bsz = min(self.bsz, len(cur_bucket) - start_id)

            ids = []
            gold_mention_spans = []

            cur_batch = cur_bucket[start_id: start_id + cur_bsz]
            cur_batch.sort(key=lambda x: (x['context_idxs'] > 0).long().sum(), reverse=True)

            max_sent_cnt = 0
            for mapping in [start_mapping, end_mapping, all_mapping]:
                mapping.zero_()
            is_support.fill_(IGNORE_INDEX)
            is_y1.fill_(IGNORE_INDEX)
            is_y2.fill_(IGNORE_INDEX)

            for i in range(len(cur_batch)):
                if self.ques_type:
                    _type = self.type_and_id[cur_batch[i]['id']] 
                    if _type == 'bridge':
                        ques_type[i] = 0
                    elif _type == 'comparison':
                        ques_type[i] = 1

                context_idxs[i].copy_(cur_batch[i]['context_idxs'])
                ques_idxs[i].copy_(cur_batch[i]['ques_idxs'])
                context_char_idxs[i].copy_(cur_batch[i]['context_char_idxs'])
                ques_char_idxs[i].copy_(cur_batch[i]['ques_char_idxs'])
                if cur_batch[i]['y1'] >= 0:
                    y1[i] = cur_batch[i]['y1']
                    y2[i] = cur_batch[i]['y2']
                    q_type[i] = 0
                elif cur_batch[i]['y1'] == -1:
                    y1[i] = IGNORE_INDEX
                    y2[i] = IGNORE_INDEX
                    q_type[i] = 1
                elif cur_batch[i]['y1'] == -2:
                    y1[i] = IGNORE_INDEX
                    y2[i] = IGNORE_INDEX
                    q_type[i] = 2
                elif cur_batch[i]['y1'] == -3:
                    y1[i] = IGNORE_INDEX
                    y2[i] = IGNORE_INDEX
                    q_type[i] = 3
                else:
                    assert False
                ids.append(cur_batch[i]['id'])
                _gold_mention_span = None
                if self.update_span:
                    try:
                        if 'orig_answer_span' in self.new_spans[cur_batch[i]['id']].keys():
                            assert [y1[i], y2[i]] == self.new_spans[cur_batch[i]['id']]['orig_answer_span']
                            if 'gold_mention_spans' in self.new_spans[cur_batch[i]['id']].keys():
                                _gold_mention_span = self.new_spans[cur_batch[i]['id']]['gold_mention_spans']
                                is_y1[i,:torch.sum(context_idxs[i]>0)] = 0
                                is_y2[i,:torch.sum(context_idxs[i]>0)] = 0

                                for start, end in self.new_spans[cur_batch[i]['id']]['gold_mention_spans']:
                                    is_y1[i,start] = 1
                                    is_y2[i,end] = 1
                    except AssertionError:
                        ipdb.set_trace()
                gold_mention_spans.append(_gold_mention_span)
                if self.bert:
                    try:
                        bert_vectors_c = torch.from_numpy(np.array(self.bert_buckets[0][0].get(cur_batch[i]['id']))).cuda()
                        bert_vectors_q = torch.from_numpy(np.array(self.bert_buckets[0][1].get(cur_batch[i]['id']))).cuda()
                        bert_context[i][:bert_vectors_c.shape[0]] = bert_vectors_c
                        bert_ques[i][:bert_vectors_q.shape[0]] = bert_vectors_q
                    except RuntimeError as re:
                        ipdb.set_trace()

                for j, cur_sp_dp in enumerate(cur_batch[i]['start_end_facts']):
                    if j >= self.sent_limit: break
                    if len(cur_sp_dp) == 3:
                        start, end, is_sp_flag = tuple(cur_sp_dp)
                    else:
                        start, end, is_sp_flag, is_gold = tuple(cur_sp_dp)
                    if start < end:
                        start_mapping[i, start, j] = 1
                        end_mapping[i, end-1, j] = 1
                        all_mapping[i, start:end, j] = 1
                        is_support[i, j] = int(is_sp_flag)

                max_sent_cnt = max(max_sent_cnt, len(cur_batch[i]['start_end_facts']))

            input_lengths = (context_idxs[:cur_bsz] > 0).long().sum(dim=1)
            max_c_len = int(input_lengths.max())
            max_q_len = int((ques_idxs[:cur_bsz] > 0).long().sum(dim=1).max())

            self.bkt_ptrs[bkt_id] += cur_bsz
            if self.bkt_ptrs[bkt_id] >= len(cur_bucket):
                self.bkt_pool.remove(bkt_id)

            output_dict = {'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
                'ques_idxs': ques_idxs[:cur_bsz, :max_q_len].contiguous(),
                'context_char_idxs': context_char_idxs[:cur_bsz, :max_c_len].contiguous(),
                'ques_char_idxs': ques_char_idxs[:cur_bsz, :max_q_len].contiguous(),
                'context_lens': input_lengths,
                'y1': y1[:cur_bsz],
                'y2': y2[:cur_bsz],
                'is_y1': is_y1[:cur_bsz, :max_c_len].contiguous(),
                'is_y2': is_y2[:cur_bsz, :max_c_len].contiguous(),
                'gold_mention_spans': gold_mention_spans,
                'ids': ids,
                'q_type': q_type[:cur_bsz],
                'is_support': is_support[:cur_bsz, :max_sent_cnt].contiguous(),
                'start_mapping': start_mapping[:cur_bsz, :max_c_len, :max_sent_cnt],
                'end_mapping': end_mapping[:cur_bsz, :max_c_len, :max_sent_cnt],
                'all_mapping': all_mapping[:cur_bsz, :max_c_len, :max_sent_cnt]}
            if self.bert:
                output_dict['bert_ques'] = bert_ques[:cur_bsz, :max_q_len, :].contiguous()
                output_dict['bert_context'] = bert_context[:cur_bsz, :max_c_len, :].contiguous()
            if self.ques_type:
                output_dict['ques_type'] = ques_type[:cur_bsz]
            yield output_dict

def get_buckets(record_file, example_ids=None):
    if example_ids is not None:
        eids = [ex.rstrip() for ex in open(example_ids).readlines()]
        datapoints = torch.load(record_file)
        datapoints = [dp for dp in datapoints if dp['id'] in eids]
    else:
        datapoints = torch.load(record_file)
    return [datapoints]

def get_buckets_bert(hdf5_file):
    datapoints = h5py.File(hdf5_file, 'r')
    return [datapoints]

def convert_tokens(eval_file, qa_id, pp1, pp2, p_type):
    answer_dict = {}
    for qid, p1, p2, type in zip(qa_id, pp1, pp2, p_type):
        if type == 0:
            context = eval_file[str(qid)]["context"]
            spans = eval_file[str(qid)]["spans"]
            start_idx = spans[p1][0]
            end_idx = spans[p2][1]
            answer_dict[str(qid)] = context[start_idx: end_idx]
        elif type == 1:
            answer_dict[str(qid)] = 'yes'
        elif type == 2:
            answer_dict[str(qid)] = 'no'
        elif type == 3:
            answer_dict[str(qid)] = 'noanswer'
        else:
            assert False
    return answer_dict

def evaluate(eval_file, answer_dict):
    f1 = exact_match = total = 0
    for key, value in answer_dict.items():
        total += 1
        ground_truths = eval_file[key]["answer"]
        prediction = value
        assert len(ground_truths) == 1
        cur_EM = exact_match_score(prediction, ground_truths[0])
        cur_f1, _, _ = f1_score(prediction, ground_truths[0])
        exact_match += cur_EM
        f1 += cur_f1

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}


def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


class Translator(object):
    """Translate a np array of token ids to words
    """
    def __init__(self, idx2word):
        self.word_dict = json.load(open(idx2word))
    
    def translate(self, tensor):
        return ' '.join([self.word_dict[str(wid)] for wid in tensor.data.cpu().numpy()])

    def register_batch(self, context_idxs, ques_idxs, all_mapping):
        self.context_idxs = context_idxs
        self.ques_idxs = ques_idxs
        self.all_mapping = all_mapping


    def register_sp_att(self, sp_att):
        self.sp_att1 = sp_att[0]
        self.sp_att2 = sp_att[1]
        self.sp_att3 = sp_att[2]

    def ques(self, bid):
        print(self.translate(self.ques_idxs[bid]))

    def context(self, bid):
        print(self.translate(self.context_idxs[bid]))

    def top_att(self, bid, att, top):
        if att == 1:
            att_value, sentence_id = self.sp_att1[bid].topk(top, dim=0)
        if att == 2:
            att_value, sentence_id = self.sp_att2[bid].topk(top, dim=0)
        if att == 3:
            att_value, sentence_id = self.sp_att3[bid].topk(top, dim=0)
        if att == 4:
            att_value, sentence_id = self.sp_att4[bid].topk(top, dim=0)
        att_value = att_value[-1].cpu().data.numpy()[0]
        sentence_id = sentence_id[-1].cpu().data.numpy()[0] - 2
        print('Sentence id:',sentence_id)
        word_ids = self.all_mapping[bid,:,sentence_id].nonzero().data.cpu().numpy()
        if len(word_ids) > 0:          
            sent_start = int(word_ids[0])
            sent_end = int(word_ids[-1])

            words = self.context_idxs[bid][sent_start:sent_end]
            print(att_value, self.translate(words))
        else:
            print('Selected sentence is out of mask bounds')
