import shutil
import h5py
import argparse
import collections
from tqdm import tqdm
import logging
import itertools
import ipdb
from pytorch_bert.pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_bert.pytorch_pretrained_bert.modeling import BertModel
from torch.utils.data import TensorDataset, DataLoader, Dataset

import random
from tqdm import tqdm
import spacy
import ujson as json
from collections import Counter
import numpy as np
import os.path
import argparse
import torch
import torch
import os
from joblib import Parallel, delayed
import bisect
import re

train_record_path = 'train_record.pkl' 
train_example_path = 'train_example.json'
#train_example_path = 'dev_example.json'
train_eval_path = 'train_eval.json' 
dev_record_path = 'dev_record.pkl' 
dev_example_path = 'dev_example.json'
dev_eval_path = 'dev_eval.json' 
test_record_path = 'test_record.pkl' 
test_example_path = 'test_example.json'
test_eval_path = 'test_eval.json' 
idx2word_path = 'idx2word.json'
idx2char_path = 'idx2char.json'
char2idx_path = 'char2idx.json'
word2idx_path = 'word2idx.json'
word_emb_path = 'word_emb.json'
char_emb_path = 'char_emb.json'
save_format = '{}_bert_emb_{}.h5'
save_eid_format = '{}_example_ids.txt'

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

nlp = spacy.blank("en")

parser = argparse.ArgumentParser()
parser.add_argument('--bert_model', type=str, default='pytorch_bert/weights/cased_L-12_H-768_A-12/')
parser.add_argument('--maxlen', type=int, default=512, help='Bert model max seq length')
parser.add_argument('--stride', type=int, default=256, help='Sliding window stride')
parser.add_argument('--n_proc', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--data_start', type=int, default=0)
parser.add_argument('--data_end', type=int, default=9999999)

# Pooling choices
parser.add_argument('--layer_pooling', type=str, default='sum_last_four', 
        choices=['sum_last_four', 'avg_last_four', 'last', '2nd_last'] + list(map(str, range(-24,0))), help='How to pool the different layers')
parser.add_argument('--wordpiece_pooling', type=str, default='avg', choices=['avg', 'sum'], help='How to pool wordpieces into a single token')
parser.add_argument('--window_pooling', type=str, default='avg', choices=['avg', 'sum'], help='How to pool sliding windows')

# Other choices
parser.add_argument('--do_lower_case', action='store_true', help='Case sensitivity. Doesnt really matter since we call wordpiece tokenizer directly')
parser.add_argument('--data_split', type=str, choices=['train', 'dev', 'test'], default='train')
parser.add_argument('--fullwiki', action='store_true')
parser.add_argument('--save_dir', type=str, default='debug1')

config = parser.parse_args()

def _concat(filename):
    if config.fullwiki:
        return 'fullwiki.{}'.format(filename)
    return filename

dev_record_path = _concat(dev_record_path) 
dev_example_path = _concat(dev_example_path)
dev_eval_path = _concat(dev_eval_path)
test_record_path = _concat(test_record_path)
test_example_path = _concat(test_example_path)
test_eval_path = _concat(test_eval_path)
save_format = _concat(save_format)
save_eid_format = _concat(save_eid_format)

json.dump(config, open(os.path.join(config.save_dir, _concat('{}_config.json'.format(config.data_split))), 'w'))

bert_tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=config.do_lower_case)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def wordpiece_tokenize(word):
    if word == '--OOV--':
        wordpieces = ['[UNK]']
    elif word == '--NULL--':
        wordpieces = ['[PAD]']
    else:
        wordpieces = bert_tokenizer.wordpiece_tokenizer.tokenize(word)
    return wordpieces

def prepro_bert(config):
    idx2word = json.load(open(idx2word_path, 'rb'))
    idx2char = json.load(open(idx2char_path, 'rb'))
    char2idx = json.load(open(char2idx_path, 'rb'))
    word2idx = json.load(open(word2idx_path, 'rb'))
    word_emb = json.load(open(word_emb_path, 'rb'))
    char_emb = json.load(open(char_emb_path, 'rb'))

    if config.data_split == 'train':
        example_path = train_example_path
        example_data = json.load(open(example_path, 'rb'))
        logger.info('loaded example file {}'.format(example_path))

    if config.data_split == 'dev':
        example_path = dev_example_path
        example_data = json.load(open(example_path, 'rb'))
        logger.info('loaded example file {}'.format(example_path))

    if config.data_split == 'test':
        example_path = test_example_path
        example_data = json.load(open(example_path, 'rb'))
        logger.info('loaded example file {}'.format(example_path))
 
    def _process_example(example):
        context_tokens = example['context_tokens']
        context_wordpieces = [wordpiece_tokenize(token) for token in context_tokens]
        ques_tokens = example['ques_tokens']
        ques_wordpieces = [wordpiece_tokenize(token) for token in ques_tokens]

        # Map
        ques_tok_to_orig_index = []
        all_ques_wordpieces = []
        for (i, token) in enumerate(ques_tokens):
            sub_tokens = wordpiece_tokenize(token)
            for sub_token in sub_tokens:
                ques_tok_to_orig_index.append(i)
                all_ques_wordpieces.append(sub_token)


        context_tok_to_orig_index = []
        all_context_wordpieces = []
        for (i, token) in enumerate(context_tokens):
            sub_tokens = wordpiece_tokenize(token)
            for sub_token in sub_tokens:
                context_tok_to_orig_index.append(i)
                all_context_wordpieces.append(sub_token)

        # Because context can be longer than 512, we take sliding windows
        # The -3 accounts for [CLS] and [SEP]
        max_tokens_for_doc = config.maxlen - 2
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_context_wordpieces):
            length = len(all_context_wordpieces) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_context_wordpieces):
                break
            start_offset += min(length, config.stride)

        features = []
        for (doc_span_index, doc_span) in enumerate(doc_spans):
            final_context_tokens = []
            context_token_to_orig_map = {}
            final_context_tokens.append("[CLS]")
           
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                context_token_to_orig_map[len(final_context_tokens)] = context_tok_to_orig_index[split_token_index]
                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                final_context_tokens.append(all_context_wordpieces[split_token_index])

            final_context_tokens.append("[SEP]")
            context_input_ids = bert_tokenizer.convert_tokens_to_ids(final_context_tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            context_input_mask = [1] * len(context_input_ids)

            # Zero-pad up to the sequence length.
            while len(context_input_ids) < config.maxlen:
                context_input_ids.append(0)
                context_input_mask.append(0)

            assert len(context_input_ids) == config.maxlen
            assert len(context_input_mask) == config.maxlen


            # Question
            final_ques_tokens = []
            ques_token_to_orig_map = {}
            final_ques_tokens.append("[CLS]")
                    
            for i, token in enumerate(all_ques_wordpieces):
                split_token_index = i
                ques_token_to_orig_map[len(final_ques_tokens)] = ques_tok_to_orig_index[split_token_index]
                final_ques_tokens.append(token)
            final_ques_tokens.append("[SEP]")
            ques_input_ids = bert_tokenizer.convert_tokens_to_ids(final_ques_tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            ques_input_mask = [1] * len(ques_input_ids)

            # Zero-pad up to the sequence length.
            while len(ques_input_ids) < config.maxlen:
                ques_input_ids.append(0)
                ques_input_mask.append(0)

            assert len(ques_input_ids) == config.maxlen
            assert len(ques_input_mask) == config.maxlen

            if config.n_proc == 1:
                ipdb.set_trace()
            features.append({'id': example['id'], 'context_input_ids': context_input_ids, 'context_input_mask': context_input_mask, 'context_token_to_orig_map': context_token_to_orig_map,  'ques_input_ids': ques_input_ids, 'ques_input_mask': ques_input_mask, 'ques_token_to_orig_map': ques_token_to_orig_map})
        return features

    filtered_ids = set([item.rstrip() for item in open(os.path.join(config.save_dir, 'ids.txt')).readlines()])
    example_data = [item for item in example_data if item['id'] in filtered_ids]

    outputs = Parallel(n_jobs=config.n_proc, verbose=10)(delayed(_process_example)(example) for example in example_data[config.data_start:config.data_end])
    return outputs

def merge_windows(example):
    """Merge sliding windows"""
    n_context_words = max(list(example['context_token_maps'][-1].values())) + 1
    n_ques_words = max(list(example['ques_token_maps'][-1].values())) + 1
    n_windows = len(example['context_embeddings_windows'])

    context_embeddings = [{item: [] for item in range(n_context_words)} for w in range(n_windows)]
    ques_embeddings = {item: [] for item in range(n_ques_words)}

    for window in range(n_windows):
        # Process one window
        for k, v in example['context_token_maps'][window].items():
            context_embeddings[window][v].append(example['context_embeddings_windows'][window][k])

    # All ques embedding should be copies of the same thing
    assert False not in ([(not 0 in (item == example['ques_embeddings'][0])) for item in example['ques_embeddings'] if not 0 in (item == example['ques_embeddings'][0])])
    for k, v in example['ques_token_maps'][0].items():
            ques_embeddings[v].append(example['ques_embeddings'][window][k])
    
    try:
        #qe_mat = np.stack([item[1] for item in list(ques_embeddings.items())]).squeeze(1)
        qe_mat = np.stack([item[1][0] for item in list(ques_embeddings.items())])
    except:
        ipdb.set_trace()
    ce_mat = []

    # Wordpiece pooling
    for window in range(n_windows):
        for k, v in context_embeddings[window].items():
            if v: # If in current window
                if config.wordpiece_pooling == 'avg':
                    context_embeddings[window][k] = np.mean(v, 0)
                elif config.wordpiece_pooling == 'sum':
                    context_embeddings[window][k] = np.sum(v, 0)
                if np.sum(v, 0).shape == ():
                    # Catch bug
                    ipdb.set_trace()
    for k, v in ques_embeddings.items():
        if config.wordpiece_pooling == 'avg':
            ques_embeddings[k] = np.mean(v, 0)
        elif config.wordpiece_pooling == 'sum':
            ques_embeddings[k] = np.sum(v, 0)
        if np.sum(v, 0).shape == ():
            # Catch bug
            ipdb.set_trace()


    # Window pooling
    for k in range(n_context_words):
        try:
            window_vecs = [context_embeddings[window][k] for window in range(n_windows)]
            window_vecs = [item for item in window_vecs if not len(item) == 0] # Get rid of [] entries from other windows 
            if config.window_pooling == 'avg':
                ce_mat.append(np.mean(window_vecs, 0))
            elif config.window_pooling == 'sum':
                ce_mat.append(np.sum(window_vecs, 0))
        except:
            ipdb.set_trace()
    ce_mat = np.stack(ce_mat)
    return [qe_mat, ce_mat]

outputs = prepro_bert(config)

model = BertModel.from_pretrained(config.bert_model)
device = torch.device("cuda")
model.to(device)
model = torch.nn.DataParallel(model)
model.eval()

train_features = list(itertools.chain(*outputs))

example_ids = [f['id'] for f in train_features]
all_context_input_ids = torch.tensor([f['context_input_ids'] for f in train_features], dtype=torch.long)
all_context_input_mask = torch.tensor([f['context_input_mask'] for f in train_features], dtype=torch.long)
all_ques_input_ids = torch.tensor([f['ques_input_ids'] for f in train_features], dtype=torch.long)
all_ques_input_mask = torch.tensor([f['ques_input_mask'] for f in train_features], dtype=torch.long)

all_context_token_maps = [f['context_token_to_orig_map'] for f in train_features]
all_ques_token_maps = [f['ques_token_to_orig_map'] for f in train_features]

batch_to_example_map = []
for idx, item in enumerate(outputs):
    for _ in range(len(item)):
        batch_to_example_map.append(idx)
start_ids = np.concatenate([np.array([0]), np.cumsum([len(item) for item in outputs])[:-1]])
end_ids = np.cumsum([len(item) for item in outputs]) -1

train_data = TensorDataset(all_context_input_ids, all_context_input_mask, all_ques_input_ids, all_ques_input_mask)

train_dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=False, num_workers=8)
example = {'context_embeddings_windows': [], 'ques_embeddings': [], 'context_token_maps': [], 'ques_token_maps': []}
num_data = 0
save_data = []
example_ids_list = open(os.path.join(config.save_dir, save_eid_format.format(config.data_split)), 'w')
ques_hf = h5py.File(os.path.join(config.save_dir, save_format.format(config.data_split, 'ques')), 'w')
context_hf = h5py.File(os.path.join(config.save_dir, save_format.format(config.data_split, 'context')), 'w')

for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
    context_input_ids, context_input_mask, ques_input_ids, ques_input_mask = batch
    # Run BERT encoder on context and questions
    context_all_encoder_layers, _ = model(context_input_ids, token_type_ids=None, attention_mask=context_input_mask)
    context_all_encoder_layers = [item.cpu().data.numpy() for item in context_all_encoder_layers]
    ques_all_encoder_layers, _ = model(ques_input_ids, token_type_ids=None, attention_mask=ques_input_mask)
    ques_all_encoder_layers = [item.cpu().data.numpy() for item in ques_all_encoder_layers]

    all_ids = list(range(num_data, num_data+len(context_input_ids)))
    num_data += len(context_input_ids)
    for batch_idx, idx in enumerate(all_ids):
        if config.layer_pooling.lstrip('-').isdigit():
            context_bert_embeddings = context_all_encoder_layers[int(config.layer_pooling)]#.cpu().data.numpy()
            ques_bert_embeddings = ques_all_encoder_layers[int(config.layer_pooling)]#.cpu().data.numpy()
        elif config.layer_pooling == 'sum_last_four': # In the BERT paper summing the last four layers is as good as concatting the last four on NER task.
            context_bert_embeddings = np.sum(torch.stack(context_all_encoder_layers[-4:],-1), -1)#.cpu().data.numpy()
            ques_bert_embeddings = np.sum(torch.stack(ques_all_encoder_layers[-4:],-1), -1)#.cpu().data.numpy()
        elif config.layer_pooling == 'avg_last_four': # Jacob Devlin also suggests averaging the last four layers.
            context_bert_embeddings = np.mean(torch.stack(context_all_encoder_layers[-4:],-1), -1)#.cpu().data.numpy()
            ques_bert_embeddings = np.mean(torch.stack(ques_all_encoder_layers[-4:],-1), -1)#.cpu().data.numpy()

        example['context_embeddings_windows'].append(context_bert_embeddings[batch_idx])
        example['ques_embeddings'].append(ques_bert_embeddings[batch_idx])
        example['context_token_maps'].append(all_context_token_maps[idx])
        example['ques_token_maps'].append(all_ques_token_maps[idx])

        if idx in end_ids:
            #assert len(example['windows']) ==  # TODO: A check
            question_embeddings, context_embeddings = merge_windows(example)
            _eid = example_ids[idx]
            #if _eid == '5ae1d1a55542997f29b3c138' or _eid =='5ae3fe635542996836b02c07':
            #    ipdb.set_trace()
            example_ids_list.write(_eid + '\n')
            ques_hf.create_dataset(_eid, data=question_embeddings)
            context_hf.create_dataset(_eid, data=context_embeddings)

            # We finished one example (article). Reset the example dict.
            example = {'context_embeddings_windows': [], 'ques_embeddings': [], 'context_token_maps': [], 'ques_token_maps': []}

example_ids_list.close()
ques_hf.close()
context_hf.close()
