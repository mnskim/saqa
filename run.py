from collections import OrderedDict
import math
import ipdb
import ujson as json
import numpy as np
from tqdm import tqdm
import os
from torch import optim, nn
from model import Model #, NoCharModel, NoSelfModel
from sp_model import SPModel
from util import convert_tokens, evaluate
from util import get_buckets, get_buckets_bert, DataIterator, IGNORE_INDEX
import time
import shutil
import random
import torch
from torch.autograd import Variable
import sys
from torch.nn import functional as F

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

nll_sum = nn.CrossEntropyLoss(size_average=False, ignore_index=IGNORE_INDEX)
nll_average = nn.CrossEntropyLoss(size_average=True, ignore_index=IGNORE_INDEX)
nll_all = nn.CrossEntropyLoss(reduce=False, ignore_index=IGNORE_INDEX)
nll_raw = nn.CrossEntropyLoss(reduce=False, ignore_index=IGNORE_INDEX, size_average=False)

def train(config):
    if config.bert:
        if config.bert_with_glove:
            with open(config.word_emb_file, "r") as fh:
                word_mat = np.array(json.load(fh), dtype=np.float32)
        else:
            word_mat = None
    else:
        with open(config.word_emb_file, "r") as fh:
            word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)
    with open(config.idx2word_file, 'r') as fh:
        idx2word_dict = json.load(fh)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    config.save = '{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(config.save, scripts_to_save=['run.py', 'model.py', 'util.py', 'sp_model.py', 'macnet_v2.py'])
    def logging(s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(config.save, 'log.txt'), 'a+') as f_log:
                f_log.write(s + '\n')

    logging('Config')
    for k, v in config.__dict__.items():
        logging('    - {} : {}'.format(k, v))

    logging("Building model...")
    train_buckets = get_buckets(config.train_record_file)
    dev_buckets = get_buckets(config.dev_record_file)
    max_iter = math.ceil(len(train_buckets[0])/config.batch_size)
        
    def build_train_iterator():
        if config.bert:
            train_context_buckets = get_buckets_bert(os.path.join(config.bert_dir, config.train_bert_emb_context))
            train_ques_buckets = get_buckets_bert(os.path.join(config.bert_dir, config.train_bert_emb_ques))
            return DataIterator(train_buckets, config.batch_size, config.para_limit, config.ques_limit, config.char_limit, True, config.sent_limit, bert=True, bert_buckets=list(zip(train_context_buckets, train_ques_buckets)), new_spans=True)
        else:
            return DataIterator(train_buckets, config.batch_size, config.para_limit, config.ques_limit, config.char_limit, True, config.sent_limit, bert=False, new_spans=True)

    def build_dev_iterator():
        # Iterator for inference during training
        if config.bert:
            dev_context_buckets = get_buckets_bert(os.path.join(config.bert_dir, config.dev_bert_emb_context))
            dev_ques_buckets = get_buckets_bert(os.path.join(config.bert_dir, config.dev_bert_emb_ques))
            return DataIterator(dev_buckets, config.batch_size, config.para_limit, config.ques_limit, config.char_limit, False, config.sent_limit, bert=True, bert_buckets=list(zip(dev_context_buckets, dev_ques_buckets)))
        else:
            return DataIterator(dev_buckets, config.batch_size, config.para_limit, config.ques_limit, config.char_limit, False, config.sent_limit, bert=False)

    if config.sp_lambda > 0:
        model = SPModel(config, word_mat, char_mat)
    else:
        model = Model(config, word_mat, char_mat)

    logging('nparams {}'.format(sum([p.nelement() for p in model.parameters() if p.requires_grad])))
    ori_model = model.cuda()
    model = nn.DataParallel(ori_model)

    lr = config.init_lr
    if config.optim == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config.init_lr)
    elif config.optim == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.init_lr)
    if config.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epoch*max_iter, eta_min=0.0001)
    elif config.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=config.patience, verbose=True)
    cur_patience = 0
    total_loss = 0
    global_step = 0
    best_dev_F1 = None
    stop_train = False
    start_time = time.time()
    eval_start_time = time.time()
    model.train()

    for epoch in range(config.epoch):
        for data in build_train_iterator():
            scheduler.step()
            context_idxs = Variable(data['context_idxs'])
            ques_idxs = Variable(data['ques_idxs'])
            context_char_idxs = Variable(data['context_char_idxs'])
            ques_char_idxs = Variable(data['ques_char_idxs'])
            context_lens = Variable(data['context_lens'])
            y1 = Variable(data['y1'])
            y2 = Variable(data['y2'])
            is_y1 = Variable(data['is_y1'])
            is_y2 = Variable(data['is_y2'])

            q_type = Variable(data['q_type'])
            is_support = Variable(data['is_support'])
            start_mapping = Variable(data['start_mapping'])
            end_mapping = Variable(data['end_mapping'])
            all_mapping = Variable(data['all_mapping'])
            if config.bert:
                bert_context = Variable(data['bert_context'])
                bert_ques = Variable(data['bert_ques'])
            else:
                bert_context = None
                bert_ques = None

            support_ids = (is_support == 1).nonzero()
            sp_dict = {idx:[] for idx in range(context_idxs.shape[0])}
            for row in support_ids:
                bid, sp_sent_id = row
                sp_dict[bid.data.cpu()[0]].append(sp_sent_id.data.cpu()[0])

            if config.sp_shuffle:
                [random.shuffle(value) for value in sp_dict.values()]

            sp1_labels = []
            sp2_labels = []
            sp3_labels = []
            sp4_labels = []
            sp5_labels = []
            for item in sorted(sp_dict.items(), key= lambda t:t[0]):
                bid, supports = item
                if len(supports) == 1:
                    sp1_labels.append(supports[0])
                    sp2_labels.append(-1)
                    sp3_labels.append(-2)
                    sp4_labels.append(-2)
                    sp5_labels.append(-2)
                elif len(supports) == 2:
                    sp1_labels.append(supports[0])
                    sp2_labels.append(supports[1])
                    sp3_labels.append(-1)
                    sp4_labels.append(-2)
                    sp5_labels.append(-2)
                elif len(supports) == 3:
                    sp1_labels.append(supports[0])
                    sp2_labels.append(supports[1])
                    sp3_labels.append(supports[2])
                    sp4_labels.append(-1)
                    sp5_labels.append(-2)
                elif len(supports) >= 4: # 4 or greater sp are treated the same
                    sp1_labels.append(supports[0])
                    sp2_labels.append(supports[1])
                    sp3_labels.append(supports[2])
                    sp4_labels.append(supports[3])
                    sp5_labels.append(-1)

            # We will append 2 vectors to the front (sp_output_with_end), so we increment indices by 2
            sp1_labels = np.array(sp1_labels) + 2
            sp2_labels = np.array(sp2_labels) + 2
            sp3_labels = np.array(sp3_labels) + 2
            sp4_labels = np.array(sp4_labels) + 2
            sp5_labels = np.array(sp5_labels) + 2
            sp_labels_mod = Variable(torch.LongTensor(np.stack([sp1_labels, sp2_labels, sp3_labels, sp4_labels, sp5_labels],1)).cuda())

            sp_mod_mask = sp_labels_mod>0

            if config.bert:              
                logit1, logit2, grn_logit1, grn_logit2, coarse_att, predict_type, predict_support, sp_att_logits = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, return_yp=False, support_labels=is_support, is_train=True, sp_labels_mod=sp_labels_mod, bert=True, bert_context=bert_context, bert_ques=bert_ques)
            else:
                logit1, logit2, grn_logit1, grn_logit2, coarse_att, predict_type, predict_support, sp_att_logits = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, return_yp=False, support_labels=is_support, is_train=True, sp_labels_mod=sp_labels_mod, bert=False)

            sp_att1 = sp_att_logits[0].squeeze(-1)
            sp_att2 = sp_att_logits[1].squeeze(-1)
            sp_att3 = sp_att_logits[2].squeeze(-1)
            if config.reasoning_steps == 4:
                sp_att4 = sp_att_logits[3].squeeze(-1)

            # Add masks to targets(labels) to ignore in loss calculation
            sp1_labels = ((sp_mod_mask[:,0].float() - 1 )*100).cpu().data.numpy() + sp1_labels
            sp2_labels = ((sp_mod_mask[:,1].float() - 1 )*100).cpu().data.numpy() + sp2_labels
            sp3_labels = ((sp_mod_mask[:,2].float() - 1 )*100).cpu().data.numpy() + sp3_labels
            sp4_labels = ((sp_mod_mask[:,3].float() - 1 )*100).cpu().data.numpy() + sp4_labels

            sp1_loss = nll_average(sp_att1, Variable(torch.LongTensor(sp1_labels).cuda()))
            sp2_loss = nll_average(sp_att2, Variable(torch.LongTensor(sp2_labels).cuda()))
            sp3_loss = nll_average(sp_att3, Variable(torch.LongTensor(sp3_labels).cuda()))
            if config.reasoning_steps == 4:
                sp4_loss = nll_average(sp_att4, Variable(torch.LongTensor(sp4_labels).cuda()))
            else:
                sp4_loss = 0


            
            GRN_SP_PRED = True# temporary flag
            
            batch_losses = []
            for bid, spans in enumerate(data['gold_mention_spans']):
                if (not spans == None) and (not spans == []):
                    try:
                        _loss_start_avg = F.cross_entropy(logit1[bid].view(1, -1).expand(len(spans), len(logit1[bid])) , Variable(torch.LongTensor(spans))[:,0].cuda(), reduce=True)
                        _loss_end_avg = F.cross_entropy(logit2[bid].view(1, -1).expand(len(spans), len(logit2[bid])) , Variable(torch.LongTensor(spans))[:,1].cuda(), reduce=True)
                    except IndexError:
                        ipdb.set_trace()
                    batch_losses.append((_loss_start_avg + _loss_end_avg))
            loss_1 = torch.mean(torch.stack(batch_losses)) + nll_sum(predict_type, q_type) / context_idxs.size(0)

            if GRN_SP_PRED:
                loss_2 = 0
            else:
                loss_2 = nll_average(predict_support.view(-1, 2), is_support.view(-1))
            loss = loss_1 + config.sp_lambda * loss_2 + sp1_loss + sp2_loss + sp3_loss + sp4_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.data[0]
            global_step += 1
            if global_step % config.period == 0:
                cur_loss = total_loss / config.period
                elapsed = time.time() - start_time
                logging('| epoch {:3d} | step {:6d} | lr {:05.5f} | ms/batch {:5.2f} | train loss {:8.3f}'.format(epoch, global_step, optimizer.param_groups[0]['lr'], elapsed*1000/config.period, cur_loss))
                total_loss = 0
                start_time = time.time()

            if global_step % config.checkpoint == 0:
                model.eval()
                metrics = evaluate_batch(build_dev_iterator(), model, 0, dev_eval_file, config)
                model.train()
                logging('-' * 89)
                logging('| eval {:6d} in epoch {:3d} | time: {:5.2f}s | dev loss {:8.3f} | EM {:.4f} | F1 {:.4f}'.format(global_step//config.checkpoint,
                    epoch, time.time()-eval_start_time, metrics['loss'], metrics['exact_match'], metrics['f1']))
                logging('-' * 89)

                eval_start_time = time.time()

                dev_F1 = metrics['f1']
                if config.scheduler == 'plateau':
                    scheduler.step(dev_F1)
                if best_dev_F1 is None or dev_F1 > best_dev_F1:
                    best_dev_F1 = dev_F1
                    torch.save(ori_model.state_dict(), os.path.join(config.save, 'model.pt'))
                    cur_patience = 0
        if stop_train: break
    logging('best_dev_F1 {}'.format(best_dev_F1))

def evaluate_batch(data_source, model, max_batches, eval_file, config):
    answer_dict = {}
    sp_dict = {}
    total_loss, step_cnt = 0, 0
    iter = data_source
    for step, data in enumerate(iter):
        if step >= max_batches and max_batches > 0: break

        context_idxs = Variable(data['context_idxs'], volatile=True)
        ques_idxs = Variable(data['ques_idxs'], volatile=True)
        context_char_idxs = Variable(data['context_char_idxs'], volatile=True)
        ques_char_idxs = Variable(data['ques_char_idxs'], volatile=True)
        context_lens = Variable(data['context_lens'], volatile=True)
        y1 = Variable(data['y1'], volatile=True)
        y2 = Variable(data['y2'], volatile=True)
        q_type = Variable(data['q_type'], volatile=True)
        is_support = Variable(data['is_support'], volatile=True)
        start_mapping = Variable(data['start_mapping'], volatile=True)
        end_mapping = Variable(data['end_mapping'], volatile=True)
        all_mapping = Variable(data['all_mapping'], volatile=True)
        if config.bert:
            bert_context = Variable(data['bert_context'], volatile=True)
            bert_ques = Variable(data['bert_ques'], volatile=True)
        else:
            bert_context = None
            bert_ques = None

        if config.bert:
            logit1, logit2, grn_logit1, grn_logit2, coarse_att, predict_type, predict_support, yp1, yp2 = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, return_yp=True, support_labels=is_support, bert=True, bert_context=bert_context, bert_ques=bert_ques)
        else: 
            logit1, logit2, grn_logit1, grn_logit2, coarse_att, predict_type, predict_support, yp1, yp2 = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, return_yp=True, support_labels=is_support, bert=False)
        loss = (nll_sum(predict_type, q_type) + nll_sum(logit1, y1) + nll_sum(logit2, y2)) / context_idxs.size(0) + config.sp_lambda * nll_average(predict_support.view(-1, 2), is_support.view(-1))
        answer_dict_ = convert_tokens(eval_file, data['ids'], yp1.data.cpu().numpy().tolist(), yp2.data.cpu().numpy().tolist(), np.argmax(predict_type.data.cpu().numpy(), 1))
        answer_dict.update(answer_dict_)

        total_loss += loss.data[0]
        step_cnt += 1
    loss = total_loss / step_cnt
    metrics = evaluate(eval_file, answer_dict)
    metrics['loss'] = loss

    return metrics

def predict(data_source, model, eval_file, config, prediction_file):
    answer_dict = {}
    sp_dict = {}
    sp_th = config.sp_threshold
    for step, data in enumerate(tqdm(data_source)):
        #if '5a7f5da85542992097ad2f3e' in data['ids']:
        #    ipdb.set_trace()
        context_idxs = Variable(data['context_idxs'], volatile=True)
        ques_idxs = Variable(data['ques_idxs'], volatile=True)
        context_char_idxs = Variable(data['context_char_idxs'], volatile=True)
        ques_char_idxs = Variable(data['ques_char_idxs'], volatile=True)
        context_lens = Variable(data['context_lens'], volatile=True)
        is_support = Variable(data['is_support'], volatile=True) # @Minsoo
        start_mapping = Variable(data['start_mapping'], volatile=True)
        end_mapping = Variable(data['end_mapping'], volatile=True)
        all_mapping = Variable(data['all_mapping'], volatile=True)


        if config.bert:
            bert_context = Variable(data['bert_context'], volatile=True)
            bert_ques = Variable(data['bert_ques'], volatile=True)
        else:
            bert_context = None
            bert_ques = None

        if config.bert:
            logit1, logit2, grn_logit1, grn_logit2, coarse_att, predict_type, predict_support, yp1, yp2 = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, return_yp=True, support_labels=is_support, bert=True, bert_context=bert_context, bert_ques=bert_ques)
        else:
            logit1, logit2, grn_logit1, grn_logit2, coarse_att, predict_type, predict_support, yp1, yp2 = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, return_yp=True, support_labels=is_support, bert=False)
        answer_dict_ = convert_tokens(eval_file, data['ids'], yp1.data.cpu().numpy().tolist(), yp2.data.cpu().numpy().tolist(), np.argmax(predict_type.data.cpu().numpy(), 1))  
        answer_dict.update(answer_dict_)

        predict_support_np = torch.sigmoid(predict_support[:, :, 1]).data.cpu().numpy()
        for i in range(predict_support_np.shape[0]):
            cur_sp_pred = []
            cur_id = data['ids'][i]
            for j in range(predict_support_np.shape[1]):
                if j >= len(eval_file[cur_id]['sent2title_ids']): break
                if predict_support_np[i, j] > sp_th:
                    cur_sp_pred.append(eval_file[cur_id]['sent2title_ids'][j])
            sp_dict.update({cur_id: cur_sp_pred})

    prediction = {'answer': answer_dict, 'sp': sp_dict}
    with open(prediction_file, 'w') as f:
        json.dump(prediction, f)

def test(config):
    # Inference mode (after training)
    if config.bert:
        if config.bert_with_glove:
            with open(config.word_emb_file, "r") as fh:
                word_mat = np.array(json.load(fh), dtype=np.float32)
        else:
            word_mat = None
    else:
        with open(config.word_emb_file, "r") as fh:
            word_mat = np.array(json.load(fh), dtype=np.float32)

    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    if config.data_split == 'dev':
        with open(config.dev_eval_file, "r") as fh:
            dev_eval_file = json.load(fh)
    else:
        with open(config.test_eval_file, 'r') as fh:
            dev_eval_file = json.load(fh)
    with open(config.idx2word_file, 'r') as fh:
        idx2word_dict = json.load(fh)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    def logging(s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(config.save, 'log.txt'), 'a+') as f_log:
                f_log.write(s + '\n')

    if config.data_split == 'dev':
        dev_buckets = get_buckets(config.dev_record_file, os.path.join(config.bert_dir, config.dev_example_ids))
        if config.bert:
            dev_context_buckets = get_buckets_bert(os.path.join(config.bert_dir, config.dev_bert_emb_context))
            dev_ques_buckets = get_buckets_bert(os.path.join(config.bert_dir, config.dev_bert_emb_ques))

        para_limit = config.para_limit
        ques_limit = config.ques_limit
    elif config.data_split == 'test':
        para_limit = None
        ques_limit = None
        dev_buckets = get_buckets(config.test_record_file, os.path.join(config.bert_dir, config.test_example_ids))
        if config.bert:
            dev_context_buckets = get_buckets_bert(os.path.join(config.bert_dir, config.test_bert_emb_context))
            dev_ques_buckets = get_buckets_bert(os.path.join(config.bert_dir, config.test_bert_emb_ques))

    def build_dev_iterator():
        if config.bert:
            return DataIterator(dev_buckets, config.batch_size, para_limit, ques_limit, config.char_limit, False, config.sent_limit, bert=True, bert_buckets=list(zip(dev_context_buckets, dev_ques_buckets)))
        else:
            return DataIterator(dev_buckets, config.batch_size, para_limit,
                ques_limit, config.char_limit, False, config.sent_limit, bert=False)

    if config.sp_lambda > 0:
        model = SPModel(config, word_mat, char_mat)
    else:
        model = Model(config, word_mat, char_mat)
    ori_model = model.cuda()
    ori_model.load_state_dict(torch.load(os.path.join(config.save, 'model.pt')))
    model = nn.DataParallel(ori_model)

    model.eval()
    predict(build_dev_iterator(), model, dev_eval_file, config, config.prediction_file)

