# Improving Reasoning in Multi-Hop QA with Self-attention

This repository contains the implementation of the SAQA model and the HotpotQA experiments included in "Improving Reasoning in Multi-Hop QA with Self-attention". 

The test set results can be found at the [HotpotQA leaderboard](https://hotpotqa.github.io).

The base code is from the official [HotpotQA repository](https://github.com/hotpotqa/hotpot).

## Requirements

Python 3, pytorch 0.3.0, spacy

To install pytorch 0.3.0, follow the instructions at https://pytorch.org/get-started/previous-versions/ . For example, with
CUDA8 and conda you can do
```
conda install pytorch=0.3.0 cuda80 -c pytorch
```

To install spacy, run
```
conda install spacy
```

While the model code requires pytorch 0.3.0, the pytorch_pretrained_bert library (an older version of [Huggingface's Pytorch-Transformers](https://github.com/huggingface/pytorch-transformers) ), which we use for getting BERT embeddings, requires pytorch 1.0. We recommend keeping a separate conda environment for each. We provide the files for pytorch_pretrained_bert, so installation is unnecessary.

## Data Download and Preprocessing

Run the script to download the data, including HotpotQA data and GloVe embeddings, as well as spacy packages.
```
./download.sh
```

## Preprocessing

Preprocess the training and dev sets in the distractor setting:
```
python main.py --mode prepro --data_file hotpot_train_v1.1.json --para_limit 2250 --data_split train
python main.py --mode prepro --data_file hotpot_dev_distractor_v1.json --para_limit 2250 --data_split dev
```

Preprocess the dev set in the full wiki setting:
```
python main.py --mode prepro --data_file hotpot_dev_fullwiki_v1.json --data_split dev --fullwiki --para_limit 2250
```

## Creating ELMo-style BERT vectors

We use BERT embeddings as additional word embeddings, like GloVe and ELMo. Once you've preprocessed the training or dev set, you can create the BERT embeddings for the dataset by running:

```
python process_bert_hdf5_split.py --data_split train --layer_pooling -2 --bert_model bert-base-cased --n_proc 12 --window_pooling avg --wordpiece_pooling sum --batch_size 8 --save_dir $BERTDIR
python process_bert_hdf5_split.py --data_split dev --layer_pooling -2 --bert_model bert-base-cased --n_proc 12 --window_pooling avg --wordpiece_pooling sum --batch_size 8 --fullwiki --save_dir $BERTDIR
```
This will save the BERT embeddings into $BERTDIR . The BERT embeddings are quite heavy and require a lot of storage. For the training set, roughly 320GB of space is needed to store the embeddings. However, since we use the HDF5 format, there is no memory burden during training. 

Although saving all of the embeddings is required during training, we can save only portions of them at a time during evaluation. A script is provided to do this, which can be useful for minimizing storage, and for running on environments such as CodaLab.


## Training

Train SAQA with the following command, setting $BERTDIR the same as the previous step: 

```
PYTHONDONTWRITEBYTECODE=1 python main.py --mode train --para_limit 2250 --batch_size 24 --init_lr 0.001 --hidden 80 --keep_prob 1.0  --sp_lambda 1.0 --epoch 15 --optim adam --scheduler cosine --patience 2 --pointing --reasoning_steps 3 --bert --bert_with_glove --bert_dir $BERTDIR --sp_shuffle --teacher_forcing
```

This will train SAQA with support sentence teacher forcing, support sentence order shuffling, 3 maximum sentence identification steps, with cosine learning rate annealing. The args `--bert` and `--bert_with_glove` mean the model will use both BERT and GloVe. To only use BERT, remove `--bert_with_glove`, and to only use GloVe, remove both.


## Evaluation

The following script will run all the steps from creating BERT embeddings and running predictions using them, to calculating the scores for the evaluation metrics.
```
./split_eval.sh 10 dev false 24 2
```
Here, the evaluation is performed with 10 splits/shards (which reduces the maximum storage usage by 10x), for the dev set, for the distractor setting (the usage of fullwiki is set to false). The prediction batchsize during prediction is set to 24, and the batchsize for getting BERT embeddings is set to 2.

