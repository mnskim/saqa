import os
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('data_split', type=str, choices=['train', 'dev', 'test'])
parser.add_argument('save_path', type=str)
parser.add_argument('--fullwiki', action='store_true')
# Choose one of two
config = parser.parse_args()

if not os.path.exists(config.save_path):
    os.makedirs(config.save_path)

record_path = '{}_record.pkl'.format(config.data_split)

def _concat(filename):
    if config.fullwiki:
        return 'fullwiki.{}'.format(filename)
    return filename

record_path = _concat(record_path)

rec = torch.load(record_path)
with open(os.path.join(config.save_path, 'ids.txt'), 'w') as out:
    for item in rec:
        out.write(item['id'] + '\n')
print(len(rec))

