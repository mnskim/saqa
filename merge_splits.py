import os
import json
#import ipdb
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str)
parser.add_argument('splits', type=int, default=5)
args = parser.parse_args()


merged_pred = {'sp': {}, 'answer': {}}
for i in range(args.splits):
    split_pred = json.load(open(os.path.join(args.dir, 'split{}_pred.json'.format(i)), 'r'))
    #ipdb.set_trace()
    merged_pred['sp'] = dict(list(merged_pred['sp'].items()) + list(split_pred['sp'].items()))
    merged_pred['answer'] = dict(list(merged_pred['answer'].items()) + list(split_pred['answer'].items()))
#ipdb.set_trace()
json.dump(merged_pred, open(os.path.join(args.dir, 'merged_pred.json'),'w'))
json.dump(merged_pred, open('./pred.json','w'))

