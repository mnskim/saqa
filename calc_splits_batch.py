import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('length', type=int, default=7405)
parser.add_argument('splits', type=int, default=5)
parser.add_argument('batch_size', type=int, default=24)
parser.add_argument('print_opt', type=str)
args = parser.parse_args()

def split(n_examples, n_splits, batch_size, print_opt):
    if n_splits == 0:
        raise NotImplementedError
    elif n_splits == 11:
        if print_opt == 'start':
            print_out = '0'
        elif print_opt == 'end':
            print_out = str(n_examples)
    else:
        total_batches, batch_remainder = divmod(n_examples, batch_size)
        total_splits, split_remainder = divmod(total_batches, n_splits)
        split_sizes = np.cumsum([total_splits*batch_size for split in range(n_splits)])
        split_sizes = split_sizes.tolist()
        assert n_examples == split_sizes[-1] + batch_size*split_remainder+batch_remainder
        if split_sizes[-1] == n_examples:
            pass
        else:
            split_sizes[-1] = n_examples
        split_sizes = map(str, split_sizes)
        if print_opt == 'start':
            split_sizes = list(split_sizes)[:-1]
            print_out = '0 ' + ' '.join(split_sizes)
        elif print_opt == 'end':
            print_out = ' '.join(split_sizes)
    print(print_out)
split(args.length, args.splits, args.batch_size, print_opt=args.print_opt)
