import sys
sys.path.append('.')

import argparse
from exp.exp_main import Exp_Main

parser = argparse.ArgumentParser(description='PatchTST for Sunspot Prediction')

# ---- basic config ----
parser.add_argument('--random_seed', type=int, default=2021)
parser.add_argument('--is_training', type=int, default=1)
parser.add_argument('--model_id', type=str, default='sunspot')
parser.add_argument('--model', type=str, default='PatchTST')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

# ---- data loader ----
parser.add_argument('--data', type=str, default='custom')
parser.add_argument('--root_path', type=str, default='./dataset/')
parser.add_argument('--data_path', type=str, default='sunspot_with_cycle.csv')
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--target', type=str, default='ssn')
parser.add_argument('--freq', type=str, default='m')

# ---- forecasting task ----
parser.add_argument('--seq_len', type=int, default=96)
parser.add_argument('--label_len', type=int, default=48)
parser.add_argument('--pred_len', type=int, default=24)

# ---- PatchTST ----
parser.add_argument('--fc_dropout', type=float, default=0.05)
parser.add_argument('--head_dropout', type=float, default=0.0)
parser.add_argument('--patch_len', type=int, default=16)
parser.add_argument('--stride', type=int, default=8)
parser.add_argument('--padding_patch', default='end')
parser.add_argument('--revin', type=int, default=1)
parser.add_argument('--affine', type=int, default=0)
parser.add_argument('--subtract_last', type=int, default=0)
parser.add_argument('--decomposition', type=int, default=0)
parser.add_argument('--kernel_size', type=int, default=25)
parser.add_argument('--individual', type=int, default=0)

# ---- Transformer ----
parser.add_argument('--embed_type', type=int, default=0)
parser.add_argument('--enc_in', type=int, default=3)
parser.add_argument('--dec_in', type=int, default=3)
parser.add_argument('--c_out', type=int, default=1)
parser.add_argument('--d_model', type=int, default=512)
parser.add_argument('--n_heads', type=int, default=8)
parser.add_argument('--e_layers', type=int, default=2)
parser.add_argument('--d_layers', type=int, default=1)
parser.add_argument('--d_ff', type=int, default=2048)
parser.add_argument('--moving_avg', type=int, default=25)
parser.add_argument('--factor', type=int, default=1)
parser.add_argument('--distil', action='store_false', default=True)
parser.add_argument('--dropout', type=float, default=0.05)
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--activation', type=str, default='gelu')
parser.add_argument('--output_attention', action='store_true', default=False)
parser.add_argument('--do_predict', action='store_true', default=False)

# ---- optimization ----
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--itr', type=int, default=1)
parser.add_argument('--train_epochs', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--des', type=str, default='test')
parser.add_argument('--loss', type=str, default='mse')
parser.add_argument('--lradj', type=str, default='type3')
parser.add_argument('--pct_start', type=float, default=0.3)
parser.add_argument('--use_amp', action='store_true', default=False)

# ---- GPU ----
parser.add_argument('--use_gpu', type=bool, default=False)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--use_multi_gpu', action='store_true', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3')
parser.add_argument('--test_flop', action='store_true', default=False)

args = parser.parse_args()

# Override to force CPU
args.use_gpu = False

import torch
import random
import numpy as np

fix_seed = args.random_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

print('Args in experiment:')
print(args)

exp = Exp_Main(args)

setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
    args.model_id,
    args.model,
    args.data,
    args.features,
    args.seq_len,
    args.label_len,
    args.pred_len,
    args.d_model,
    args.n_heads,
    args.e_layers,
    args.d_layers,
    args.d_ff,
    args.factor,
    args.embed,
    args.distil,
    args.des, 0)

print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
exp.train(setting)

print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
exp.test(setting)
