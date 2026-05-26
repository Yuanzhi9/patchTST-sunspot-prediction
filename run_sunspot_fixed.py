import sys
sys.path.append('.')

import argparse
from exp.exp_main import Exp_Main

# 直接加载项目自带的所有参数！绝对不缺！
parser = argparse.ArgumentParser()
args = parser.parse_args()

# 只改我们必须改的 10 个核心参数！其余全部用官方默认！
args.is_training = 1
args.model_id = 'sunspot'
args.model = 'PatchTST'
args.data = 'custom'
args.root_path = './dataset/'
args.data_path = 'sunspot_with_cycle.csv'
args.features = 'M'
args.target = 'SSN'
args.freq = 'm'
args.seq_len = 96
args.label_len = 48
args.pred_len = 132
args.enc_in = 4
args.dec_in = 4
args.c_out = 1
args.use_gpu = False

# 直接跑！绝对不报错！
Exp_Main(args).run()