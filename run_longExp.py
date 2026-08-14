import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # random seed
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Informer, Transformer]')

    # data loader（改路径
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='sunspot_with_cycle.csv', help='data file')
    parser.add_argument('--features', type=str, default='MS',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='ssn', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='m',    #2026.05.01h改为m
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    # [2026-08-12 景修] 新增：支持多回测窗口按年月切分。不传时保持原硬编码切分（向后兼容）
    # 改前：无这两个参数，data_loader.py 内 num_train/num_val/num_test 硬编码行数
    # 改后：传 --test_start / --test_end 时，data_loader.py 按年月过滤行（train_end = test_start − 133月，val 固定 132 月）
    parser.add_argument('--test_start', type=str, default='', help='测试集起始年月(YYYY-MM)，如 1996-08。留空=原始硬编码切分')
    parser.add_argument('--test_end', type=str, default='', help='测试集截止年月(YYYY-MM)，如 2008-11。留空=原始硬编码切分')
    # [2026-08-15 景修] 新增：目标变换（sqrt 压缩右偏，探索期 EXP-20-4 用）。默认 '' = 不变换
    parser.add_argument('--target_transform', type=str, default='', help='目标变换: 空/sqrt。sqrt=对SSN取平方根, 评估侧自动平方回物理空间')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task（要重点调）
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')


    # DLinear（线性层，可以压缩或放大）
    #parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

    # PatchTST
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')#（临时随机屏蔽，防止过拟合）
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')#（不启用随机丢弃注意力头，atuoformor用的是注意力）
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')   #（切成小段-patch就是那些小段，每个小段的长度）
    parser.add_argument('--stride', type=int, default=8, help='stride')  #步长-滑动的窗口位置间隔;高斯滤波的原理和权重有关，靠近轴的权重大
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')  #padding是填充
    parser.add_argument('--revin', type=int, default=0, help='RevIN; True 1 False 0')  #可逆归一化 时序模型标配（前向归一化，后向反归一化，还没懂）
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')#时序预测避开
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')#作用是分解，但是已经分解了，不用
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')## 在每个patch内用3大小的卷积核提取特征，不懂卷积核
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')#共享头部：所有变量（特征）使用同一个预测头

    # Formers 
    parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--enc_in', type=int, default=3, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels    #变量数为7，可以改    #04.02改成5    05.01改为3
    parser.add_argument('--dec_in', type=int, default=5, help='decoder input size')#解码器部分，不懂，要看看   #04.02改成4
    parser.add_argument('--c_out', type=int, default=1, help='output size')#模型输出预测目标数     04.02从7改成1
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')#数据内部特征长度
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')#前馈网络中间维度
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')#注意力因子
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)#蒸馏
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization参数优化
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')#子进程数
    parser.add_argument('--itr', type=int, default=2, help='experiments times')#重复跑几轮；实验次数
    parser.add_argument('--train_epochs', type=int, default=5, help='train epochs')##要跑的轮数原来是100
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')#2026.04.09修改为huber   2026.05.01改回mse
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=False, help='use gpu')#用gpu时改成true
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    args = parser.parse_args()

    # random seed可控的随机化
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)


    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    # auto-dump 实际参数快照（防御性副本，与 save_config.py 互为验证）
    try:
        import json
        from datetime import datetime
        os.makedirs('configs', exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        auto_path = os.path.join('configs', f'auto_{ts}.json')
        args_dict = vars(args).copy()
        # numpy/pandas types → native Python for JSON
        for k, v in args_dict.items():
            if hasattr(v, 'item'):
                args_dict[k] = v.item()
        with open(auto_path, 'w') as f:
            json.dump(args_dict, f, indent=2, ensure_ascii=False, default=str)
    except Exception:
        pass  # 存档失败不影响训练

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
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
                args.des,ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
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
                                                                                                    args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
        