"""
训练 PatchTST 模型预测太阳黑子数
=====================================
正确的参数配置记录
"""

import subprocess
import os

# 切换到项目目录
os.chdir(r'F:\Downloads\patchTST_main\PatchTST-main\PatchTST_supervised')

# 训练命令
cmd = [
    'python', 'run_longExp.py',
    '--is_training', '1',
    '--model_id', 'Sunspot_PatchTST_MS',
    '--model', 'PatchTST',
    '--data', 'custom',
    '--root_path', './dataset',
    '--data_path', 'sunspot_with_cycle.csv',
    '--target', 'ssn',
    '--features', 'MS',
    '--enc_in', '5',
    '--dec_in', '5',
    '--c_out', '1',
    '--seq_len', '132',
    '--label_len', '66',
    '--pred_len', '24',
    '--patch_len', '12',
    '--stride', '6',
    '--revin', '1',
    '--individual', '0',
    '--train_epochs', '50',
    '--batch_size', '32',
    '--learning_rate', '0.0001',
    '--patience', '10',
    '--use_gpu', 'False',
    '--itr', '1'
]

print("开始训练...")
print("命令:", ' '.join(cmd))
subprocess.run(cmd)