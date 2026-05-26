import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免GUI问题

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ===================== 1. 全局配置：解决中文乱码 =====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.size'] = 10  # 统一字体大小
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['figure.dpi'] = 100

# ===================== 2. 配置参数（避免硬编码） =====================
class Config:
    # 路径配置
    pred_path = './results/Sunspot_PatchTST_MS_PatchTST_custom_ftMS_sl132_ll66_pl24_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0/pred.npy'
    data_path = './dataset/sunspot_with_cycle.csv'
    save_fig_path = 'sunspot_prediction_result.png'
    save_npy_path = './'
    
    # 模型与数据参数
    seq_len = 132
    pred_len = 24
    train_end_idx = 2520  # 训练集结束索引（用于反归一化）
    border1 = 3057        # 测试集起始索引
    border2 = 3321        # 测试集结束索引

config = Config()

# ===================== 3. 数据加载与校验 =====================
print("="*60)
print("步骤1/6: 加载数据")
print("="*60)

# 检查文件是否存在
if not os.path.exists(config.pred_path):
    raise FileNotFoundError(f"预测文件不存在: {config.pred_path}")
if not os.path.exists(config.data_path):
    raise FileNotFoundError(f"数据文件不存在: {config.data_path}")

# 加载预测值
pred = np.load(config.pred_path)
print(f"预测值原始形状: {pred.shape}")  # (n_samples, pred_len, 1)
assert len(pred.shape) == 3, "预测值形状错误，应为 (样本数, 预测步长, 变量数)"
assert pred.shape[2] == 1, "仅支持单变量预测"

pred_ssn = pred[:, :, 0]  # 提取太阳黑子数预测
print(f"预测值（太阳黑子数）形状: {pred_ssn.shape}")

# 加载真实数据
df = pd.read_csv(config.data_path)
print(f"数据总行数: {len(df)}")
assert 'ssn' in df.columns, "数据文件中缺少'ssn'列"

# 提取测试集真实值
test_ssn_all = df['ssn'].iloc[config.border1:config.border2].values
print(f"测试集长度: {len(test_ssn_all)}")
print(f"测试集ssn范围: {test_ssn_all.min():.1f} ~ {test_ssn_all.max():.1f}")

# 构建滑动窗口的真实值（与预测一一对应）
true_ssn_windows = []
n_samples = pred_ssn.shape[0]
valid_samples = 0

for i in range(n_samples):
    start = i + config.seq_len
    end = start + config.pred_len
    if end <= len(test_ssn_all):
        true_ssn_windows.append(test_ssn_all[start:end])
        valid_samples += 1
    else:
        print(f"警告: 第{i}个窗口超出测试集范围，已跳过")

true_ssn = np.array(true_ssn_windows)
pred_ssn = pred_ssn[:valid_samples]  # 对齐有效样本数
print(f"有效样本数: {valid_samples}")
print(f"最终预测值形状: {pred_ssn.shape}, 真实值形状: {true_ssn.shape}")

# ===================== 4. 反归一化 =====================
print("\n步骤2/6: 反归一化（使用训练集统计量）")
train_data = df['ssn'].iloc[:config.train_end_idx].values
train_mean = train_data.mean()
train_std = train_data.std()
print(f"训练集ssn统计量: 均值={train_mean:.2f}, 标准差={train_std:.2f}")

pred_original = pred_ssn * train_std + train_mean
true_original = true_ssn  # 真实值未归一化，直接使用

print(f"反归一化后:")
print(f"预测值范围: {pred_original.min():.1f} ~ {pred_original.max():.1f}")
print(f"真实值范围: {true_original.min():.1f} ~ {true_original.max():.1f}")

# ===================== 5. 误差指标计算 =====================
print("\n步骤3/6: 计算误差指标")
errors = np.abs(pred_original - true_original)
mae = np.mean(errors)
rmse = np.sqrt(np.mean((pred_original - true_original)**2))

# 每个样本的MAE
sample_mae = np.mean(errors, axis=1)
# 每个预测步的MAE
step_mae = np.mean(errors, axis=0)

print(f"整体MAE: {mae:.2f}")
print(f"整体RMSE: {rmse:.2f}")
print(f"样本MAE范围: {sample_mae.min():.2f} ~ {sample_mae.max():.2f}")
print(f"预测步MAE范围: {step_mae.min():.2f} ~ {step_mae.max():.2f}")

# ===================== 6. 可视化绘图 =====================
print("\n步骤4/6: 生成可视化图表")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
x_axis = np.arange(1, config.pred_len + 1)  # 预测步数（1~24）

# -------------------- 图1: 单样本预测 vs 真实 --------------------
ax1 = axes[0, 0]
sample_idx = 0
true_line, = ax1.plot(x_axis, true_original[sample_idx], 'b-', linewidth=2, label='真实值')
pred_line, = ax1.plot(x_axis, pred_original[sample_idx], 'r--', linewidth=2, label='预测值')
ax1.set_xlabel('预测步数（月）')
ax1.set_ylabel('太阳黑子数')
ax1.set_title(f'样本 {sample_idx+1}: 预测 vs 真实')
ax1.legend(handles=[true_line, pred_line])

# -------------------- 图2: 平均预测 vs 平均真实（±1标准差） --------------------
ax2 = axes[0, 1]
mean_true = np.mean(true_original, axis=0)
std_true = np.std(true_original, axis=0)
mean_pred = np.mean(pred_original, axis=0)
std_pred = np.std(pred_original, axis=0)

# 真实值：蓝色实线+淡蓝色填充
ax2.plot(x_axis, mean_true, 'b-', linewidth=2, label='真实值（均值）')
ax2.fill_between(x_axis, mean_true - std_true, mean_true + std_true, color='b', alpha=0.2, label='真实值±1标准差')
# 预测值：红色虚线+淡红色填充
ax2.plot(x_axis, mean_pred, 'r--', linewidth=2, label='预测值（均值）')
ax2.fill_between(x_axis, mean_pred - std_pred, mean_pred + std_pred, color='r', alpha=0.2, label='预测值±1标准差')

ax2.set_xlabel('预测步数（月）')
ax2.set_ylabel('太阳黑子数')
ax2.set_title('平均预测 vs 平均真实（±1标准差）')
ax2.legend(loc='upper left')

# -------------------- 图3: 预测值 vs 真实值散点图 --------------------
ax3 = axes[1, 0]
all_pred = pred_original.flatten()
all_true = true_original.flatten()

ax3.scatter(all_true, all_pred, alpha=0.3, s=15, color='#1f77b4', label='样本点')
# 理想线（y=x）
min_val = min(all_true.min(), all_pred.min())
max_val = max(all_true.max(), all_pred.max())
ax3.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label='理想线（预测=真实）')

ax3.set_xlabel('真实值')
ax3.set_ylabel('预测值')
ax3.set_title(f'预测值 vs 真实值散点图 (整体MAE={mae:.2f})')
ax3.legend()

# -------------------- 图4: 各预测步的MAE --------------------
ax4 = axes[1, 1]
bars = ax4.bar(x_axis, step_mae, width=0.7, color='#1f77b4')
# 在柱子上方标注数值（可选）
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}', ha='center', va='bottom', fontsize=8)

ax4.set_xlabel('预测步数（月）')
ax4.set_ylabel('MAE')
ax4.set_title('各预测步的平均绝对误差')
ax4.set_xticks(x_axis[::2])  # 每2步显示一个刻度，避免拥挤

# 调整布局并保存
plt.tight_layout()
plt.savefig(config.save_fig_path, dpi=150, bbox_inches='tight')
print(f"图表已保存至: {config.save_fig_path}")

# ===================== 7. 保存结果并输出汇总 =====================
print("\n步骤5/6: 保存结果文件")
np.save(os.path.join(config.save_npy_path, 'pred_original.npy'), pred_original)
np.save(os.path.join(config.save_npy_path, 'true_original.npy'), true_original)

print("\n" + "="*60)
print("最终结果汇总")
print("="*60)
print(f"测试样本数: {valid_samples}")
print(f"预测长度: {config.pred_len} 个月")
print(f"整体MAE: {mae:.2f}")
print(f"整体RMSE: {rmse:.2f}")
print(f"预测值范围: {pred_original.min():.1f} ~ {pred_original.max():.1f}")
print(f"真实值范围: {true_original.min():.1f} ~ {true_original.max():.1f}")
print("="*60)