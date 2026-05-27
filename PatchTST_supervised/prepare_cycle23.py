import pandas as pd
import numpy as np

# 1. 读取带特征的数据
df = pd.read_csv('./dataset/sunspot_with_cycle.csv')
df['date'] = pd.to_datetime(df['date'])

# 2. 划分时间点
train_end = pd.to_datetime('1996-08-01')
test_start = pd.to_datetime('1996-08-01')
test_end = pd.to_datetime('2008-12-01')

# 3. 训练集：1996年8月之前
train_df = df[df['date'] < train_end].copy()

# 4. 测试集：第23周期（1996.8 - 2008.12）
test_df = df[(df['date'] >= test_start) & (df['date'] <= test_end)].copy()

print(f"训练集: {train_df['date'].min()} 至 {train_df['date'].max()}, {len(train_df)} 个月")
print(f"测试集(第23周期): {test_df['date'].min()} 至 {test_df['date'].max()}, {len(test_df)} 个月")

# 5. 保存新文件
train_df.to_csv('./dataset/sunspot_train_before_1996.csv', index=False)
test_df.to_csv('./dataset/sunspot_test_cycle23.csv', index=False)

print("✅ 数据已保存")