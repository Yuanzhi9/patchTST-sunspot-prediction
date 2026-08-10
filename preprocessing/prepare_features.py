import pandas as pd
import numpy as np

# 读取原数据
df = pd.read_csv('./dataset/sunspot_monthly_clean.csv')

# 添加月份正余弦
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# 可选：年份归一化（防止数值太大）
df['year_norm'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())

# 保存新文件
df.to_csv('./dataset/sunspot_with_features.csv', index=False)
print("新数据已保存，包含列：", df.columns.tolist())