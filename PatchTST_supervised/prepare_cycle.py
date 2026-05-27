import pandas as pd
import numpy as np

# 1. 读取现有的特征数据
df = pd.read_csv('./dataset/sunspot_with_features.csv')

# 2. 计算11年周期相位
# 以1749年为起点（你的数据从1749年开始）
years_since_start = df['year'] - 1749
df['cycle_phase'] = np.sin(2 * np.pi * years_since_start / 11)

# 3. 另存为新文件
df.to_csv('./dataset/sunspot_with_cycle.csv', index=False)

print(f"周期相位已添加")
print("新文件: sunspot_with_cycle.csv")
print("包含列:", df.columns.tolist())