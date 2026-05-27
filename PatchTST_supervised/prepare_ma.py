import pandas as pd
import numpy as np

# 1. 读取基础数据（用最原始的有sin/cos的版本）
df = pd.read_csv('./dataset/sunspot_with_features.csv')

# 2. 按时间排序
df = df.sort_values(['year', 'month'])

# 3. 计算移动平均
df['ssn_ma3'] = df['ssn'].rolling(window=3, min_periods=1).mean()
df['ssn_ma12'] = df['ssn'].rolling(window=12, min_periods=1).mean()

# 4. 保存新文件
df.to_csv('./dataset/sunspot_with_ma.csv', index=False)

print("移动平均特征已添加")
print("新文件: sunspot_with_ma.csv")
print("包含列:", df.columns.tolist())