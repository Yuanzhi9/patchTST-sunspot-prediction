import pandas as pd
import numpy as np

# 1. 读取现有的特征数据（有 sin, cos 那个）
df = pd.read_csv('./dataset/sunspot_with_features.csv')

# 2. 按时间排序（确保顺序正确）
df = df.sort_values(['year', 'month'])

# 3. 计算滞后特征
df['ssn_lag1'] = df['ssn'].shift(1)    # 上个月
df['ssn_lag12'] = df['ssn'].shift(12)  # 12个月前

# 4. 删除前12行（因为 lag12 是空的）
df_clean = df.dropna().reset_index(drop=True)

# 5. 另存为新文件（不覆盖原文件！）
df_clean.to_csv('./dataset/sunspot_with_lags.csv', index=False)

print(f"原数据行数: {len(df)}")
print(f"新数据行数: {len(df_clean)}（删除了前12行）")
print("新文件已保存: sunspot_with_lags.csv")
print("包含列:", df_clean.columns.tolist())