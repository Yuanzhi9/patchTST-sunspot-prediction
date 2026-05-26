import pandas as pd

# 1. 读取原特征数据
df = pd.read_csv('./dataset/sunspot_with_features.csv')

# 2. 删除前12行
df_trimmed = df.iloc[12:].reset_index(drop=True)

# 3. 另存为新文件
df_trimmed.to_csv('./dataset/sunspot_features_trimmed.csv', index=False)

print(f"原数据行数: {len(df)}")
print(f"新数据行数: {len(df_trimmed)}（删除了前12行）")
print("新文件已保存: sunspot_features_trimmed.csv")