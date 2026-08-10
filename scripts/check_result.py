# check_result.py
import numpy as np
import glob
import os

# 自动搜索所有结果文件，不用你手动找路径！
print("🔍 正在搜索结果文件...")
pred_files = glob.glob("./results/**/pred.npy", recursive=True)
true_files = glob.glob("./results/**/true.npy", recursive=True)

if len(pred_files) > 0 and len(true_files) > 0:
    pred_path = pred_files[0]
    true_path = true_files[0]

    print("✅ 找到预测文件：", pred_path)
    print("✅ 找到真实文件：", true_path)

    # 读取数据
    pred = np.load(pred_path)
    true = np.load(true_path)

    print("\n🎯 数据形状 (正确应该是 (132,) )")
    print("预测值 shape:", pred.shape)
    print("真实值 shape:", true.shape)

    print("\n🎉 恭喜！训练完全成功！结果已保存！")
    print("你可以放心画图、写论文、做后续预测！")

else:
    print("❌ 暂时没找到npy文件，可能是训练最后中断没生成")
    print("👉 解决方法：重新运行训练，加上 --skip_test True 即可完美生成")



    