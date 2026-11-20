# PR132 改进实验 backlog

> 状态：想法库，不进入当前标准基线搜索。  
> 规则：基线冻结后，每个系列只改一个主因素，并同时报告 curve MAE、最差折、峰谷幅度/时间和预测振幅比。

## B：标准基线搜索

- [ ] B0 预登记标准 PatchTST 搜索空间和停止规则
- [ ] B1 smoke：`SL264`、小模型、小 batch 各一个 fold
- [ ] B2 96 组随机/TPE 搜索，固定 MSE、三特征、direct132，只用开发折
- [ ] B3 前10配置补三 seed
- [ ] B4 前3配置补五 seed
- [ ] B5 最终候选做一次 deferred-fold 审计
- [ ] B6 冻结 PR132 standard baseline

## I1：架构对照

- [ ] Ridge 多输出，同折同输入
- [ ] DLinear，同折同输入
- [ ] PatchTST 小数据配方：`D16/H4/F128`
- [ ] 多尺度 patch encoder
- [ ] 局部 patch + 全局 cycle token
- [ ] TimeXer/TFT/N-BEATSx 作为外源融合架构

## I2：loss 与目标

- [ ] MAE
- [ ] Huber
- [ ] 高值加权 MSE
- [ ] 阈值加权 MSE
- [ ] 低估非对称损失
- [ ] 多分位数损失
- [ ] 曲线 + 峰值 + 峰时 + 周期长度多任务损失
- [ ] 一阶差分/形状损失
- [ ] sqrt/log1p/pow 目标变换

## I3：相位与周期级变量

- [ ] 使用官方已确认极小值的 point-in-time 相位
- [ ] 因果 Hilbert 相位
- [ ] `months_since_last_confirmed_min`
- [ ] 上一周期峰值、长度、偏度、aa max
- [ ] 当前周期迄今最大 SSN、累计活动量
- [ ] cycle-aligned 起点
- [ ] 峰值、峰时、谷时、周期长度辅助头

## I4：外部物理量

- [ ] aa：1868 起独立数据集与 fold spec
- [ ] aa 早期 patch 融合
- [ ] aa cross-attention
- [ ] aa + 周期级变量联合融合
- [ ] F10.7：1947 起独立小样本实验
- [ ] 北/南半球 SSN 或黑子面积
- [ ] 极区磁场仅在覆盖允许时评估

## I5：物理轮廓与残差

- [ ] Waldmeier/HWR 峰值 + PatchTST 归一化轮廓
- [ ] PatchTST 预测 HWR 参数
- [ ] HWR 轮廓 + 学习残差
- [ ] 可微 HWR 多任务头
- [ ] 双峰/非对称扩展
- [ ] 平滑主曲线 + 高频残差（数据口径另开）

## I6：数据与训练策略

- [ ] PR132-update12 正式评价
- [ ] PR132-update24 正式评价
- [ ] update cadence 与 fixed-lead 分层评价
- [ ] 按周期相位分层采样
- [ ] 窗口重叠降权
- [ ] 物理约束的时间伸缩/幅度缩放
- [ ] 合成周期 masked pretraining
- [ ] 合成周期整周期预训练
- [ ] 多代理量迁移学习

## I7：评价与治理

- [ ] 峰谷匹配加入最大时间距离和未匹配惩罚
- [ ] nested CV 超参数选择
- [ ] 严格 rolling-origin 真实预报图
- [ ] 未来新数据审计
- [ ] 冻结标准基线与所有改进对照
