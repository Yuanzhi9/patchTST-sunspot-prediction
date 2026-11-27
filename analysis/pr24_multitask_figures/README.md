# PR24 多任务曲线图

所有图均为 SC20/SC21/SC24 时间顺序盲测，五个 seed。

- 黑线：真实月均 SSN；
- 蓝线：PR24 baseline 五 seed 中位数；
- 橙线：多任务模型五 seed 中位数；
- 阴影：五 seed 范围；
- `recursive`：无未来真值、block24 递归 132 月；
- `update`：每 24 月用真实观测更新的 132 月曲线。

每个周期、每种推理方式单独出 PNG/PDF。

## 固定权重 w=0.1

`fixed_v2/`

## 不确定度权重

`unc_v2/`

## 固定权重 w=0.3

`w030_v2/`

## 真实未来预测与峰值修正

目录：

```text
analysis/pr24_multitask_future_correction/
```

图：

```text
mt_real_forecast_raw_and_13m.png
old_profiles_with_R1A_R1B.png
mt_profiles_with_R1A_R1B.png
```
