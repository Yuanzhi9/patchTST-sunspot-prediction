# T1 v1 中间结果作废说明

该目录中的 run 是 T1-v2 选模修正前的设计诊断，不进入结论。

原因：验证损失最低点出现在 scheduled sampling 启动之前，导致 teacher 与
scheduled 选中同一个干预前 checkpoint。SC20 两模式测试曲线逐位相同。

正式实验只引用：

- `analysis/t1_scheduled_sampling_v2/`
- `logs/B26M-T1V2-*.log`
