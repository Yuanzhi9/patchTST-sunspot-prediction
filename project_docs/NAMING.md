# 预测任务与配置命名

## 规则

- `PR<h>`：prediction length，例如 `PR24`、`PR132`。
- `SL<n>`：sequence length，例如 `SL132`。
- `PL<n>`：patch length，例如 `PL36`。
- `S<n>`：patch stride，例如 `S18`。
- `BS<n>`：batch size。
- `DO<n>`：dropout。

配置 ID 示例：

```text
PR24-SL132-PL36-S18
PR132-SL132-PL36-S18
```

## 已验收候选的规范名称

| 规范名称 | 历史实验别名 | 含义 |
|---|---|---|
| `PR24-SL132-PL36-S18-BS32-DO02` | `BASE-2026-11-M-CAND-P36` / `B26M-B-P36-BS32` | `pred_len=24` |
| `PR132-SL132-PL36-S18-BS32-DO03` | `BASE-2026-11-M-P132-CAND-P36` / `B26M-P132-S132-P36-LR1E3-DO03` | `pred_len=132` |

## 约束

- 历史实验目录、配置文件名和结果路径保持原样，确保已有证据可复现。
- 新文档和新实验使用规范名称；需要引用旧实验结果时同时写历史别名。
- 不再单写 `P24`、`P36`。`P36` 必须写成 `PL36`，避免与 `PR36` 或其他任务 ID 混淆。
