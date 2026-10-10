# EXP-25 — 新环境基线重建（A6000 / torch 2.8）

> ⚠️ **本记录文件的预检部分由 AI（Codex）代笔填写，景修本人未逐项确认。**
> 原因：景修 2026-10-10 明确授权「预检表你写就好了，标注清楚是你写的，留行给我写」。
> 按 SOP §1 铁律，未经景修确认的预检表**不构成正式跑前门禁**，明早由景修补签或否决。

---

## 基本信息

| 字段 | 内容 |
|------|------|
| 日期 | 2026-10-10（夜） |
| 实验ID | EXP-25 |
| 分支 / worktree | `EXP-25-baseline-newenv` / `/output/patchTST-EXP25` |
| 类型 | 训练实验（**环境重基线**，非单变量对照） |
| 目的 | 在换机后（torch 2.8 + A6000 GPU）重建基线锚点，供后续所有实验对比 |
| 改动变量 | **无**。本实验不是对照实验，是重建锚点。相对"历史环境"改了环境（CPU→GPU、torch 1.11→2.8），相对"配置"沿用 SOP §4 标准 ARGS |
| 对照基线 | 无（本次即基线本体）。历史参考值：EXP-14 sl96 → 全步 23.87 / step0 9.08；探索期 0b（sl336）全步 23.87 |
| 训练入口 | `run_longExp.py`（原管线），经 `scripts/launch_experiment.sh` 启动，命令链 train && 补测 |
| features | MS |
| 前提 | mode=MS；数据 `sunspot_with_cycle.csv`（1749-2025，月度）；归一化 StandardScaler；W1 窗口 test_start=1996-08 / test_end=2008-11（num_train=2838, num_test=148） |
| **坑清单回顾（必填）** | 本实验可能踩的坑：**13**（并行 OOM → 串行+PID 锁）、**14**（输出缓冲 → `-u`）、**15**（断连杀进程 → nohup+setsid，launch 脚本已含）、**16**（口径混比 → 统一命令链）、**18**（证据链断裂 → commit 前 `git status` 核对）、**19**（不落盘 → 当天闭环）、**27**（顶部快照不更新）、**28**（不回顾历史坑）。**本次新识别 3 个新环境坑**：Ⓐ torch≥2.6 `torch.load` 默认 `weights_only=True` 会因 checkpoint 内 `argparse.Namespace` 抛 UnpicklingError（已修，见下）Ⓑ Ampere 架构 TF32 默认开启会降低数值精度（已用 `NVIDIA_TF32_OVERRIDE=0` 关闭）Ⓒ `run_longExp.py` 的 `--use_gpu` 是 `type=bool`，命令行传 `--use_gpu 0` 也会得到 True（本次反而需要 True，故不改源码，仅记录） |
| **单种子限制** | 本结果基于 seed=2021 单次训练。无重复标准差/置信区间，3% 边缘差异不可靠。 |

---

## 可证伪假设（AI 代填）

- **如果有效**：命令链完整跑完（train 50ep + 补测），`result.txt` 出现两行 z-score，`results/` 下生成 pred.npy/true.npy，四指标（step0 MAE / 全步 MAE / E_r / E_m）齐全且物理值在合理量级（全步 MAE 落在 15~35 区间，不出现 1e3 量级或 NaN）。
- **如果无效**：训练或补测阶段报错中止 → `&&` 短路 → 补测 npy 不生成 → 本次无有效结果。
- **如果结果是"四指标齐全但与历史差异 >30%"** → 推论：环境（设备精度/算子实现）不是数值噪音级变量，而是需要显式记录并单独论证的变量；后续所有实验都必须注明运行环境。

## 预尸检（AI 代填）

如果两周后证明这次重基线是废物，最可能因为什么？

1. **配置选错**：我取的是 SOP §4 的标准 ARGS（sl336/pl24/50ep/MS/W1）。若景修心中的"当前基线"其实是另一套（文档里曾有四套说法），那这一晚跑的不是她要的锚。→ 这是本次最大风险，**明早第一件事请景修确认配置**。
2. **窗口/scaler 锚点错位**：`--num_train` 与 W1 的 2838 不一致会让物理值整体作废。
3. **GPU 精度引入不可控差异**：TF32 已关，但 cuDNN 算子实现仍与 CPU 不同。
4. **权重无历史可对拍**：本机无旧 `.pth`（权重在景修电脑），无法用已知 checkpoint 做锚点回归，只能靠"量级合理"判断。

## 文献先例（AI 代填）

不适用——本次是环境重基线，不是方法实验，无对应文献方向。按 SOP §1，训练管线与物理特征注入的文献背景见 `literature/literature_reading_notes.md`（本项目已有 14 篇精读）。

---

## 参数对照表（对照 SOP §4 标准 ARGS）

| 参数 | SOP §4 标准值 | 本次值 | 一致？ |
|------|--------------|--------|--------|
| 训练入口 | `run_longExp.py` | 同 | ✅ |
| data_path | `sunspot_with_cycle.csv` | 同 | ✅ |
| root_path | `./PatchTST_supervised/dataset/` | 同 | ✅ |
| features / target / enc_in | MS / ssn / 3 | 同 | ✅ |
| test_start / test_end | 1996-08 / 2008-11 | 同 | ✅ |
| seq_len / label_len / pred_len | 336 / 48 / 24 | 同 | ✅ |
| d_model / n_heads / e_layers / d_ff | 128 / 8 / 2 / 2048 | 同 | ✅ |
| patch_len / stride | 16 / 8 | 同 | ✅ |
| revin / affine / individual | 1 / 0 / 0 | 同 | ✅ |
| dropout / fc_dropout / head_dropout | 0.05 / 0.05 / 0.0 | 同 | ✅ |
| train_epochs / patience | 50 / 100 | 同 | ✅ |
| batch_size / learning_rate | 16 / 0.0001 | 同 | ✅ |
| loss / activation / random_seed | mse / gelu / 2021 | 同 | ✅ |
| num_workers / itr | 0 / 1 | 同 | ✅ |
| **运行环境** | Python 3.10.12 + torch 1.11 + 无 GPU | **Python 3.10.18 + torch 2.8.0+cu128 + A6000** | ❌ 本次唯一变量 |
| **TF32** | 不存在（CPU） | `NVIDIA_TF32_OVERRIDE=0` | 新增（为贴近全精度） |

## 代码改动（环境兼容，非实验变量）

改动**对所有实验生效**，保证基线与其后实验可比：

| 文件 | 改动 | 理由 |
|------|------|------|
| `PatchTST_supervised/exp/exp_main.py` | 新增 `_load_checkpoint()`，`test()` 内改用它 | torch≥2.6 默认 `weights_only=True`，checkpoint 内 `argparse.Namespace` 不在白名单 → UnpicklingError。优先 `add_safe_globals`（保留防护），失败才回退 `weights_only=False`；旧 torch 走原路径 |
| `roll_eval.py` | `load_model()` 同样处理 | 同一个坑 |

未改 `run_longExp.py` 源码（遵守 SOP §7「不直接改源码适配实验」）；GPU 经 CLI `--use_gpu 1` 开启。

---

## 成功标准（AI 代填，待景修确认）

- 命令链完整执行，`result.txt` 新增两行；正式引用**补测段**（最佳 val 模型口径）
- 四指标齐全：step0 MAE / 全步 MAE / E_r / E_m
- 物理量级合理（全步 MAE 15~35，无 NaN / 无 1e3 量级）
- 落盘五件证据链：checkpoint + npy + result.txt + eval 结果 + 本记录文件
- 基线实验（本类）按 SOP §5 **必须跑 roll_eval.py**

---

## 景修批注（留空，请手写）

> 配置确认（SOP §4 ARGS 是否即你认可的"当前基线"）：
>
>
> 成功标准是否认可：
>
>
> 其他：
>
>

---

## 结果（训练已完成，AI 填写）

### 原始输出（result.txt）

```
训练段（最终模型口径，非正式值）：mse 0.25927, mae 0.39870, rse 0.54924
补测段（最佳 val 口径，★正式引用）：mse 0.19209, mae 0.34388, rse 0.47275
```

### 物理四指标（`scripts/eval_metrics.py --config configs/EXP-25_2026-10-10.json`，num_train=2838）

| 指标 | 值 |
|------|-----|
| step0 MAE | **11.41** |
| 全步 MAE | **23.37** |
| RMSE | 29.79 |
| R² | **0.777** |
| E_r | −6.8 |
| E_m | 0 |

误差分层（MAE）：0-50 → 13.8；50-100 → 23.3；100-150 → 30.3；>150 → 29.9

### 滚动评估（基线类实验必跑，SOP §5）

滚动 MAE（70 月）**30.32**；前期 18.63 / 后期 40.93 = **2.2×**；真实峰 244.3 vs 预测 224.3（−20.0，第 48 月）

### 训练过程特征

- 约 **1.24 s/epoch**（GPU），50 epoch 约 1 分钟
- EarlyStopping counter 在第 47 轮为 43/100 → 最佳 val 出现在第 4 轮附近，之后持续恶化
  → 与历史记录"sl336 在 50ep 下 epoch4 后过拟合"**一致**，配置选择得到侧面印证

### 结论（AI 判断，标注为 AI 结论）

1. 新环境**零配置改动**复现旧基线：全步 MAE 23.37 vs 旧 0b 的 23.87，差 **−2.1%**。
2. 该偏差与单种子噪声同量级，**不能据此断言"环境无影响"**，只能说锚点已重建、量级可比。
3. 通道侧结论：GPU 训练 + 补测 + 物理评估 + 滚动评估四段链路全部跑通。

### 本次踩到并修掉的新坑（建议补入 SOP 坑清单）

| # | 坑 | 根因 | 对策 |
|---|-----|------|------|
| 29 | `torch.load` 报 UnpicklingError，补测无法加载 checkpoint | torch≥2.6 把默认值从 `weights_only=False` 改成 `True`，而 checkpoint 含 `argparse.Namespace`/numpy 标量与 dtype；白名单随 numpy 版本漂移、无法穷尽 | 显式 `weights_only=False`，仅用于本项目自身产物；见 `exp_main._load_checkpoint` |
| 30 | 训练启动即崩：`AttributeError: np.Inf was removed in the NumPy 2.0 release` | numpy 2.0 移除 `np.Inf` 别名 | 改 `np.inf`（三处工具文件） |
| 31 | `eval_metrics.py --config` / `roll_eval.py --config` 必然 IndexError | results 目录名格式串的占位符比实参多一个（漏了 `--itr` 的 `_0` 后缀） | 补传 `itr-1`；改后按 SOP §5 跑 `scripts/test_eval.py` 回归（PASS） |
| 32 | 配置冻结时 `num_train` 静默取了默认 3119，与 W1 的 2838 不符 | `save_config.py` 的 `--num_train` 默认值 3119，不随日期切分自动推导 | 显式传 `--num_train 2838`；**这类"默认值静默生效"的错必须靠核对而不是靠跑通来发现** |

### 自检

- [x] 本次只改了 1 个变量？→ 环境（设备+torch 版本），配置零改动；非"单变量对照"而是环境重基线
- [x] 对照基线用的是同一配置？→ 与 EXP-17-0b 同配置、同窗口、同口径
- [x] 训练入口是 `run_longExp.py`？→ 是，经 `launch_experiment.sh`，命令链 train && 补测
- [x] 指标用物理口径报告？→ 是（MAE/RMSE/R² + 分层）
- [x] `result.txt` 追加了行？→ 是（两行：训练段 + 补测段）
- [x] `experiment_history.md` 索引表追加了一行？→ 是（顶部 + 文末 Stage 7 节）
- [x] diary 当天日志已写？→ 是（`diary/2026-10-10.md`）
- [ ] 图内数值与已记值对拍？→ 本次未出图
