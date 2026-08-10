# 实验操作规程

> 为什么有这份规程：12 个坑的教训，每条对应下面一条规则。
> 这份文件给两个读者看——你 和 未来跟你协作的 AI。

---

## 两档

| 档位 | 用在哪 | 记录方式 |
|------|--------|---------|
| 全流程 | 正式对照实验（一次一个变量） | 模板 + 索引表 |
| 批量 | 参数扫描（如 seq_len sweep） | 汇总表，不逐次填模板 |

---

## 全流程

### 1. 实验前（5 分钟，不写完不跑）

[ ] **训练入口脚本**：必须是 `run_longExp.py`（原管线）
    → `run_sunspot_fixed.py` 仅为 EXP-13/14/15 快速校准，不做新实验
    → 运行时：`PYTHONPATH=PatchTST_supervised python3 run_longExp.py ...`
    → run_longExp.py 的 checkpoint 存 `./checkpoints/`（不是 `PatchTST_supervised/checkpoints/`）

[ ] **features 确认**：MS（单目标预测SSN）还是 M（多变量全通道预测）？
    → 对照基线用什么就跟着用什么。当前约定：MS

[ ] **batch_size / num_workers / enc_in**：与对照基线一致
    → Baseline B 为 bs=32, num_workers=0, enc_in=3

[ ] **数据路径**：`--root_path` 显式传 `./PatchTST_supervised/dataset/`（run_longExp.py 默认是 `./data/ETT/` 会错）

[ ] **可证伪假设**（三行填完）：
    如果有效，step0 ≤ ___ 且 E_r ≤ ___
    如果无效，step0 ≈ ___
    如果结果是 ___ → 推论是 ___

[ ] **预尸检**：如果两周后证明这实验是废物，最可能因为什么？
    → 回答后检查：前提条件对吗（mode/数据范围/归一化）？

[ ] 只改了一个变量？是 ___（≥2 → 不跑）

[ ] 对照基线：___（索引表里的实验 ID）

[ ] 这个结论只能用在什么前提条件下？
    （mode: M/MS？数据: 1749+/1867+？epochs: 10/50？）

[ ] 成功标准：MAE 从 ___ 到 ___ 算有效；不到 ___ 算无效

### 2. 工作区隔离

[ ] `git worktree add ../EXP-XX -b EXP-XX`
[ ] `cd ../EXP-XX`
[ ] 读主项目 `AGENTS.md` 和 `experiment_SOP.md`

### 3. 配置冻结

```bash
python save_config.py EXP-XX --reason "一句话目的"
```
→ 输出 `configs/EXP-XX_YYYY-MM-DD.json`

### 4. 训练

```bash
python run_sunspot_fixed.py [参数...]
```
⚠️ 先存 config 再改 rsf.py 参数。顺序反过来 = 记录的配置和实际不一样。

### 5. 评估（门禁——缺一个不算完成）

```bash
python eval_metrics.py --config configs/EXP-XX_YYYY-MM-DD.json
```

固定输出：
- [ ] step0 MAE（物理）：______
- [ ] 全步 MAE（物理）：______
- [ ] E_r（峰值幅度误差）：______
- [ ] E_m（峰值时间误差）：______
- [ ] 误差分层：0-50 / 50-100 / 100-150 / >150
- [ ] 滚动 MAE（每一步推进1月，滚动70月）：
    ⚠️ 已自动化：`python roll_eval.py --config configs/EXP-XX.json`。
    正式对照实验需根据实验目的决定是否跑：
    - 基线实验（如 Baseline B 复现）→ 必须跑
    - 参数消融（如改 loss/激活）→ 可不跑（全步已覆盖）
    结果解读：前期 MAE 和后期 MAE 的比值反映误差累积速度。

### 6. 记录（先结论，后细节）

[ ] 一句话结论（跑完立刻写，别拖）：
    "___：step0 ___→___，全步 ___→___，有效/无效"

[ ] 填 `experiment_template.md`

[ ] 追加索引表一行（`experiment_history.md` 顶部）

[ ] 追加 `result.txt` 一行

[ ] 更新未做方向追踪表（§8）

[ ] git add → commit → push

### 7. 红线

- 改 ≥2 个变量 → 不跑
- z-score RSE 和物理 MAE 混用 → 不算
- "看看结果再说" → 没写假设就不跑
- 四个指标缺一 → 不算完成
- 跑完说不出"它告诉了我什么" → 必须说出来再开下一个
- 说不出来"什么样算无效" → 假设不完整，不跑
- 先改 rsf.py 再存 config → 不跑（先存再改）

### 8. 失败处理

连续 3 个实验没超过 10% 改善 → 停止，写总结

实验前提是错的（如 mode 不对、归一化不匹配） → 不补实验，先纠正前提再开新方向

自己也不知道"它说明了什么" → 不跑下一个实验

---

## 批量

适用场景：参数扫描（如 seq_len=48/96/132/192）

[ ] 扫描矩阵：变量名 × 取值列表
[ ] 存 `scan_config.json`（矩阵 + 固定参数）
[ ] 预期趋势：参数往哪个方向，指标应该往哪个方向
[ ] 跑
[ ] 每行四个固定指标 + 一句话结论
[ ] 汇总成一张表，追加到 `experiment_history.md` 末尾
[ ] 每个子实验的 config JSON 存 `configs/scan_ID_N.json`
[ ] commit 时：scan_config.json + 汇总表 + 所有 config JSON

---

## 未做方向追踪

跑完一个实验后扫一眼此表。能顺手做就做，不能就更新"为什么没做"。

| 方向 | 提出日期 | 来源 | 为什么没做 | 还做吗 |
|------|---------|------|-----------|--------|
| ReLU head (A1) | 05.07 | A-H 清单 | — | ⬜ |
| 高值加权 loss (D1) | 05.07 | A-H 清单 | — | ⬜ |
| Log/sqrt 变换 (E1) | 05.07 | A-H 清单 | — | ⬜ |
| 相位编码 | — | — | — | ⬜ |
| GRC 残差校正 | — | 文献 | — | ⬜ |

---

## 环境信息

| 项目 | 值 |
|------|-----|
| Python | 3.10.12 |
| PyTorch | 1.11.0 |
| GPU | 无（纯 CPU 训练） |
| 训练耗时 | seq=132/bs=32/50ep：约 22s/epoch（CPU），50 ep≈18 分钟。seq=96/bs=16/10ep：约 2-3 分钟 |
| 评估耗时 | npy 反算 + 4 指标：< 5 秒。滚动评估（70 次推理）：约 2-3 分钟 |

---

## AI 执行规范

给 AI 看的固定信息，不推导。

### 路径（硬编码）

| 用途 | 路径 |
|------|------|
| 数据 | `PatchTST_supervised/dataset/sunspot_with_cycle.csv` |
| 训练入口 | `run_sunspot_fixed.py` |
| 配置快照 | `save_config.py` |
| 配置输出 | `configs/` |
| 评估脚本 | `eval_metrics.py` |
| 索引表 | `experiment_history.md` 顶部 |
| 结果记录 | `result.txt` |
| 单次记录模板 | `experiment_template.md` |

### 四指标定义

- **step0 MAE**：`mean(abs(pred_phy[0] - true_phy[0]))` — 窗口 0 的全部 pred_len 步平均
- **全步 MAE**：`mean(abs(pred_phy - true_phy))` — 所有 47 窗口 × 24 步平均
- **E_r**：`max(pred_phy) - max(true_phy)` — 峰值幅度误差（正=高估，负=低估）
- **E_m**：预测峰值所在月份 - 真实峰值所在月份（正=偏晚，负=偏早）

### 索引表追加格式

```
| EXP-XX | YYYY-MM-DD | Stage | 改动 | 配置 | 全步MAE | step0 | R² | E_r | E_m | 结论 | 行号 | 有config? |
```

### 目录命名解析

- `sl`=seq_len / `pl`=pred_len / `ll`=label_len
- `dm`=d_model / `nh`=n_heads / `el`=e_layers / `df`=d_ff
- **patch_len 和 stride 不在目录名中**——只能在 config JSON 或 checkpoint 里确认

---

## 参考

| 文档 | 用途 |
|------|------|
| `experiment_template.md` | 单次实验记录模板 |
| `experiment_history.md` | 全量实验索引 + 详情 |
| `AGENTS.md` | 项目架构、基线参数、红线 |
| `project_roadmap.md` | 路线图、下一步任务 |
