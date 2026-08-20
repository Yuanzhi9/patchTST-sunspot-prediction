# 实验历史记录 — PatchTST 太阳黑子预测项目

> 创建日期：2026-08-09
> 版本 v1.0
>
> 记录项目从 2026.04 到 2026.08 的全部实验阶段，包括数据、配置、代码路径、训练命令、结果及出处。
> 每个数值标注可查证性：✓ 服务器可查 / ⚠️ 用户本机存档 / ✗ 缺失 / ? 待确认。
>
> 每阶段末尾留有「探索目标」「收获」「不足」三个空位，由用户自行填写。

---

## 总览时间线

```
2026.04.08 ─── 04.16   Stage 0: MinMax + 1749+ 时代
2026.04.xx ─── 05.13   (EXP-01 ~ EXP-12, 各方向初步探索，穿插于 Stage 0-2)
2026.05.06 ─── 05.07   Stage 1: 1867+ Baseline B 确立
2026.05.07 ─── 05.15   Stage 2: H1/H2 诊断 + A-H 清单
2026.05.18 ─── 06.14   Stage 3: 配置变更 + M4引入 + 单变量对照（EXP-13~15）
2026.07.13              Stage 4: Level 3 残差预测
2026.07.17              Stage 5: 天花板探测
2026.07.23 ─── 08.09   Stage 6: Phase 0 文献奠基
```

---

## 快速索引

> 用法：Ctrl+F 搜实验 ID，跳转到对应行号看详情。
> 新增实验后，在本表中追加一行 + 在 Stage 末尾追加一节记录。
> 记录模板：`project_docs/experiment_template.md`

| ID | 日期 | 阶段 | 一句话改动 | 数据/配置 | 物理 MAE 全步 | step0 | R² | 结论 | 行号 | 有config? |
|----|------|------|-----------|----------|--------------|-------|-----|------|------|-----------|
| **S0 基线** | 04.08-16 | Stage 0 | MinMax基线 | 1749+,512,sl132 | — | 25.80 | — | 上限144<<216，极值压缩 | L27 | ❌ 无 |
| S0 ReLU | 04.09 | Stage 0 | Linear→ReLU | 同上 | — | ≈25.8 | — | 负值率5.56%→0% | L27 | ❌ 无 |
| S0 Huber | 04.09 | Stage 0 | MSE→Huber | 同上 | — | 27.36 | — | 无贡献（MinMax配置下） | L27 | ❌ 无 |
| S0 Softplus | 04.xx | Stage 0 | ReLU→Softplus | 同上 | — | 28.12 | — | 更差，系统性高估 | L27 | ❌ 无 |
| **实验A** | 05.06 | Stage 1 | buggy基线(drop_last) | 1867+,dm128,sl132,n16,e3 | — | 11.15 | — | 32样本，不可用 | L155 | ❌ 无 |
| **Baseline B** | 05.07 | Stage 1 | bug修复后基线 | 1867+,dm128,sl132,n16,e3,50ep | — | 13.02 | — | 峰值低估18% | L166 | ❌ 无 |
| H2 多步长 | 05.07 | Stage 2 | step 0/6/12/18/23 | 同Baseline B checkpoint | 23.91 | — | — | ⚠️ [2026-08-10提出] 口径待确认 | L224 | ❌ 无 |
| H1 滚动 | 05.07 | Stage 2 | 逐月H1滚动预测 | 同Baseline B checkpoint | — | — | — | MAE=33.13，峰顶≈54 | L232 | ❌ 无 |
| EXP-01 | 04.xx-05.13 | Stage 3 | Weather基准验证 | Weather数据集 | — | — | — | 代码环境OK | L349 | ❌ 无 |
| 分阶段 | 04.xx-05.13 | Stage 3 | 四模型阶段训练 | dm512,~200样本/阶段 | — | — | — | 过拟合，已暂停 | L355 | ❌ 无 |
| EXP-02 | 04.xx-05.13 | Stage 3 | Cycle23留出 | M mode,3特征 | — | — | — | RSE=0.196(z-score) | L359 | ❌ 无 |
| EXP-03 | 04.xx-05.13 | Stage 3 | 特征4→3 | M mode | — | — | — | year_norm无贡献 | L359 | ❌ 无 |
| EXP-04 | 04.xx-05.13 | Stage 3 | pred_len 96→132 | M mode | — | — | — | RSE=0.166-0.193 | L359 | ❌ 无 |
| **EXP-05** | 04.xx-05.13 | Stage 3 | M→MS+sl96→132+RevIN1→0 | MS mode,sl132 | — | — | — | ❌ 3变量同改，退化 | L367 | ❌ 无 |
| EXP-06 | 04.xx-05.13 | Stage 3 | 关EarlyStopping | MS mode | — | — | — | RSE=0.521，无改善 | L367 | ❌ 无 |
| EXP-07 | 04.xx-05.13 | Stage 3 | StandardScaler→MinMax | MS mode | — | — | — | RSE=0.397 | L367 | ❌ 无 |
| EXP-08 | 04.xx-05.13 | Stage 3 | GELU→ReLU | MS mode | — | — | — | RSE=0.389 ⚠️ MS下 | L367 | ❌ 无 |
| EXP-09 | 04.xx-05.13 | Stage 3 | MSE→Huber | MS mode | — | — | — | RSE=0.389=MSE ⚠️ M下未测 | L367 | ❌ 无 |
| EXP-10 | 04.xx-05.13 | Stage 3 | 回到MSE基线 | MS mode | — | — | — | RSE=0.389 | L367 | ❌ 无 |
| EXP-11 | 04.xx-05.13 | Stage 3 | Softplus+Huber | MS mode | — | — | — | RSE=0.474，更差 | L367 | ❌ 无 |
| EXP-12 | 04.xx-05.13 | Stage 3 | 1867+截断+4参数 | dm128,n16,e3,df256 | — | — | — | ❌ 5变量同改 | L386 | ❌ 无 |
| EXP-13 | 06.14 | Stage 3 | dm512基线 | 1749+,sl96,dm512,e2,n8 | 25.27 | — | 0.539 | 与EXP-14成对对照 | L392 | ❌ 无 |
| **EXP-14** | 06.14 | Stage 3 | **dm512→128（唯一干净对照）** | 1749+,sl96,dm128,e2,n8,10ep | **23.87** | **9.08** | **0.568** | 降参有泛化提升；峰值仍68.8 | L392 | ❌ 无 |
| EXP-15 | 06.14 | Stage 3 | dm128 dry run | df? ⚠️ [2026-08-10提出] 待确认 | — | — | — | RSE=0.813，无效 | L392 | ❌ 无 |
| **Level 3** | 07.13 | Stage 4 | M4+PatchTST残差预测 | dm128,sl96 | — | — | — | 退化（3.32→4.48），不可行 | L450 | ❌ 无 |
| **A-sl96** | 07.17 | Stage 5 | PatchTST sl96基线 | RevIN=1,lr=0.0001,10ep | 23.87 | 9.08 | 0.568 | 复用EXP-14 | L498 | ❌ 无 |
| B-sl192 | 07.17 | Stage 5 | seq_len 96→192 | RevIN=1,lr=0.0001,30ep | 22.02 | 12.30 | 0.625 | 更多上下文有边际帮助 | L498 | ❌ 无 |
| C-sl336 | 07.17 | Stage 5 | seq_len 192→336 | RevIN=1,lr=0.0001,30ep | 20.54 | 10.87 | 0.692 | 信息量边际递减 | L498 | ❌ 无 |
| D1-DLinear | 07.17 | Stage 5 | DLinear(ind=0) | RevIN=0,lr=0.005,30ep | 20.31 | 19.04 | 0.722 | ⚠️ 与PatchTST不可跨组比 | L498 | ❌ 无 |
| D2-DLinear-I | 07.17 | Stage 5 | DLinear(ind=1) | RevIN=0,lr=0.005,30ep | **19.30** | 19.36 | **0.751** | 组内最优 ⚠️ 不可跨组比 | L498 | ❌ 无 |
| **EXP-16** | 08.10 | Stage 7 | ~~Baseline B参数×1749+~~ | sl132,... | 24.02 | 18.07 | 0.553 | ❌ 无效：误用run_sunspot_fixed.py | L691 | — |
| **EXP-16b** | 08.10 | Stage 7 | ~~patience 3→20~~ | 同上,pat=20 | 24.85 | 13.90 | 0.524 | ❌ 无效：同上 | L691 | — |
| **EXP-16c** | 08.10 | Stage 7 | **Baseline B参数×1749+（run_longExp.py原管线）** | sl132,dm128,nh16,el3,df256,pl12,str6,MS,bs32,50ep(ES@38) | **24.54** | **11.21** | **0.532** | step0优于B(13.02)，滚动36.37差于B(33.13)，峰值-85.5。1749+宽range瓶颈 | L691 | — |
| **EXP-17-0a** | 08.12 | Stage 8 | **Round 0 sl96 MS 50ep W1** | sl96,dm128,MS,50ep,W1(Cycle23) | 25.24 | 8.81 | 0.736 | 滚动43.51，E_r=-36.2 ⚠️ 最佳ckpt口径+并行，已被-r2取代 | L795 | ✓ |
| **EXP-17-0b** | 08.12 | Stage 8 | **Round 0 sl336 MS 50ep W1** | sl336,dm128,MS,50ep,W1(Cycle23) | **23.87** | 11.39 | **0.769** | 滚动71.79但全步+峰值优于0a ⚠️ 最佳ckpt口径+并行，已被-r2取代 | L795 | ✓ |
| **EXP-17-0a-r2** | 08.13 | Stage 8 | **Round 0 口径修正重跑 sl96 串行** | sl96,dm128,MS,50ep,W1,串行 | 24.27 | 18.21 | 0.757 | 最终模型口径。step0从8.81→18.21证实口径影响巨大 | L795 | ✓ |
| **EXP-17-0b-r2** | 08.13 | Stage 8 | **Round 0 口径修正重跑 sl336 串行** | sl336,dm128,MS,50ep,W1,串行 | 26.53 | 12.87 | 0.704 | 🛑 最终模型口径。过拟合(val epoch4后恶化46轮)。门禁11.1%触发，结论反转待重审 | L795 | ✓ |
| **EXP-18-1a** | 08.12 | Stage 8 | **Round 1 patch_len 16→12 (stride 8→6)** | sl336,pl12,str6,MS,50ep,W1 | 25.54 | 16.40 | 0.739 | ❌ 差基线6.9%淘汰 ⚠️ 08.14补测最佳模型口径 | L828 | ✓ |
| **EXP-18-1c** | 08.12 | Stage 8 | **Round 1 patch_len 16→24 (stride 8→12)** | sl336,pl24,str12,MS,50ep,W1 | 23.53 | 13.30 | 0.775 | 优于基线1.4%<3%=无差异，维持(16,8) ⚠️ 08.14补测最佳模型口径 | L828 | ✓ |
| **EXP-18-2b** | 08.14 | Stage 8 | **Round 2 e_layers 2→3** | sl336,el3,MS,50ep,W1 | 23.43 | 10.00 | 0.777 | 改善1.8%<3%无差异，维持el=2 | L860 | ✓ |
| **EXP-18-2c** | 08.14 | Stage 8 | **Round 2 e_layers 2→4** | sl336,el4,MS,50ep,W1 | 23.35 | 11.88 | 0.774 | 改善2.2%<3%无差异；E_r=-11.8优35%但主指标不支持 | L860 | ✓ |
| **EXP-18-3a** | 08.14 | Stage 8 | **Round 3 d_ff 2048→512** | sl336,df512,MS,50ep,W1 | 25.32 | 12.67 | 0.741 | ❌ 恶化6.1%≥5%，512容量不足 | L860 | ✓ |
| **EXP-18-3b** | 08.14 | Stage 8 | **Round 3 d_ff 2048→1024** | sl336,df1024,MS,50ep,W1 | 24.83 | 14.97 | 0.752 | 恶化4.0%，趋势单调，维持2048 | L860 | ✓ |
| **EXP-18-4a** | 08.14 | Stage 8 | **Round 4 d_model 128→64** | sl336,dm64,MS,50ep,W1 | 24.17 | 13.39 | 0.763 | 差1.3%但step0恶化17.6%，不采"无差异" | L860 | ✓ |
| **EXP-18-4b** | 08.14 | Stage 8 | **Round 4 d_model 128→256** | sl336,dm256,MS,50ep,W1 | 24.70 | 13.67 | 0.751 | ❌ 恶化3.5%≥3%淘汰；E_r=-1.4留档 | L860 | ✓ |
| **EXP-18-5a** | 08.14 | Stage 8 | **Round 5 dropout 0.05→0.1** | sl336,do0.1,MS,50ep,W1 | 22.92 | 10.56 | 0.785 | 改善4.0%≥3%，正则不足 | L860 | ✓ |
| **EXP-18-5b** | 08.14 | Stage 8 | **Round 5 dropout 0.05→0.2** | sl336,do0.2,MS,50ep,W1 | **22.62** | 11.01 | **0.789** | ✅ 改善5.2%，**新基线**。E_r=-24.4留档 | L860 | ✓ |
| **EXP-18-6a** | 08.14 | Stage 8 | **Round 6 gelu→relu** | sl336,relu,MS,50ep,W1 | 22.96 | 12.51 | 0.785 | 差1.5%<3%无差异，维持gelu | L860 | ✓ |
| **EXP-18-7a** | 08.14 | Stage 8 | **Round 7 individual 0→1** | sl336,ind1,MS,50ep,W1 | 24.13 | 13.26 | 0.764 | ❌ 恶化6.7%≥3%，维持ind=0 | L860 | ✓ |
| **EXP-19-2** | 08.14 | Stage 8 | **验证 W2(Cycle24) 最优配置** | do0.2最优配置,W2 | 20.03 | 36.00 | 0.579 | 未退化；弱周期R²/step0弱 | L880 | ✓ |
| **EXP-19-3** | 08.14 | Stage 8 | **验证 W3(Cycle25部分) 最优配置** | do0.2最优配置,W3 | 21.11 | 10.12 | 0.684 | E_r=-70.2，峰值压制系统性瓶颈 | L880 | ✓ |
| **EXP-20-1** | 08.14 | Stage 9 | **探索期 revin=0 诊断** | sl336,revin0,MS,50ep,W1 | 22.72 | 14.99 | 0.791 | 滚动崩溃(1967万)。机制3证伪：RevIN是滚动稳定性支柱 | L900 | ✓ |
| **EXP-20-2a** | 08.15 | Stage 9 | **探索期 pred_len=12** | sl336,pl12,MS,W1 | 21.38* | 8.18 | 0.813 | 滚动52.21≈pl24。*跨pl不可比 | L900 | ✓ |
| **EXP-20-2b** | 08.15 | Stage 9 | **探索期 pred_len=48** | sl336,pl48,MS,W1 | 24.05* | 21.41 | 0.756 | 滚动崩溃(37万)，pl48禁区 | L900 | ✓ |
| **EXP-20-3** | 08.14 | Stage 9 | **探索期 wmse α=1.0** | sl336,wmse,MS,W1 | 25.92 | 10.09 | 0.730 | 峰值改善92%(滚动峰-2.2)但全步恶化14.6% | L900 | ✓ |
| **EXP-20-4** | 08.15 | Stage 9 | **探索期 sqrt 变换** | sl336,sqrt,MS,W1 | 21.40 | 11.82 | 0.800 | 滚动改善45%(27.80)但峰值更差(-64.6) | L900 | ✓ |
| **EXP-21-1** | 08.15 | Stage 9 | **探索期 pow0.7 变换** | sl336,pow07,MS,W1 | 22.15 | 14.98 | 0.793 | 滚动22.50 | L900 | ✓ |
| **EXP-21-2** | 08.15 | Stage 9 | **探索期 pow2/3 变换** | sl336,pow23,MS,W1 | 22.15 | 15.35 | 0.792 | 滚动22.13(变换系最优) | L900 | ✓ |
| **EXP-21-3** | 08.15 | Stage 9 | **探索期 log1p 变换** | sl336,log1p,MS,W1 | 21.85 | 7.70 | 0.769 | 全步E_r=+1.3惊人但滚动峰-47.8 | L900 | ✓ |
| **EXP-22-1** | 08.15 | Stage 9 | **探索期 loss=mae** | sl336,mae,MS,W1 | 22.88 | 16.84 | 0.780 | 滚动124恶化2.5倍，mae关停 | L900 | ✓ |
| **EXP-22-2** | 08.15 | Stage 9 | **探索期 wmse α=0.5** | sl336,wmse0.5,MS,W1 | 24.35 | 9.72 | 0.760 | 滚动峰-9.7(改善64%)+滚动38.87(改善23%)，折中 | L900 | ✓ |
| **EXP-22-3** | 08.15 | Stage 9 | **探索期 wmse_th 阈值权重** | sl336,wmse_th,MS,W1 | 24.03 | 9.72 | 0.765 | **loss系最佳：峰-6.2(77%)+滚动33.92(33%)+低值损伤最小** | L900 | ✓ |
| **EXP-22-4** | 08.15 | Stage 9 | **探索期 非对称惩罚** | sl336,asym,MS,W1 | 26.04 | 12.04 | 0.731 | 整体抬高不稳，滚动峰转高估+25.5 | L900 | ✓ |
| **EXP-23-1** | 08.15 | Stage 9 | **探索期 block24无重叠（外推策略）** | 5b/19-2/19-3模型 | W1 21.29 / W2 27.28 / W3 23.75 | — | — | **三窗口优于滚动44-59%！阶段3外推策略证据** | L900 | ✓ |
| **EXP-23-2** | 08.15 | Stage 9 | **探索期 block24重叠12** | 同上 | W1 58.23 / W2 56.76 / W3 56.03 | — | — | 接缝平均反而更差，重叠法放弃 | L900 | ✓ |
| **EXP-24** | 08.20 | Stage 9 | **极小值专项评估（分析任务）** | 三窗口已定论模型 | — | — | — | 谷值区=模型最弱区：滚动无谷/负值崩塌，block时间±3月可用但幅度差1-2量级。阶段3谷值需专门策略 | L900 | ✓ |

> ⚠️ **必须了解的坑**：
> - Baseline B (13.02) 和 EXP-14 (9.08) 的 step0 不可直接比——配置差 9 项
> - EXP-05~11 在 MS mode 下，M mode 下未验证的结论不能推广
> - ⚠️ 目录命名：`pl`=pred_len[^1]
>   不是`patch_len`！patch_len 和 stride 不在目录名中，只能从 checkpoint 读取。AI 曾将 `pl24` 误写为 `patch=24`，是错的。
> - z-score RSE 和物理 MAE 不可互推，指标不统一
> 
> > 待查问题标注 `⚠️ [YYYY-MM-DD提出]`。超过 3 个月未解决的自动归档为"无法查证"。

---

## Stage 0: MinMax + 1749+ 时代（2026.04.08 — 04.16）

### 0.1 数据

| 项目 | 内容 |
|------|------|
| 数据文件 | `sunspot_with_cycle.csv`（推测，五月文档提及使用的是 1867+ 文件不同） |
| 时间范围 | 1749-2020（约 3321 个月） |
| 训练集 | 1749–1995（约 2964 个月，含峰值 398.2） |
| 测试集 | 2009–2020（132 个月，低活动期） |
| 特征列 | ssn, month_sin, month_cos, year_norm, cycle_phase（5 维） |
| 标准化 | MinMaxScaler（全局 fit，无区分 train/test） |
| 可查证性 | ⚠️ 代码和数据位于用户本机 `F:\Downloads\patchTST_main\PatchTST-main\PatchTST_supervised\` |

### 0.2 配置

| 参数 | 值 |
|------|-----|
| 模型 | PatchTST |
| seq_len | 132（11 年） |
| pred_len | 24（2 年） |
| d_model | 512 |
| n_heads | 8 |
| e_layers | 2 |
| d_ff | 2048 |
| patch_len | 12 |
| stride | 6 |
| revin | 0 |
| batch_size | 32 |
| learning_rate | 0.0001 |
| patience | 10 |
| train_epochs | 50 |
| seed | ? 待确认 |
| 训练入口脚本 | ? 待确认 (run_longExp.py 或 run_sunspot_fixed.py) |
| 可查证性 | ⚠️ 本机存档 |

**代码改动记录（来源：用户 2026.04.16 文档）：**
- `models/PatchTST.py` 第 95 行：添加 `x = F.relu(x)  #2026.04.09添加`
- `run_longExp.py` 第 63 行：`default='huber'  #0206.04.09修改`
- 操作顺序：先加 ReLU，后改默认 Loss

**训练命令**：⚠️ 待补充（位于用户本机）

### 0.3 实验结果

Five experiments, all under MinMax normalization, all evaluated on the same test set (2009→2020, 132 months):

| 实验 | Loss | 激活 | MAE (物理) | RMSE | 预测最小值 | 预测最大值 | 负值率 | 出处 |
|------|------|------|-----------|------|-----------|-----------|--------|------|
| MinMax 基线 | MSE | Linear | 25.80 | 34.98 | -11.99 | 144.21 | 5.56% | 用户 04.16 文档 |
| 声称 ReLU | MSE | ReLU | — | — | 0.00 | ~144 | 0% | 用户 04.16 文档 |
| Huber_Test | Huber | ReLU | 27.36 | 36.51 | 0.00 | 140.82 | 0% | 用户 04.16 文档 |
| 实验 D | Huber | ReLU | 27.36 | 36.51 | 0.00 | 140.82 | 0% | 用户 04.16 文档 |
| Softplus_Huber | Huber | Softplus | 28.12 | 35.66 | 9.20 | 126.73 | 0% | 用户 04.16 文档 |

测试集物理范围：[0.20, 216.00]

数据来源标注：「用户 04.16 文档」指用户 2026.04.16 提供的阶段总结报告。具体计算方式和 scaler 参数待确认。

### 0.4 关键发现与结论

- ReLU 消除负值（5.56% → 0%）
- Huber Loss 在已有 ReLU 时无额外贡献（结果与 MSE+ReLU 几乎相同）
- Softplus 不适用（引入系统性高估，上限从 ~144 降至 ~126）
- **所有配置预测上限均远低于真实峰值 216**，根因判断为 MinMax 归一化受 398.2 极值压缩
- 下一步计划：换 Log / Box-Cox 归一化

### 0.5 探索目标

【留空】

### 0.6 收获

【留空】

### 0.7 不足

【留空】

---

## Stage 1: 1867+ Baseline B 确立（2026.05.06 — 05.07）

### 1.1 数据

| 项目 | 内容 |
|------|------|
| 数据文件 | `sunspot_1867-02_2025-10_original_sincos.csv` |
| 服务器路径 | `/root/code/patchTST-sunspot-prediction/PatchTST_supervised/dataset/sunspot_1867-02_2025-10_original_sincos.csv` |
| 时间范围 | 1867-02 到 2025-10（1905 个月） |
| 训练集 | 1867-02 到 2008-12（1703 个月） |
| 验证集 | 2009-01 到 2019-12（132 个月） |
| 测试集 | 2020-01 到 2025-10（70 个月） |
| 特征列 | date, ssn, month_sin, month_cos（4 列） |
| 归一化 | z-score（StandardScaler，fit on train only）+ RevIN（模型内部） |
| 可查证性 | ✓ 数据文件在服务器上可查；⚠️ 训练代码和 checkpoint 在本机 |

### 1.2 配置

| 参数 | 值 |
|------|-----|
| 模型 | PatchTST |
| Head | **Linear**（无激活函数） |
| seq_len | 132（11 年） |
| pred_len | 24（2 年） |
| d_model | 128 |
| n_heads | 16 |
| e_layers | 3 |
| d_ff | 256 |
| patch_len | **12**（目录名写 16，实际 checkpoint 存的是 12） |
| stride | **6**（目录名写 8，实际 checkpoint 存的是 6） |
| revin | 1 |
| dropout | 0.05 |
| fc_dropout | 0.05 |
| head_dropout | 0.0 |
| 优化器 | Adam |
| learning_rate | 0.0001 |
| epochs | 50 |
| seed | ? 待确认 |
| 训练入口脚本 | ? 待确认 (run_longExp.py 或 run_sunspot_fixed.py) |
| 可查证性 | ⚠️ 代码在本机 |

**代码 bug 修复记录（来源：用户 2026.05.06 文档）：**
1. `data_factory.py`：`drop_last=True` → `drop_last=False`（否则测试集丢弃最后 15 个样本）
2. `exp_main.py`：`np.array(preds)` → `np.concatenate(preds, axis=0)`（否则 batch 拼接报错）

**训练命令**：⚠️ 待补充（位于用户本机）

### 1.3 实验 A（buggy，drop_last=True）

| 指标 | 数值 | 出处 |
|------|------|------|
| 测试样本数 | 32（47 − 15 被丢弃） | 用户 05.06 文档 |
| 预测覆盖 | 2020-01 ~ 2022-08 | 用户 05.06 文档 |
| 物理 MAE | 11.15 | 用户 05.06 文档 |
| 物理 RMSE | 12.71 | 用户 05.06 文档 |
| 预测范围 | [0.9, 93.5] | 用户 05.06 文档 |
| 真实范围 | [0.2, 96.5] | 用户 05.06 文档 |

### 1.4 实验 B（完整基线，bug fixed）

| 指标 | 数值 | 出处 |
|------|------|------|
| 测试样本数 | 47 | 用户 05.06 文档 |
| 预测覆盖 | 2020-01 ~ 2023-11 | 用户 05.06 文档 |
| 物理 MAE | 13.02 | 用户 05.06 文档 |
| 物理 RMSE | 16.66 | 用户 05.06 文档 |
| 峰值真实值 | 160.5 | 用户 05.06 文档 |
| 峰值预测值 | 131.3（低估 ~18%） | 用户 05.06 文档 |
| z-score MAE | ? 待查证（用户本机 npy） | — |
| 负值预测率 | 0% | 用户 05.06 文档 |

**Checkpoint 路径**（来源：用户 05.06 文档）：
```
checkpoints/Baseline_1867plus_D128_RevIN1_Linear_PatchTST_custom_ftMS_sl132_ll48_pl24_dm128_nh16_el3_dl1_df256_fc1_ebtimeF_dtTrue_Baseline_1867plus_D128_RevIN1_Linear_0/full_checkpoint.pth
```

**Results 路径**（来源：用户 05.06 文档）：
```
F:\Downloads\patchTST_main\PatchTST-main\PatchTST_supervised\results\Baseline_1867plus_D128_RevIN1_Linear_PatchTST_custom_ftMS_sl132_ll48_pl24_dm128_nh16_el3_dl1_df256_fc1_ebtimeF_dtTrue_Baseline_1867plus_D128_RevIN1_Linear_0\
```

### 1.5 关键发现与结论

- 低值区（SSN < 50）拟合良好
- 高值区系统性低估（真实 160.5 → 预测 131.3）
- Step 23 预测在 2024-2025 段完全平坦化，丢失上升趋势
- 诊断：Linear head 在 pred_len=24 的任务难度下碰到了能力上限
- 建议：从 A1（ReLU head）开始改进

### 1.6 数据来源—结果出处对照

| 结果 | 数值 | 出处 |
|------|------|------|
| 物理 MAE=13.02 | step 0 物理 MAE（用户 5 月原始记录确认） | 用户 05.06/05.07 文档，npy 在 F:\Downloads |
| 物理 RMSE=16.66 | 全 47 窗口物理 RMSE | 同上 |
| 峰值低估 18% | 真实 160.5 → 131.3 | 同上 |
| z-score MAE | ? 待查证 | — |

### 1.7 探索目标

【留空】

### 1.8 收获

【留空】

### 1.9 不足

【留空】

---

## Stage 2: H1/H2 诊断 + A-H 清单（2026.05.07 — 05.15）

基于 Stage 1 实验 B 的同一个 checkpoint（未重新训练）。

### 2.1 H2 多步长视图

用 `pred.npy`（47 samples × 24 steps）中 step 0/6/12/18/23 的预测值分别画曲线。

- step 越大 → 曲线越平坦，退化是渐进式的
- 没有任何一条线能完整覆盖 2020-2025
- 全步平均 MAE = 23.91 ⚠️ 待确认计算口径：全部 47×24=1128 个预测点，还是仅取了 step 0/6/12/18/23 五个位置的 47×5=235 个点？这决定了能否与 EXP-14 的全步 23.87 直接比较。（来源：用户 05.07 文档）

### 2.2 H1 逐月滚动预测

每次只取 step 0，推进 1 个月后更新输入。

- 全周期（2020-01 ~ 2025-10）MAE = 33.13（来源：用户 05.07 文档）
- 初期 MAE ≈ 24，峰顶期 MAE ≈ 54
- 结论：模型对自身输出的微小误差极其敏感

### 2.3 学长 24-step block 建议

每次取全部 24 个预测值，每隔 24 个月向前推进。

- 2020-01 ~ 2021-12（24 个月，全真实输入）
- 2022-01 ~ 2023-12（24 个月，输入中 18% 是预测值）
- 2024-01 ~ 2025-12（24 个月，输入中 36% 是预测值）
- 覆盖范围解决了，但失真速度更快
- 这是 Stage 3 转向 M4 路线的前序工作

### 2.4 A-H 改进清单

| 维度 | 编号 | 改动 |
|------|------|------|
| A. Head | A1 | Linear → ReLU head |
| | A2 | 增大 hidden dim |
| B. Backbone | B1 | d_model 128→256 |
| | B2 | encoder layers 增加 |
| C. Training | C1-C5 | epochs/scheduler/lr |
| D. Loss | D1 | 高值区加权 Loss |
| | D2 | Huber Loss |
| E. Data | E1 | Log 变换 |
| | E2-E4 | 过采样、增强 |
| F. Post | F1 | 线性校准 |
| G. Long | G1-G3 | 换模型/Ensemble |
| H. Coverage | H1-H4 | 滚动/缩短 seq/pred |

05.07 建议第一优先级为 A1（ReLU head）+ D1（高值加权 Loss）。

### 2.5 05.15 完整项目总结

用户产出最终版项目文档，包含配置、结果、代码路径、改进方向。文档位于用户本机。

### 2.6 探索目标

【留空】

### 2.7 收获

【留空】

### 2.8 不足

【留空】

---

## Stage 3: 15 次实验（2026.04.xx — 06.14）

> ⚠️ 时间线澄清：EXP-01~12 在 04.xx-05.13 期间完成（穿插于 Stage 0-2）；EXP-13~15 在 05.18-06.14 完成（配置变更后 + M4 路线引入）。<br>
> 05.18 同时发生两件事：用户转向 M4 路线（§3.1）+ PatchTST 配置系统性变更（§3.2）。但 EXP-01~12 在此变更之前，配置属于前一期。仅 EXP-13~15 使用了新配置。

### 3.1 学长 M4 Waldmeier 方案引入

用户 05.18 起转向学长博士的 M4 Waldmeier 参数化曲线方案。

- M4 代码位于（服务器）：`PatchTST_supervised/shixiong_m4/coding/solar_cycle_optimization_framework_package(1)/`
- M4 方案逻辑：Waldmeier Gamma 曲线（4 参数：A 峰值振幅, tp 上升时间, alpha 不对称参数, floor 基础噪声），通过早期 36 个月观测拟合参数，绘制完整曲线
- M4 Cycle 25 预报 MAE = 3.32（远优于 PatchTST 基线，来源：AGENTS.md）

### 3.2 PatchTST 配置变更记录

M4 方案引入后，PatchTST 侧的实验配置发生了系统性变更：

#### 数据变更

| 项目 | Stage 1-2 | Stage 3 →
|------|-----------|----------|
| 数据文件 | `sunspot_1867-02_2025-10_original_sincos.csv` | `sunspot_with_cycle.csv` |
| 时间范围 | 1867-2025（1905 月） | 1749-2025（3321 月） |
| 训练集行数 | 1703 | 3119 |
| 列数 | 4 (date, ssn, sin, cos) | 8 (year, month, ssn, date, sin, cos, year_norm, cycle_phase) |
| 可查证性 | ⚠️ 本机 | ✓ 服务器 `/root/code/patchTST-sunspot-prediction/PatchTST_supervised/dataset/sunspot_with_cycle.csv` |

#### 模型参数变更

| 参数 | Stage 1-2 | Stage 3 → |
|------|-----------|----------|
| seq_len | 132 | **96** |
| n_heads | 16 | **8** |
| e_layers | 3 | **2** |
| d_ff | 256 | **2048** |
| patch_len | 12 | **16** |
| stride | 6 | **8** |
| dropout | 0.05 | 0.2（默认） |
| epochs | 50 | **10** |
| 可查证性 | ⚠️ 本机 | ✓ 服务器 |

变更原因：? 无文档记录

#### ⚠️ 目录命名规则的已知坑

PatchTST 实验目录名中：
- `sl` = seq_len（序列长度）
- `pl` = pred_len（预测长度）
- `ll` = label_len（标签长度）
- `dm` = d_model
- `nh` = n_heads
- `el` = e_layers
- `df` = d_ff

**patch_len 和 stride 不在目录名中。** 唯一确认方式是从 checkpoint `.pth` 文件中读取 `PatchTST_backbone.patch_len` 和 `PatchTST_backbone.stride`。历史文档中凡标注了 patch_len 却未注明"checkpoint 读取"的，均为推测，不可靠。

例如 Baseline B（Stage 1）的目录名不含 patch_len，但 05.07 从 checkpoint 确认实际值为 patch_len=12, stride=6（目录名中有 `pl24`，那是 pred_len=24，不是 patch_len）。

### 3.3 15 次实验记录

数据来源：`/root/code/patchTST-sunspot-prediction/result.txt`（69 行）
附加结果：`/root/code/patchTST-sunspot-prediction/PatchTST_supervised/result.txt`（1 行，dry run）

注：以下 z-score MAE/MSE/RSE 均来自 result.txt。物理 MAE 来自 AGENTS.md 记录或 npy 反算。

#### 阶段 1：代码验证

| 实验 | 名称 | result.txt 行 | MSE(z) | MAE(z) | RSE | 说明 |
|------|------|--------------|--------|--------|-----|------|
| EXP-1 | weather_96_96 | 1-9 (3 runs) | 0.186 | 0.227 | 0.569 | Weather 基准数据集，代码验证通过 |

#### 阶段 2：分阶段训练探索

Recorded in project_summary_2026-07-17.md as "Phase Training" — three cycle division methods with four-stage independent training + rolling concatenation. Results and models were on a worktree branch, not on this main branch. See project_docs/project_summary for details: rise MAE=40.4, peak MAE=51.8, decline MAE=36.7, trough MAE=6.3.

#### 阶段 3：全量数据 + M mode 早期尝试

| 实验 | 名称 | result.txt 行 | MSE(z) | MAE(z) | RSE | 说明 |
|------|------|--------------|--------|--------|-----|------|
| EXP-2 | sunspot_cycle_epoch10 | 10-15 (2 runs) | 0.087~0.097 | 0.110~0.111 | 0.196~0.204 | Cycle 23 留出测试 |
| EXP-3 | sunspot_feat3_epoch10 | 16-17 | 0.090 | 0.121 | 0.197 | 特征 4→3 列 |
| EXP-4 | sunspot_96_132 | 19-23 (2 runs) | 0.042~0.057 | 0.116~0.158 | 0.166~0.193 | pred_len 96→132 |

#### 阶段 4：MS mode 消融（死路）

全部 MS mode (ftMS)，sl=132。⚠️ patch_len 未在目录名中记录，不能确认；Baseline B 同期 sl=132 时 checkpoint 读出的 patch_len=12。

| 实验 | 名称 | result.txt 行 | MSE(z) | MAE(z) | RSE | 说明 |
|------|------|--------------|--------|--------|-----|------|
| EXP-5 | Sunspot_PatchTST_MS | 25-29 (2 runs) | 0.090~0.152 | 0.227~0.289 | 0.495~0.645 | M→MS（一次改 3 变量） |
| EXP-6 | Sunspot_PatchTST_MS_NoES | 31-32 | 0.099 | 0.234 | 0.521 | 关 EarlyStopping |
| EXP-7 | Sunspot_PatchTST_MS_MinMax | 34-35 | 0.002 | 0.031 | 0.397 | StandardScaler→MinMax |
| EXP-8 | Sunspot_PatchTST_ReLU_Test | 37-38 | 0.002 | 0.030 | 0.389 | GELU→ReLU |
| EXP-9 | Sunspot_Huber_Test | 40-41 | 0.002 | 0.030 | 0.389 | MSE→Huber |
| EXP-10 | Sunspot_Baseline_MSE | 43-44 | 0.002 | 0.030 | 0.389 | 回到 MSE baseline |

EXP-9 的关键发现：Huber loss 的 RSE 和 MSE 完全一样（0.389），瓶颈不在 loss 形式。

| 实验 | 名称 | result.txt 行 | MSE(z) | MAE(z) | RSE | 说明 |
|------|------|--------------|--------|--------|-----|------|
| EXP-11 | Sunspot_Softplus_Huber | 46-50 (2 runs) | 0.002~0.002 | 0.040~0.040 | 0.473~0.474 | Softplus+Huber，更差 |

#### 阶段 5：1867+ 截断（死路）

| 实验 | 名称 | result.txt 行 | MSE(z) | MAE(z) | RSE | 说明 |
|------|------|--------------|--------|--------|-----|------|
| EXP-12 | Baseline_1867plus_D128_RevIN1_Linear | 52-62 (4 runs) | 0.107~0.215 | 0.249~0.337 | 0.524~0.643 | 1867+ 截断 + dm128 + nh16 + el3 + df256（同时改多变量） |

#### 阶段 6：单变量对照

| 实验 | 名称 | result.txt 行 | MSE(z) | MAE(z) | RSE | 物理 MAE | 物理 RMSE | R² | 出处 |
|------|------|--------------|--------|--------|-----|----------|-----------|-----|------|
| EXP-13 | dm512 基线 | 64-65 | 0.085 | 0.141 | 0.316 | 25.27 | 34.41 | 0.539 | result.txt + AGENTS.md |
| **EXP-14** ✓ | **dm128 基线（当前）** | **67-68** | **0.079** | **0.125** | **0.304** | **23.87** | **33.29** | **0.568** | result.txt + npy 反算 |
| EXP-15 | dm128 dry run | PatchTST_supervised/result.txt:1-2 | 0.565 | 0.563 | 0.813 | — | — | — | ⚠️ 待确认 d_ff（2048 还是 256？如为 256 则与 EXP-14 同时改变了 dm+df 两个变量） |

**EXP-14 误差分层（来源：AGENTS.md）：**
- SSN 0-50: MAE=6.6
- SSN 50-100: MAE=12.4
- SSN 100-150: MAE=27.1
- SSN >150: MAE=68.8

**EXP-14 npy 数据（✓ 服务器可查）：**

| 指标 | z-score 值 | 物理值 |
|------|-----------|--------|
| MAE (全 47 步平均) | 0.125 | 23.87 |
| MAE (step 0 only) | 0.052 | 9.08 |
| R² | — | 0.568 |

反算代码：
```python
import numpy as np; import pandas as pd
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('PatchTST_supervised/dataset/sunspot_with_cycle.csv')
scaler = StandardScaler(); scaler.fit(df['ssn'].values[:3119].reshape(-1,1))
pred_z = np.load('results/sunspot_PatchTST_custom_ftM_sl96_ll48_pl24_dm128_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0/pred.npy')
true_z = np.load('results/sunspot_PatchTST_custom_ftM_sl96_ll48_pl24_dm128_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0/true.npy')
pred_phy = scaler.inverse_transform(pred_z[:,:,2].reshape(-1,1)).reshape(47,24)
true_phy = scaler.inverse_transform(true_z[:,:,2].reshape(-1,1)).reshape(47,24)
mae_all = np.mean(np.abs(pred_phy - true_phy))    # 23.87
mae_s0  = np.mean(np.abs(pred_phy[0] - true_phy[0]))  # 9.08
```

Files:
- pred.npy: `/root/code/patchTST-sunspot-prediction/results/sunspot_PatchTST_custom_ftM_sl96_ll48_pl24_dm128_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0/pred.npy`
- true.npy: `/root/code/patchTST-sunspot-prediction/results/sunspot_PatchTST_custom_ftM_sl96_ll48_pl24_dm128_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0/true.npy`

### 3.4 Level 2 相关代码（服务器上）

M4 包络代码路径：`/root/code/patchTST-sunspot-prediction/PatchTST_supervised/shixiong_m4/coding/solar_cycle_optimization_framework_package(1)/`

### 3.5 探索目标

【留空】

### 3.6 收获

【留空】

### 3.7 不足

【留空】

---

## Stage 4: Level 3 残差预测（2026.07.13）

### 4.1 方案设计

- M4 Waldmeier 生成 Gamma 曲线（物理基线）
- PatchTST 预测 SSN 与 M4 包络之间的残差
- 训练用 best-fit 残差，测试期望降低 M4 预报误差

### 4.2 配置

延续 Stage 3 EXP-14 配置（dm128, seq=96, pred=24, patch16/8）。

代码位于 worktree：`/root/code/patchTST-sunspot-prediction-level3/`

关键代码文件（来源：AGENTS.md）：
- `PatchTST_supervised/prepare_level3_residual.py` — M4 校准 + 预报 + 残差生成
- `PatchTST_supervised/eval_level3_residual.py` — 残差预测评估

### 4.3 结果

来源：`/root/code/patchTST-sunspot-prediction-level3/result.txt` line 70-74

| 运行 | MSE(z) | MAE(z) | RSE |
|------|--------|--------|-----|
| run 1 | 0.111 | 0.154 | 0.384 |
| run 2 | 0.149 | 0.162 | 0.449 |

物理 MAE：M4 alone = 3.32，M4 + PatchTST 残差 = 4.48（退化）（来源：AGENTS.md）

### 4.4 失败原因分析

记录于 AGENTS.md：
> best-fit 包络的残差（训练集 std=10.16）无预测信号——每个周期 Gamma 偏差是独有不重复的结构。PatchTST 学到的是输出均值≈0，叠加到 M4 预报上只加噪声。

### 4.5 探索目标

【留空】

### 4.6 收获

【留空】

### 4.7 不足

【留空】

---

## Stage 5: 天花板探测（2026.07.17）

### 5.1 目的

探测纯数据驱动方法在太阳黑子预测中的性能表现：
1. 信息量饱和：seq_len 96→192→336
2. 模型结构：DLinear vs DLinear-I vs PatchTST

### 5.2 实验矩阵

代码位于 worktree：`/root/code/patchTST-sunspot-prediction-ceiling-probe/`

数据文件：`sunspot_with_cycle.csv`（Stage 3 的 1749+ 版本）
数据切分：同 Stage 3（train 3119 / val 132 / test 70）

#### 五组实验

| 实验 | 模型 | seq_len | 归一化 | lr | epoch | 训练脚本 | 日志文件 | ✓ 可查证 |
|------|------|---------|--------|-----|-------|---------|---------|---------|
| A | PatchTST | 96 | RevIN=1 | 0.0001 | 10 | run_sunspot_fixed.py | — | ✓ (npy + result.txt) |
| B | PatchTST | 192 | RevIN=1 | 0.0001 | 30 | run_seq192.py | log_seq192.txt | ✓ |
| C | PatchTST | 336 | RevIN=1 | 0.0001 | 30 | run_seq336.py | log_seq336.txt | ✓ |
| D1 | DLinear (ind=0) | 96 | RevIN=0, affine=0 | 0.005 | 30 | run_dlinear.py | log_dlinear.txt | ✓ |
| D2 | DLinear-I (ind=1) | 96 | RevIN=0, affine=0 | 0.005 | 30 | run_dlinear_i.py | log_dlinear_i.txt | ✓ |

**DLinear-I 完整参数（来源：log_dlinear_i.txt）：**
```
model=DLinear, seq_len=96, pred_len=24, enc_in=3, seed=2021
d_model=128, e_layers=2, d_ff=2048, kernel_size=25
revin=0, affine=0, individual=1, moving_avg=25
es=16, patience=5, epoch=30, lr=0.005, loss=mse, dropout=0.05
train=3000 samples, val=109, test=47
```

**统一参数（A-E）：** pred_len=24, enc_in=3, seed=2021, bs=16

**各组不同的关键参数：**
- A 组 lr=0.0001, 10 epochs；B-C 组 lr=0.0001, 30 epochs；D1-D2 组 lr=0.005, 30 epochs
- A-C 组 RevIN=1；D1-D2 组 RevIN=0, affine=0

### 5.3 结果

来源：`/root/code/patchTST-sunspot-prediction-ceiling-probe/result.txt` + npy 反算（反算代码同 §3.3 EXP-14 的方法，StandardScaler fit on 3119 train samples）

| 实验 | result.txt 行 | MSE(z) | MAE(z) | step 0 物理 MAE | 全步 物理 MAE | R² |
|------|--------------|--------|--------|----------------|-------------|-----|
| A PatchTST sl96 | 67-68 (复用 EXP-14) | 0.079 | 0.125 | **9.08** | 23.87 | 0.568 |
| B PatchTST sl192 | 76-77 | 0.069 | 0.120 | **12.30** | 22.02 | 0.625 |
| C PatchTST sl336 | 79-80 | 0.057 | 0.112 | **10.87** | 20.54 | 0.692 |
| D1 DLinear ind=0 | 70-71 | 0.051 | 0.106 | 19.04 | 20.31 | 0.722 |
| D2 DLinear-I ind=1 | 73-74 | 0.046 | 0.094 | 19.36 | **19.30** | **0.751** |

npy 文件路径（✓ 服务器可查）：
- A: `results/sunspot_PatchTST_custom_ftM_sl96_ll48_pl24_dm128_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0/pred.npy`
- B: `results/ceiling_seq192_PatchTST_custom_ftM_sl192_ll48_pl24_dm128_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_ceiling_seq192_0/pred.npy`
- C: `results/ceiling_seq336_PatchTST_custom_ftM_sl336_ll48_pl24_dm128_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_ceiling_seq336_0/pred.npy`
- D1: `results/ceiling_dlinear_DLinear_custom_ftM_sl96_ll48_pl24_ceiling_dlinear_0/pred.npy`
- D2: `results/ceiling_dlinear_i_DLinear_custom_ftM_sl96_ll48_pl24_ceiling_dlinear_i_0/pred.npy`

### 5.4 可比性说明

⚠️ **重要：PatchTST（A-C 组）和 DLinear（D1-D2 组）不可直接跨组比数值。**

不可比原因：
| 差异 | PatchTST 组 | DLinear 组 |
|------|------------|-----------|
| 归一化 | RevIN=1（模型内部 normalize + denormalize） | RevIN=0, affine=0（不归一化，依赖 StandardScaler 预处理） |
| 学习率 | 0.0001 | 0.005（50 倍差异） |
| 训练轮数 | 10 (sl96) / 30 (sl192/336) | 30 |
| 架构 | Transformer + patching + self-attention | Linear decomposition + linear layers |

**可比性总结：**
- A vs B vs C（PatchTST 内部）：可比 ✓，可讨论 seq_len 的影响
- D1 vs D2（DLinear 内部）：可比 ✓，可讨论 individual 通道独立的影响
- PatchTST vs DLinear：不可直接比 ✗（归一化、lr、架构均不同）

**之前 AGENTS.md 和 project_summary 中记录的「DLinear-I 全面超过 PatchTST (MAE=19.30 vs 20.54)」「硬天花板 MAE≈19.30」是基于跨组比较得出的，前提不成立，需修正。**

各组内部结论：
- PatchTST 组：seq_len 从 96→336，全步 MAE 从 23.87→20.54。更多上下文有帮助但边际递减
- DLinear 组：individual=1（每通道独立线性层）比 individual=0 更好（19.30 vs 20.31）
- PatchTST sl96 step 0（MAE=9.08）是其最佳单次预测
- DLinear-I 全步平均（MAE=19.30）是全步口径下最低

### 5.5 探索目标

【留空】

### 5.6 收获

【留空】

### 5.7 不足

【留空】

---

## Stage 6: Phase 0 文献奠基（2026.07.23 — 08.09）

### 6.1 文献清单

14 篇文献位于：`/root/code/patchTST-sunspot-prediction/literature/`

| # | 文件名 | 论文 |
|---|--------|------|
| 1 | `太阳黑子，来源可疑lrsp-2015-4.pdf` | Hathaway (2015) The Solar Cycle |
| 2 | `s41116-020-0022-z.pdf` | Petrovay (2020) Solar cycle prediction |
| 3 | `2205.13504v3.pdf` | DLinear (Zeng et al., AAAI 2023) |
| 4 | `2211.14730v2.pdf` | PatchTST (Nie et al., ICLR 2023) |
| 5 | `数据处理/Solar cycle prediction using a long short-term memory...` | Wang et al. (2021) RAA |
| 6 | `Forecasting_Sunspot_Time_Series_Using_Deep_Learnin.pdf` | Pala & Atici (2019) Solar Phys |
| 7 | `Forecasting_Solar_Cycle_25_Using_Deep_Neural_Netwo.pdf` | Benson et al. (2020) Solar Phys |
| 8 | `1-s2.0-S0273117724000371-main.pdf` | Kumar & Kumar (2024) ASR |
| 9 | `s11207-025-02510-3.pdf` | Kumar & Kumar (2025) Solar Phys |
| 10 | `stab1159.pdf` | Xiong et al. (2021) MNRAS |
| 11 | `A New Declining Phase Precursor and an Early Prediction of Cycle 26 Maximum.pdf` | Chapman (2026) ApJ |
| 12 | `s11207-025-02577-y.pdf` | Gerçeker et al. (2025) Solar Phys |
| 13 | `ARIMA预测第二十五周太阳黑子数月均值(1).pdf` | ARIMA 预测（中文） |
| 14 | `太阳黑子第25期的统计预报(5)(2).docx` | 中文统计预报（未成功解析） |

### 6.2 阅读笔记产出

- `literature/literature_reading_notes.md` — 14 篇论文精读摘要 + 横向对比
- `literature/literature_reading_notes_detailed.md` — 13 篇论文完整结构化详注（三句话、实验设计、精度数值、方法论判断）
- 中文 docx 文件未成功解析文本

### 6.3 探索目标

【留空】

### 6.4 收获

【留空】

### 6.5 不足

【留空】

---

---

## Stage 7: 新 Baseline B 复现 — EXP-16c（2026.08.10）

> ⚠️ EXP-16 和 EXP-16b 已宣告无效（误用 run_sunspot_fixed.py 而非原管线 run_longExp.py）。EXP-16c 是唯一有效版本。

### 目的

用 run_longExp.py（原管线）+ Baseline B 严格参数在 1749+ 数据上复现基线。

### 配置

| 参数 | EXP-16c | EXP-14 | 原始 Baseline B |
|------|--------|--------|----------------|
| 训练脚本 | run_longExp.py | run_longExp.py | run_longExp.py |
| 数据 | 1749+ (3321月) | 1749+ (3321月) | 1867+ (1905月) |
| features | MS | M | MS |
| seq_len | 132 | 96 | 132 |
| n_heads | 16 | 8 | 16 |
| e_layers | 3 | 2 | 3 |
| d_ff | 256 | 2048 | 256 |
| patch_len | 12 | 16 | 12 |
| stride | 6 | 8 | 6 |
| batch_size | 32 | 16 | 32 |
| epochs | 50 (ES@38) | 10 | 50 (ES@31) |
| itr | 1 | 1（推测） | 2（仅test第2轮） |

### 结果

| 指标 | EXP-16c | EXP-14 | dm512 | 原始 Baseline B |
|------|--------|--------|-------|----------------|
| step0 MAE | 11.21 | 9.08 | — | 13.02 |
| 全步 MAE | 24.54 | 23.87 | 25.27 | 23.91 |
| R² | 0.532 | 0.568 | 0.539 | — |
| E_r | -85.5 | -75.8 | — | -29.2 |
| **滚动 MAE** | **36.37** | 149.92 | 23.12 | 33.13 |
| 训练 epochs | 38 (ES@20) | 10 | 10 | 31 (ES@100) |

### 结论

**run_longExp.py 原管线复现成功。** step0=11.21 优于原始 Baseline B（13.02，1867+ 数据），说明 seq=132 + nh16 + el3 的架构在更宽数据上也有效。但滚动=36.37 差于原始 B（33.13）——1749+ 的宽 range（0-398）拉大 scaler std，在自回归累积时放大了误差。峰值-85.5 与 EXP-14（-75.8）同量级，是 StandardScaler + 宽数据范围的系统性压制，非训练轮数可解。

**配置文件**：见 project_docs/EXP-16c_experiment.md。checkpoint 在 `checkpoints/sunspot_...EXP-16c_0/`（run_longExp.py 路径，非 PatchTST_supervised/checkpoints/）。

---

## 附录 A: 配置对比总表

| 参数 | Stage 0 (Apr) | Stage 1-2 (May) | Stage 3-5 (Jun-Jul) |
|------|---------------|-----------------|---------------------|
| 数据文件 | ? | sunspot_1867-02_2025-10_original_sincos.csv | sunspot_with_cycle.csv |
| 时间范围 | 1749-2020 | 1867-2025 | 1749-2025 |
| 训练集行 | ~2964 | 1703 | 3119 |
| 特征数 | 5 | 4 | 3 (有效使用) |
| 归一化 | MinMax | z-score + RevIN=1 | z-score + RevIN=1 |
| seq_len | 132 | 132 | 96 |
| pred_len | 24 | 24 | 24 |
| d_model | 512 | 128 | 128 |
| n_heads | 8 | 16 | 8 |
| e_layers | 2 | 3 | 2 |
| d_ff | 2048 | 256 | 2048 |
| patch_len | 12 | 12 | 16 |
| stride | 6 | 6 | 8 |
| dropout | ? | 0.05 | 0.2 |
| epoch | 50 | 50 | 10/30 |

## 附录 B: 结果数值总表

| 实验 | 评估方式 | z-score MAE | 物理 MAE | R² | 出处 |
|------|---------|------------|----------|-----|------|
| Apr MinMax 基线 | ? | — | 25.80 | — | 用户 04.16 文档 |
| Apr MSE+ReLU | ? | — | — | — | 用户 04.16 文档 |
| May 实验 B | step 0 | ? | 13.02 | — | 用户 05.06 文档 |
| May H2 全步平均 | 全 47 步 ⚠️ 待确认计算口径 | — | 23.91 | — | 用户 05.07 文档 |
| May H1 滚动 | 全 70 个月 | — | 33.13 | — | 用户 05.07 文档 |
| Jun dm512 (EXP-13) | 全 47 步 | 0.141 | 25.27 | 0.539 | result.txt + AGENTS |
| Jun dm128 (EXP-14) | 全 47 步 | 0.125 | 23.87 | 0.568 | result.txt + npy 反算 |
| Jun dm128 (EXP-14) | step 0 | 0.052 | **9.08** | — | npy 反算 |
| Jul DLinear-I | 全 47 步 | 0.094 | 19.30 | 0.751 | result.txt + npy 反算 |
| Jul DLinear-I | step 0 | — | 19.36 | — | npy 反算 |
| Jul PatchTST sl336 | 全 47 步 | 0.112 | 20.54 | 0.692 | result.txt + npy 反算 |
| Jul PatchTST sl336 | step 0 | — | 10.87 | — | npy 反算 |
| Jul PatchTST sl192 | 全 47 步 | 0.120 | 22.02 | 0.625 | result.txt + npy 反算 |
| Jul M4 Waldmeier | Cycle 25 | — | 3.32 | — | AGENTS.md |
| Jul Level 3 残差 | ? | 0.154~0.162 | 4.48 | — | result.txt + AGENTS |
| **Aug EXP-17-0a (sl96)** | W1 全 125 步, MS | 0.371 | 25.24 | 0.736 | result.txt + npy 反算 |
| **Aug EXP-17-0a (sl96)** | W1 step 0, MS | — | 8.81 | — | npy 反算 |
| **Aug EXP-17-0b (sl336)** | W1 全 125 步, MS | 0.351 | 23.87 | 0.769 | result.txt + npy 反算 |
| **Aug EXP-17-0b (sl336)** | W1 step 0, MS | — | 11.39 | — | npy 反算 |
| **Aug EXP-17-0a-r2 (sl96, 串行)** | W1 全 125 步, MS | 0.357 | 24.27 | 0.757 | 最终模型口径，result.txt + npy 反算 |
| **Aug EXP-17-0b-r2 (sl336, 串行)** | W1 全 125 步, MS | 0.390 | 26.53 | 0.704 | 最终模型口径，过拟合。result.txt + npy 反算 |
| **Aug EXP-18-1a 补测 (pl12, 最佳模型)** | W1 全 125 步, MS | 0.376 | 25.54 | 0.739 | 08.14 补测口径，差基线 6.9% |
| **Aug EXP-18-1c 补测 (pl24, 最佳模型)** | W1 全 125 步, MS | 0.346 | 23.53 | 0.775 | 08.14 补测口径，优基线 1.4% 无差异 |
| **Aug EXP-18-2b (el3)** | W1 全 125 步, MS | 0.345 | 23.43 | 0.777 | 差 1.8% 无差异 |
| **Aug EXP-18-2c (el4)** | W1 全 125 步, MS | 0.344 | 23.35 | 0.774 | 差 2.2% 无差异；E_r=-11.8 留档 |
| **Aug EXP-18-3a (df512)** | W1 全 125 步, MS | 0.372 | 25.32 | 0.741 | 恶化 6.1% 淘汰 |
| **Aug EXP-18-3b (df1024)** | W1 全 125 步, MS | 0.365 | 24.83 | 0.752 | 恶化 4.0%，维持 2048 |
| **Aug EXP-18-4a (dm64)** | W1 全 125 步, MS | 0.356 | 24.17 | 0.763 | 差 1.3%，step0 劣 17.6% |
| **Aug EXP-18-4b (dm256)** | W1 全 125 步, MS | 0.363 | 24.70 | 0.751 | 恶化 3.5% 淘汰 |
| **Aug EXP-18-5a (do0.1)** | W1 全 125 步, MS | 0.337 | 22.92 | 0.785 | 改善 4.0% |
| **Aug EXP-18-5b (do0.2)** | W1 全 125 步, MS | 0.333 | **22.62** | **0.789** | **改善 5.2%，阶段1最终基线** |
| **Aug EXP-18-6a (relu)** | W1 全 125 步, MS | 0.338 | 22.96 | 0.785 | 差 1.5% 无差异 |
| **Aug EXP-18-7a (ind1)** | W1 全 125 步, MS | 0.355 | 24.13 | 0.764 | 恶化 6.7% 淘汰 |
| **Aug EXP-19-2 (W2, 最优配置)** | W2 全 109 步, MS | 0.292 | 20.03 | 0.579 | Cycle 24 弱周期；step0=36 接缝弱 |
| **Aug EXP-19-3 (W3, 最优配置)** | W3 全 48 步, MS | 0.309 | 21.11 | 0.684 | Cycle 25 部分；E_r=-70.2 峰值压制 |
| **Aug W1+W2 平均** | 主决策指标 | — | **21.33** | — | 较初始基线 23.87 改善 10.6% |
| Aug EXP-20-1 (revin=0) | W1 全125步 | 0.334 | 22.72 | 0.791 | 滚动崩溃1967万 |
| Aug EXP-20-2a (pl=12) | W1 全137步 | 0.315 | 21.38* | 0.813 | *跨pl不可比；滚动52.21 |
| Aug EXP-20-2b (pl=48) | W1 全101步 | 0.354 | 24.05* | 0.756 | *跨pl不可比；滚动崩溃37万 |
| Aug EXP-20-3 (wmse a=1.0) | W1 全125步 | 0.381 | 25.92 | 0.730 | 滚动峰-2.2(改善92%) |
| Aug EXP-20-4 (sqrt) | W1 全125步 | 0.296 | 21.40 | 0.800 | 滚动27.80(改善45%) |
| Aug EXP-21-1 (pow07) | W1 全125步 | 0.322 | 22.15 | 0.793 | 滚动22.50 |
| Aug EXP-21-2 (pow23) | W1 全125步 | 0.321 | 22.15 | 0.792 | 滚动22.13(变换系最优) |
| Aug EXP-21-3 (log1p) | W1 全125步 | 0.249 | 21.85 | 0.769 | 全步E_r=+1.3 |
| Aug EXP-22-1 (mae) | W1 全125步 | 0.337 | 22.88 | 0.780 | 滚动124恶化 |
| Aug EXP-22-2 (wmse a=0.5) | W1 全125步 | 0.358 | 24.35 | 0.760 | 滚动峰-9.7 |
| Aug EXP-22-3 (wmse_th) | W1 全125步 | 0.354 | 24.03 | 0.765 | 滚动峰-6.2+滚动33.92(loss系最优) |
| Aug EXP-22-4 (asym) | W1 全125步 | 0.383 | 26.04 | 0.731 | 滚动峰转高估+25.5 |
| Aug EXP-23-1 (block24无重叠) | W1/W2/W3 轨迹 | — | 21.29/27.28/23.75 | — | 优于滚动44-59% |
| Aug EXP-23-2 (block24重叠12) | W1/W2/W3 轨迹 | — | 58.23/56.76/56.03 | — | 重叠法更差 |

## 附录 C: 缺失信息清单

| # | 缺失内容 | 所在阶段 | 备注 |
|---|---------|---------|------|
| 1 | Stage 0 训练命令（CLI） | Stage 0 | 位于用户本机 |
| 2 | Stage 0 训练入口脚本名称 | Stage 0 | run_longExp.py 还是 run_sunspot_fixed.py？ |
| 3 | Stage 0 seed 值 | Stage 0 | — |
| 4 | Stage 1 训练命令（CLI） | Stage 1 | 位于用户本机 |
| 5 | Stage 1 训练入口脚本名称 | Stage 1 | — |
| 6 | Stage 1 实验 B 的 z-score MAE | Stage 1 | 用户本机 npy 可查 |
| 7 | Stage 1 seed 值 | Stage 1 | — |
| 8 | Stage 3 配置变更原因 | Stage 3 | 谁决定的、为什么 |
| 9 | Stage 3 各实验的具体 run 命令 | Stage 3 | — |
| 10 | A-H 清单的实际执行情况 | Stage 2→3 | 哪些试了、结果如何 |

---

## Stage 8: 阶段1 参数搜索 — Round 0 (2026-08-12)

### 目的

在 features=MS, epochs=50, W1(Cycle 23) 回测窗口上，用唯一变量 seq_len 96 vs 336 确定后续所有实验的 seq_len 骨架。

### 新基础设施

- 新增 `--test_start` / `--test_end` CLI 参数，支持多回测窗口按年月切分
- 修复 `--activation` 不传递给 PatchTST backbone 的 bug
- 修复 `exp_main.py` torch.load 的 PyTorch 1.11 兼容性
- 新增 `save_config.py` 支持 test_start/test_end 记录

### EXP-17-0a (sl96) / EXP-17-0b (sl336)

详见 `EXP-17-Round0_experiment.md`。全步 MAE：sl96=25.24, sl336=23.87。sl336 改善 5.4%，选为后续基线。

| 指标 | 0a (sl96) | 0b (sl336) |
|------|:---:|:---:|
| 全步 MAE | 25.24 | **23.87** |
| step0 MAE | **8.81** | 11.39 |
| R² | 0.736 | **0.769** |
| E_r | -36.2 | **-17.8** |
| 滚动 MAE | **43.51** | 71.79 |

> ⚠️ **2026-08-13 口径修正**：Round 0 旧数据（0a/0b）为"最佳 checkpoint 口径"（训练被 timeout 打断后经 `--is_training 0` 补测，加载 EarlyStopping 最佳权重），而 Round 1 及以后为"第 50 轮最终模型口径"（一次性 train→test）。两代口径不一致。已决定 2026-08-13 串行重跑（EXP-17-0a-r2 / EXP-17-0b-r2，最终模型口径），旧数据保留仅作追溯。官方基线以 -r2 为准。

### 2026-08-13 -r2 重跑结果 + 门禁触发

| 指标 | 0a-r2 (sl96) | 0b-r2 (sl336) |
|------|:---:|:---:|
| 全步 MAE | 24.27 | 26.53 |
| step0 | 18.21 | 12.87 |
| R² | 0.757 | 0.704 |
| E_r | -18.8 | -15.0 |

- 可复现性门禁未通过：0b-r2(26.53) vs 旧0b(23.87) 偏差 11.1% > 5%
- 根因：sl336 在 50ep 下严重过拟合（val loss epoch4=0.288 → epoch50=0.36，连续 46 轮恶化）；sl96 全程 0.25-0.32 稳定
- 代码根源：exp_main.py L229 加载最佳权重的代码 2026.04.02 被注释，train→test 用最终模型
- 决策：方案 B（零代码命令链：train && 补测，统一最佳模型口径），经补测+重审后定最终基线

---

## Stage 8b: 阶段1 参数搜索 — Round 1 patch_len/stride (2026-08-12)

### 目的

在 Round 0 基线 (sl336) 上，测试 patch_len/stride 三种组合（绑定，比值恒 2:1）。

### 结果

| 指标 | 1a (12,6) | 基线 (16,8) | 1c (24,12) |
|------|:---:|:---:|:---:|
| 全步 MAE | 26.85 | **23.87** | 25.84 |
| step0 MAE | 16.62 | 11.39 | **10.26** |
| R² | 0.693 | **0.769** | 0.720 |
| E_r | -31.6 | -17.8 | +7.6 |
| 150+ 分层 | 39.8 | **29.0** | 32.3 |

**结论**：patch_len=16/stride=8 全面胜出（1a 恶化 12.5%，1c 恶化 8.3%）。参数固定为 16/8，不再搜索。详见 `EXP-18-Round1_experiment.md`。

---

## Stage 8c: 阶段1 参数搜索完成 — 最终配置与验证 (2026-08-14)

### 搜索历程（W1窗口，最佳模型口径，命令链模式）

| Round | 变量 | 结果 |
|-------|------|------|
| 0 | seq_len | 336 胜 (23.87 vs 25.24) |
| 1 | patch_len/stride | (16,8) 维持 |
| 2 | e_layers | 2 维持（三深度 <3% 无差异） |
| 3 | d_ff | 2048 维持（512 恶化 6.1%，单调敏感） |
| 4 | d_model | 128 维持（64/256 均更差） |
| 5 | dropout | **0.2 胜，改善 5.2%（22.62）** |
| 6 | activation | gelu 维持 |
| 7 | individual | 0 维持 |

### 最终配置

```
seq_len=336, patch_len=16, stride=8, e_layers=2, d_ff=2048, d_model=128,
n_heads=8, dropout=0.2, fc_dropout=0.2, head_dropout=0.0, individual=0,
activation=gelu, features=MS, epochs=50, patience=100, seed=2021,
口径=最佳val模型（命令链 train&&补测）
```

### 三窗口验证

| 窗口 | 全步 MAE | R² | E_r | 滚动 MAE | 滚动峰值预测 |
|------|:---:|:---:|:---:|:---:|:---:|
| W1 (Cycle 23) | 22.62 | 0.789 | -24.4 | 50.41 | 217.6/244.3 |
| W2 (Cycle 24) | 20.03 | 0.579 | +16.0 | 66.80 | 121.4/146.1 |
| W3 (Cycle 25) | 21.11 | 0.684 | -70.2 | 42.05 | 85.4/216.0 |

W1+W2 平均全步 = 21.33（较初始基线 23.87 改善 10.6%）。

### 核心结论

1. 调参有效但有限：唯一实质改善来自 dropout（5.2%），其余维度在基线附近饱和
2. **峰值压制是系统性瓶颈**：三窗口 150+ 段 MAE 28-57，滚动峰值低估 -25 至 -131（W3 预测 85.4 vs 真实 216）。印证 MSE loss + StandardScaler 对高值的压制，参数不可解
3. 极小值接缝处最弱（W2 step0=36）
4. 阶段2计划：数据更新到 2026.07 后用全部数据+此配置重新训练（不调参）；阶段3滚动外推

---

## Stage 9: 探索期 — 变换/loss/外推策略三系列（2026-08-14 ~ 08-15）

> 阶段1 之后、开学之前的自主探索。判定标准为待议项（见 judgment_criteria_discussion_2026-08-15.md，待景修批注）。
> 全部实验：W1 窗口（除 EXP-23 三窗口）、最佳模型口径、单种子 seed=2021。

### 目的

阶段1 结论是"参数已饱和、峰值压制是瓶颈"。探索期从三个更外层切入：
数据表示（变换）、训练目标（loss）、使用策略（block vs 滚动）。

### 三系列结果摘要

| 系列 | 最佳候选 | 关键数字 |
|------|---------|---------|
| 变换（EXP-21） | pow23 / log1p | 滚动 MAE 22.13/22.63（基线 50.41，改善 55%+）；log1p 全步 E_r=+1.3 |
| loss（EXP-22） | wmse_th（阈值加权） | 滚动峰 -6.2（改善 77%）、滚动 33.92（改善 33%）、低值损伤最小 |
| 外推策略（EXP-23） | block24 无重叠 | 三窗口 MAE 21.29/27.28/23.75，优于滚动 44-59% |

### 核心发现

1. **机制4（新识破）**：右偏分布是滚动误差爆炸的主要放大器——所有变换让滚动改善 45-56%
2. **机制3 证伪**：RevIN 不是峰值压制元凶，是滚动稳定性支柱（revin=0 滚动崩溃）
3. **口径分裂**：全步（125 独立窗口）与滚动/block（自回归轨迹）的结论经常相反（wmse_th 全步恶化 7% 但滚动改善 33%）
4. **加权家族的权衡曲线**：α=1.0（峰-2.2/低值15.8）→ α=0.5（峰-9.7/低值14.4）→ 阈值（峰-6.2/低值13.7），阈值效率最高
5. **block 优于滚动**：回填频率低 24 倍 → 误差自回归累积机会少；重叠法反而更差

### 待办（判定批注后）

- 组合实验：pow23+wmse_th（C-1）、block+wmse_th（C-2），按"新基线+单变量"框架
- 极小值专项评估（阶段3 前置）
- 优胜配置 W2/W3 验证
