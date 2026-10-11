# 分支登记表 — BRANCHES.md（分支太多时看这一页）

> 创建：2026-10-11（景修提出"分支太多、不知道该看哪个"）；**同日完成收敛**。
> **铁律：读任何文件前，先确认自己在哪条分支。** 状态类信息只在 `STATE.md`（主线内）。
> 分支生命周期规则见 §四（已生效，写进 `AGENTS.md`「分支与账本规范（方案 C）」）。

---

## 一、当前活着的分支（收敛后只剩 2 条 + main）

| 分支 | worktree 路径 | 用途 | 文档 | 状态 |
|---|---|---|---|---|
| **`EXP-17-stage1`** | `/output/patchTST-sunspot-prediction` | **唯一主线**：状态 + 账本 + 已定论结果（12 条线内容全并入） | ✅ 最新 | 🟢 **推荐入口** |
| `GPU-basemodel-search` | `/output/GPU-basemodel` | **工作线**：GPU 基模探索（lr / bs 等） | 落后主线 1 个提交（方案 C：状态只在主线读写） | 🟢 实验用 |
| `main` | — | **临时/遗留标签**（2026-08-10 冻结） | ❌ 旧 | ⚪ 暂不管（景修 10-11 决定） |

## 二、已归档（2026-10-11 收敛时删除标签与 worktree；内容全部已在主线）

| 分支 | 原用途 | 处置 |
|---|---|---|
| `env-newserver-2026-10` | 文档线（10-10 换机文档、纪律、索引行、日记、A/B 对齐） | ✅ 内容已并入主线，标签已删 |
| `env-fix-torch28-numpy2` | torch2.8 / numpy2 代码修复 | ✅ 同上 |
| `GPU-dropout-abl` | EXP-27~30 消融 + EXP-23-R2/R3 + 泄露核查 + 谷值讨论稿 + 想法库 | ✅ 同上（worktree 已移除，checkpoint 已归档） |
| `GPU-exp26` | EXP-26 同环境对照 | ✅ 同上 |
| `GPU-exp25` | EXP-25（作废实验对照点），独有详情节已补进主线 | ✅ 同上 |

> ⚠️ **远端**目前仍留着这 5 条同名分支的指针（内容都已包含在主线里）。要不要删远端指针，另行决定。
> ⚠️ checkpoint 归档位置：主线 `checkpoints/archive/from-<分支>/`（107 MB，gitignore 不入库）。

## 三、远端历史分支（归档弃用，一般不看不改）

`EXP-16` / `ceiling-probe-v1` / `d_model_512_baseline_v1` / `level3-residual-prediction` /
`phase-method-a` / `phase-method-a-v2` / `phase-method-b` / `phase-training`

→ 2026-05 ~ 08 的旧实验快照，里程碑性质。**除非考古，不用看。**

## 四、收敛记录（✅ 2026-10-11 全部完成）

**结果：一条主线 + 一条工作线 + 两个 worktree + 一个临时 main。**

| 步 | 动作 | 状态 |
|:--:|---|---|
| 1 | `GPU-exp25` → `GPU-basemodel-search`（含 `env-newserver-2026-10`、`GPU-exp26`、`GPU-dropout-abl`、`env-fix`）依次合并到 **`EXP-17-stage1`** | ✅ **已完成（2026-10-11）** |
| 2a | checkpoint 归档：三个待归档 worktree 的 `.pth`（**gitignore 不入库，删目录即丢**）已复制到 `checkpoints/archive/from-<分支>/`（107 MB） | ✅ 已完成 |
| 2b | worktree 收敛到 2 个：主线 `/output/patchTST-sunspot-prediction`（已在 `EXP-17-stage1`）+ 工作线 `/output/GPU-basemodel`；已移除 `GPU-exp25/26/dropout-abl` 三个 worktree | ✅ **已完成** |
| 3 | 删临时分支标签（5 条）+ 推送主线 | ✅ **已完成**（`git push origin EXP-17-stage1`，快进 `1f1038b → 86297ef`，无 force） |

> **顺序铁律**：先合并、后删标签。合并前"内容都在主线"这句话不成立——现在才成立。
> **删 worktree ≠ 删提交**，但会连带删掉该目录里 gitignore 的 checkpoint（故 2a 先行）。

**收敛第 1 步完成校验（2026-10-11）**：`git merge-base --is-ancestor` 对 6 条分支全 PASS
（`env-fix-torch28-numpy2` / `env-newserver-2026-10` / `GPU-exp25` / `GPU-exp26` / `GPU-dropout-abl` / `GPU-basemodel-search`），
即**主线已包含全部内容**；主线文档三查（`scripts/check_docs.py`）全 PASS。
合并冲突两处均按"并集"处理：`result.txt`（两侧不同实验的行都保留）、`experiment_history.md`
（索引表取文档线版本 + 从 `GPU-exp25` 补回 72 行详情节）。

**主线命名建议**：`EXP-17-stage1` 是"阶段1 冻结"的历史名，作为长期主线容易误解；
可选①把 `main` 对齐到同一提交（一次普通 merge，不用 force），以后日常看 `main`；
可选②保留现名并在 `STATE.md` 写明。**待景修定。**

**分支生命周期规则（✅ 2026-10-11 景修批准，已写进 `AGENTS.md`「分支与账本规范（方案 C）」）**：
1. 只有两类分支：**主线**（唯一权威）+ **工作线**（当前 = `GPU-basemodel-search`）。
2. 新实验默认在**工作线内**做完（用 commit + 记录文件区分），**不再为每条实验线开分支**。
3. 确实需要并行（与主线大改动同时进行）时才开分支，且**当天登记到本文件**，完成后 7 天内合并或标归档。
4. 归档前必须确认内容已在主线（`git merge-base --is-ancestor <branch> <主线>` 返回 0）。

> 铁律 3 已机器兜底：`scripts/check_docs.py` 的 **D 查**会对"本地存在但未登记本文件的分支"报警。
> 启动方式：每轮收尾跑 `python3 scripts/check_docs.py`（四查；有 FAIL 非零退出）。
