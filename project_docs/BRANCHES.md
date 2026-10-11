# 分支登记表 — BRANCHES.md（分支太多时看这一页）

> 创建：2026-10-11（景修提出"分支太多、不知道该看哪个"）。
> **铁律：读任何文件前，先确认自己在哪条分支。** 状态类信息只在 `STATE.md`；
> `AGENTS.md` / `project_roadmap.md` 在**未合并的旧分支上可能是过期版本**（本页标注了哪几条是旧的）。
> 分支生命周期规则见文末 §四（提案，待景修批）。

---

## 一、当前活着的分支（有 worktree，会继续动）

| 分支 | worktree 路径 | 用途 | 文档是不是最新 | 状态 |
|---|---|---|---|---|
| **`GPU-basemodel-search`** | `/output/GPU-basemodel` | **工作线**：GPU 基模探索（lr / bs）；已含代码修复 + 全部历史文档 + 消融记录 + `STATE.md` | ✅ 最新 | 🟢 **推荐入口** |
| **`env-newserver-2026-10`** | `/output/patchTST-sunspot-prediction` | **文档线**：10-10 换机文档、纪律（坑 29-34）、索引行、日记、A/B 对齐、history 修正 | ✅ 最新（无 STATE.md） | 🟢 文档维护 / 合并来源 |
| `GPU-dropout-abl` | `/output/GPU-dropout-abl` | EXP-27~30 dropout 消融 + EXP-23-R2/R3 评估复跑 + 泄露核查 + 谷值讨论稿 + 想法库 | ❌ 旧（09-18 版） | 🟡 内容已被工作线包含，可归档 |
| `GPU-exp26` | `/output/GPU-exp26` | EXP-26（W1 do0.2 同环境对照） | ❌ 旧 | 🟡 已被 `GPU-dropout-abl` 包含 |
| `GPU-exp25` | `/output/GPU-exp25` | EXP-25（**作废**实验，留作对照点）+ 对拍图 | ❌ 旧 | 🟠 **独有内容**（别处没有，不能直接删） |

## 二、没有 worktree 的本地分支

| 分支 | 用途 | 状态 |
|---|---|---|
| `env-fix-torch28-numpy2` | torch 2.8 / numpy 2.2 代码修复 + `roll_eval` 的 `n_roll` 与峰值符号修复 | 🟡 已并入工作线（同一批提交） |
| `EXP-17-stage1` | **约定的合并目标**（2026-09-18 冻结点 `1f1038b`） | 🔵 待合并 |
| `main` | 老主线（2026-08-10） | ⚪ 冻结未动（⚠️ 远端 `origin/HEAD` 仍指向它 → 新人 clone 会先看到这个 **旧** 状态） |

## 三、远端历史分支（归档弃用，一般不看不改）

`EXP-16` / `ceiling-probe-v1` / `d_model_512_baseline_v1` / `level3-residual-prediction` /
`phase-method-a` / `phase-method-a-v2` / `phase-method-b` / `phase-training`

→ 2026-05 ~ 08 的旧实验快照，里程碑性质。**除非考古，不用看。**

## 四、收敛计划（提案，待景修批；涉及删分支/worktree 属红线）

**目标状态：一条主线 + 一条工作线 + 两个 worktree。**

| 步 | 动作 | 结果 |
|:--:|---|---|
| 1 | 把 `GPU-exp25`（独有内容）→ `GPU-dropout-abl` / `GPU-exp26`（已含）→ `env-newserver-2026-10` → `GPU-basemodel-search` 依次合并到 **`EXP-17-stage1`** | 主线 = 全量超集 |
| 2 | worktree 收敛到 2 个：主线（`/output/patchTST-sunspot-prediction` 切到 `EXP-17-stage1`）+ 工作线（`/output/GPU-basemodel`）；移除其余 3 个 worktree（**移除 worktree 不删提交**） | 只剩两个入口 |
| 3 | 删除已并入的临时分支标签（本地 + 可选远端）；**一次性推送**（顺序建议：先收敛、再推，推上去就是干净状态） | 远端 = `main` + `EXP-17-stage1` + 工作线 + 8 条历史归档 |

**主线命名建议**：`EXP-17-stage1` 是"阶段1 冻结"的历史名，作为长期主线容易误解；
可选①把 `main` 对齐到同一提交（一次普通 merge，不用 force），以后日常看 `main`；
可选②保留现名并在 `STATE.md` 写明。**待景修定。**

**分支生命周期规则（提案，需写进 AGENTS.md 才算生效）**：
1. 只有两类分支：**主线**（唯一权威）+ **工作线**（当前 = `GPU-basemodel-search`）。
2. 新实验默认在**工作线内**做完（用 commit + 记录文件区分），**不再为每条实验线开分支**。
3. 确实需要并行（与主线大改动同时进行）时才开分支，且**当天登记到本文件**，完成后 7 天内合并或标归档。
4. 归档前必须确认内容已在主线（`git merge-base --is-ancestor <branch> <主线>` 返回 0）。
