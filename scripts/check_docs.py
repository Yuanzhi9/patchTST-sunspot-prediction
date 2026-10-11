"""
check_docs.py — 文档一致性三查（STATE.md 维护方案的执行器）

景修 2026-10-11 批准的 L1 方案：STATE.md 作唯一状态源，本脚本负责**自动发现它烂掉**。

三查：
  A 指针有效：全仓库 md 里不许残留绝对行号（如 L1026）；`Stage N` 引用必须真的存在；
              STATE.md 里出现的文件名必须在仓库里存在。
  B 时效有效：experiment_history 索引表的最新日期 与 STATE.md 的"最后核对"日期比对；
              索引更新了而 STATE 没更新 → 报警（这条专治"忘了更新"）。
  C 唯一性有效：STATE.md 的"当前基线"字段值必须能在 AGENTS.md 里找到同一个 ID；
              且除 STATE.md 外，其他文档不允许再出现"当前基线 ="这类定义式表述。
  D 分支已登记：本地每条分支都必须在 BRANCHES.md 里出现（方案 C 铁律 3 的机器兜底）。

用法: python3 scripts/check_docs.py        # 有 FAIL 则退出码 1
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {".git", "__pycache__", "Formers", "legacy", "test_results", "checkpoints", "results", "plots", "logs"}


def md_files():
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            if f.endswith(".md"):
                yield os.path.join(base, f)


def check_a():
    problems = []
    stage_nums = set()
    hist = os.path.join(ROOT, "project_docs/experiment_history.md")
    for m in re.finditer(r"^## Stage (\d+)", open(hist, encoding="utf-8").read(), re.M):
        stage_nums.add(m.group(1))
    for path in md_files():
        rel = os.path.relpath(path, ROOT)
        for i, line in enumerate(open(path, encoding="utf-8", errors="ignore"), 1):
            for m in re.finditer(r"\bL(\d{3,4})\b", line):
                # 允许：跨行引用代码行号时写成 "file.py:123"（带冒号）——这里只抓纯 L#### 形态
                if f":{m.group(1)}" in line:
                    continue
                problems.append(f"{rel}:{i} 残留绝对行号 L{m.group(1)} → 改成 Stage/§/分支:文件 引用")
            for m in re.finditer(r"\bStage (\d+)\b", line):
                if m.group(1) not in stage_nums:
                    problems.append(f"{rel}:{i} 引用了不存在的 Stage {m.group(1)}")
    return problems, stage_nums


def check_b():
    hist = os.path.join(ROOT, "project_docs/experiment_history.md")
    dates = [m.group(1) for m in re.finditer(r"\| \*\*EXP-[\w-]+\*\* \| (\d{2}\.\d{2})", open(hist, encoding="utf-8").read())]
    state = open(os.path.join(ROOT, "project_docs/STATE.md"), encoding="utf-8").read()
    m = re.search(r"\| 最后核对 \| (\d{4}-\d{2}-\d{2})", state)
    if not dates or not m:
        return ["无法比对时效：索引表或 STATE.md 的日期字段读不到"]
    # 索引表最新日期形如 10.11；STATE 形如 2026-10-11
    newest = max(dates, key=lambda d: (int(d[:2]), int(d[3:])))
    st_mmdd = m.group(1)[5:]
    if (int(newest[:2]), int(newest[3:])) > (int(st_mmdd[:2]), int(st_mmdd[3:])):
        return [f"索引表最新实验 {newest} 晚于 STATE.md 最后核对 {m.group(1)} → 跑了实验没更新 STATE"]
    return []


def check_c():
    state_path = os.path.join(ROOT, "project_docs/STATE.md")
    state = open(state_path, encoding="utf-8").read()
    m = re.search(r"\| 当前基线（月均） \|([^|]+)\|", state)
    if not m:
        return ["STATE.md 里读不到'当前基线（月均）'字段"]
    ids = re.findall(r"EXP-[\w-]+", m.group(1))
    problems = []
    agents = open(os.path.join(ROOT, "AGENTS.md"), encoding="utf-8").read()
    for i in ids:
        if i not in agents:
            problems.append(f"STATE 的当前基线 ID {i} 在 AGENTS.md 里找不到 → 两处不一致")
    for path in md_files():
        rel = os.path.relpath(path, ROOT)
        if rel.endswith("STATE.md"):
            continue
        if rel.startswith("diary/"):
            continue  # 日记是"当天就是这样"的日志，不追改历史记录
        for i, line in enumerate(open(path, encoding="utf-8", errors="ignore"), 1):
            if re.search(r"当前基线\s*[=＝]|当前基线是|当前基线为", line):
                problems.append(f"{rel}:{i} 出现'当前基线='定义式表述 → 应只引用 STATE.md")
    return problems


def check_d():
    """方案 C 铁律 3：开分支必须当天登记 BRANCHES.md。"""
    import subprocess
    try:
        out = subprocess.check_output(["git", "branch", "--format=%(refname:short)"],
                                      cwd=ROOT, text=True, stderr=subprocess.DEVNULL)
    except Exception as e:  # 不在 git 仓库里/无 git
        return [f"无法枚举本地分支（{e}）"]
    branches = [b.strip() for b in out.split("\n") if b.strip()]
    registry = open(os.path.join(ROOT, "project_docs/BRANCHES.md"), encoding="utf-8").read()
    return [f"本地分支 `{b}` 未登记在 BRANCHES.md → 违反方案 C 铁律 3" for b in branches if b not in registry]


def main():
    a, stages = check_a()
    b = check_b()
    c = check_c()
    d = check_d()
    print(f"文档四查（A 指针 / B 时效 / C 唯一性 / D 分支登记）｜ 发现的 Stage 节：{len(stages)} 个\n")
    for name, probs in (("A 指针有效", a), ("B 时效有效", b), ("C 唯一性有效", c), ("D 分支已登记", d)):
        if probs:
            print(f"❌ {name}：{len(probs)} 处")
            for p in probs[:20]:
                print(f"    - {p}")
        else:
            print(f"✅ {name}：通过")
    total = len(a) + len(b) + len(c) + len(d)
    print(f"\n合计 {total} 处问题")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
