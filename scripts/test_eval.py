"""
test_eval.py — eval_metrics 回归测试

用 EXP-14 的已知结果验证 eval_metrics.py 没被改坏。
跑一次：python scripts/test_eval.py
结果：PASS 或 FAIL（带实际值）。
"""

import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

RESULT_DIR = "results/sunspot_PatchTST_custom_ftM_sl96_ll48_pl24_dm128_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0"
DATA_CSV = "PatchTST_supervised/dataset/sunspot_with_cycle.csv"

if not os.path.exists(os.path.join(RESULT_DIR, "pred.npy")):
    print(f"SKIP: {RESULT_DIR} 不存在，无法验证")
    sys.exit(0)

output = subprocess.check_output(
    [sys.executable, "scripts/eval_metrics.py", RESULT_DIR,
     "--scaler", "standard", "--data_csv", DATA_CSV, "--num_train", "3119"],
    text=True
)

checks = {
    "step0_MAE": (9.08, 0.5),
    "full_MAE": (23.87, 1.0),
    "R2": (0.568, 0.05),
}

failed = False
for line in output.split("\n"):
    for name, (expected, tol) in checks.items():
        if name in line:
            val = float(line.split(":")[1].strip())
            ok = abs(val - expected) <= tol
            status = "PASS" if ok else f"FAIL (expected {expected}±{tol}, got {val})"
            print(f"{name}: {status}")
            if not ok:
                failed = True

if failed:
    print("\n❌ REGRESSION DETECTED — eval_metrics.py 输出不一致")
    sys.exit(1)
else:
    print("\n✅ All checks passed")
