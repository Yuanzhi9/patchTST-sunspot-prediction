"""Audit PR24 multitask V2 result/config coverage and split alignment."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CYCLES = ("SC20", "SC21", "SC24")
SEEDS = (2019, 2020, 2021, 2022, 2023)
VARIANTS = {
    "fixed": ("B26M-MT-CHRONO", "fixed", 0.1),
    "unc": ("B26M-MT-CHRONO-UNC", "uncertainty", 1.0),
    "w030": ("B26M-MT-CHRONO-W030", "fixed", 0.3),
}


def main() -> None:
    errors: list[str] = []
    checks: list[dict] = []
    for variant, (prefix, weighting, weight) in VARIANTS.items():
        for cycle in CYCLES:
            cv_path = ROOT / f"configs/chrono_cycle{cycle[2:]}_holdout_2026-11.json"
            cv_spec = json.loads(cv_path.read_text(encoding="utf-8"))
            expected_outer = cv_spec["folds"][0]["outer_val"]
            des = f"{prefix}-{cycle}-V2"
            for seed in SEEDS:
                result_dirs = sorted(
                    (ROOT / "results").glob(f"*_{des}_F1_seed{seed}")
                )
                if len(result_dirs) != 1:
                    errors.append(f"{variant} {cycle} seed{seed}: {len(result_dirs)} result dirs")
                    continue
                result_dir = result_dirs[0]
                for name in ("pred.npy", "true.npy", "aux_pred.npy", "cv_window_meta.json"):
                    if not (result_dir / name).exists():
                        errors.append(f"{result_dir.name}: missing {name}")
                meta = json.loads(
                    (result_dir / "cv_window_meta.json").read_text(encoding="utf-8")
                )
                observed_outer = meta["fold_spec"]["outer_val"]
                for key in ("start", "end", "start_date", "end_date"):
                    if observed_outer[key] != expected_outer[key]:
                        errors.append(
                            f"{result_dir.name}: outer_val {key} mismatch"
                        )

                configs = sorted(
                    (ROOT / "configs").glob(
                        f"{des}_mtall_F1_seed{seed}_*.json"
                    )
                )
                if len(configs) != 1:
                    errors.append(
                        f"{variant} {cycle} seed{seed}: {len(configs)} config files"
                    )
                    continue
                params = json.loads(configs[0].read_text(encoding="utf-8"))["params"]
                if int(params["random_seed"]) != seed:
                    errors.append(f"{configs[0].name}: seed mismatch")
                if params["mt_weighting"] != weighting:
                    errors.append(f"{configs[0].name}: weighting mismatch")
                if abs(float(params["mt_aux_weight"]) - weight) > 1e-12:
                    errors.append(f"{configs[0].name}: aux weight mismatch")
                checks.append(
                    {
                        "variant": variant,
                        "cycle": cycle,
                        "seed": seed,
                        "result_dir": str(result_dir),
                        "config": str(configs[0]),
                    }
                )

    for cycle in CYCLES:
        for base_name in (
            "pr24_multitask_cycle_holdout_v2",
            "pr24_multitask_cycle_holdout_unc_v2",
            "pr24_multitask_cycle_holdout_w030_v2",
        ):
            base = ROOT / "analysis" / base_name / cycle
            for seed in SEEDS:
                for path in (
                    base / f"p24_update_seed{seed}_extrema.json",
                    base / f"p24_recursive_multitask_seed{seed}_extrema.json",
                    base / f"p24_update_seed{seed}/p24_update_curve.csv",
                    base
                    / f"p24_update_seed{seed}/p24_recursive_multitask_curve.csv",
                ):
                    if not path.exists():
                        errors.append(f"missing analysis output: {path}")

    payload = {
        "status": "PASS" if not errors else "FAIL",
        "expected_runs": len(CYCLES) * len(SEEDS) * len(VARIANTS),
        "checked_runs": len(checks),
        "errors": errors,
        "checks": checks,
    }
    out = ROOT / "analysis/pr24_multitask_v2_audit.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
