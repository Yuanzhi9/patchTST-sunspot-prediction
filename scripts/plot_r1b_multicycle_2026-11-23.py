"""Plot R1B cycle-level peak validation on both reporting tracks."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "analysis/r1b_multicycle/validation_predictions.csv"
DETAIL = ROOT / "analysis/r1b_multicycle/validation_detail.csv"
OUT = ROOT / "analysis/r1b_multicycle"

TARGETS = ("raw", "smooth")
TARGET_TITLES = {
    "raw": "Raw monthly maximum",
    "smooth": "13-month centered maximum",
}
COLORS = {
    "observed": "#111111",
    "W0_ridge": "#D55E00",
    "W0_random_forest": "#009E73",
    "W1_AA_CONTROL": "#0072B2",
    "W2_AA": "#CC79A7",
    "median": "#7F7F7F",
}


def load_expanding() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(SOURCE)
    expanding = frame.loc[frame["cv"] == "expanding"].copy()
    aggregate = (
        expanding.groupby(
            ["target_kind", "feature_set", "model", "test_cycle"],
            as_index=False,
        )
        .agg(
            true=("true", "first"),
            pred_median=("pred", "median"),
            pred_min=("pred", "min"),
            pred_max=("pred", "max"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["target_kind", "test_cycle", "feature_set", "model"])
    )
    return expanding, aggregate


def main() -> None:
    _, aggregate = load_expanding()
    detail = pd.read_csv(DETAIL)
    detail_expanding = detail.loc[detail["cv"] == "expanding"]
    aggregate.to_csv(OUT / "validation_curve_aggregate.csv", index=False)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14.5, 9.6),
        dpi=300,
        sharex="row",
        gridspec_kw={"width_ratios": (1.25, 1.0), "wspace": 0.17, "hspace": 0.48},
    )
    truth_offsets = {
        "raw_main": (14, -32, 14, 17, -32, 15),
        "smooth_main": (14, -32, 14, 15, -32, 15),
        "raw_aa": (15, -33, 15, 15),
        "smooth_aa": (15, -33, 15, 15),
    }
    for row, target_kind in enumerate(TARGETS):
        subset = aggregate.loc[aggregate["target_kind"] == target_kind]
        reference = (
            subset.loc[subset["feature_set"] == "W0"]
            .drop_duplicates("test_cycle")
            .set_index("test_cycle")
            .sort_index()
        )
        observed = reference["true"]

        median = (
            subset.loc[
                (subset["feature_set"] == "W0") & (subset["model"] == "median")
            ]
            .set_index("test_cycle")
            .sort_index()
        )
        ridge = (
            subset.loc[
                (subset["feature_set"] == "W0") & (subset["model"] == "ridge")
            ]
            .set_index("test_cycle")
            .sort_index()
        )
        forest = (
            subset.loc[
                (subset["feature_set"] == "W0")
                & (subset["model"] == "random_forest")
            ]
            .set_index("test_cycle")
            .sort_index()
        )
        aa_w1 = (
            subset.loc[
                (subset["feature_set"] == "W1_AA_CONTROL")
                & (subset["model"] == "random_forest")
            ]
            .set_index("test_cycle")
            .sort_index()
        )
        aa_w2 = (
            subset.loc[
                (subset["feature_set"] == "W2_AA")
                & (subset["model"] == "random_forest")
            ]
            .set_index("test_cycle")
            .sort_index()
        )

        main_ax = axes[row, 0]
        aa_ax = axes[row, 1]

        main_ax.plot(
            observed.index,
            observed,
            color=COLORS["observed"],
            marker="o",
            lw=2.4,
            label="Observed target-cycle peak",
            zorder=5,
        )
        main_ax.plot(
            median.index,
            median["pred_median"],
            color=COLORS["median"],
            marker="s",
            lw=1.4,
            ls="--",
            label="Climate median",
        )
        main_ax.plot(
            ridge.index,
            ridge["pred_median"],
            color=COLORS["W0_ridge"],
            marker="^",
            lw=1.7,
            label="W0 Ridge",
        )
        main_ax.plot(
            forest.index,
            forest["pred_median"],
            color=COLORS["W0_random_forest"],
            marker="D",
            lw=1.7,
            label="W0 RF median (5 seeds)",
        )
        main_ax.fill_between(
            forest.index,
            forest["pred_min"],
            forest["pred_max"],
            color=COLORS["W0_random_forest"],
            alpha=0.15,
            linewidth=0,
            label="W0 RF seed range",
        )

        aa_ax.plot(
            observed.loc[aa_w2.index].index,
            observed.loc[aa_w2.index],
            color=COLORS["observed"],
            marker="o",
            lw=2.4,
            label="Observed target-cycle peak",
            zorder=5,
        )
        aa_ax.plot(
            aa_w1.index,
            aa_w1["pred_median"],
            color=COLORS["W1_AA_CONTROL"],
            marker="v",
            lw=1.8,
            label="W1 RF: same AA coverage",
        )
        aa_ax.fill_between(
            aa_w1.index,
            aa_w1["pred_min"],
            aa_w1["pred_max"],
            color=COLORS["W1_AA_CONTROL"],
            alpha=0.12,
            linewidth=0,
        )
        aa_ax.plot(
            aa_w2.index,
            aa_w2["pred_median"],
            color=COLORS["W2_AA"],
            marker="P",
            lw=1.8,
            ls="-.",
            label="W2 RF: W1 + aa",
        )
        aa_ax.fill_between(
            aa_w2.index,
            aa_w2["pred_min"],
            aa_w2["pred_max"],
            color=COLORS["W2_AA"],
            alpha=0.10,
            linewidth=0,
        )

        for x_value, y_value, offset in zip(
            observed.index,
            observed,
            truth_offsets[f"{target_kind}_main"],
        ):
            main_ax.annotate(
                f"true {y_value:.1f}",
                (x_value, y_value),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=7.5,
                color=COLORS["observed"],
                bbox={"facecolor": "white", "edgecolor": "#999999", "pad": 2},
                zorder=7,
            )
        for x_value, y_value, offset in zip(
            observed.loc[aa_w2.index].index,
            observed.loc[aa_w2.index],
            truth_offsets[f"{target_kind}_aa"],
        ):
            aa_ax.annotate(
                f"true {y_value:.1f}",
                (x_value, y_value),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=7.5,
                color=COLORS["observed"],
                bbox={"facecolor": "white", "edgecolor": "#999999", "pad": 2},
                zorder=7,
            )

        main_ticks = observed.index.astype(int).tolist()
        aa_ticks = aa_w2.index.astype(int).tolist()
        main_ax.set_xticks(
            main_ticks,
            [f"{cycle}→SC{cycle + 1}" for cycle in main_ticks],
        )
        aa_ax.set_xticks(
            aa_ticks,
            [f"{cycle}→SC{cycle + 1}" for cycle in aa_ticks],
        )
        main_ax.set_ylabel("Peak SSN")
        main_ax.set_title(
            f"{TARGET_TITLES[target_kind]}\nA. W0 signal versus climate baseline",
            loc="left",
            fontsize=10.5,
        )
        aa_ax.set_title(
            f"{TARGET_TITLES[target_kind]}\nB. Equal-coverage AA increment test",
            loc="left",
            fontsize=10.5,
        )
        for ax in (main_ax, aa_ax):
            ax.grid(color="#E2E2E2", lw=0.7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.margins(y=0.18)
        main_ax.legend(loc="upper right", fontsize=7.3, frameon=True, framealpha=0.95)
        aa_ax.legend(loc="upper left", fontsize=7.3, frameon=True, framealpha=0.95)

        w0_rf = detail_expanding.loc[
            (detail_expanding["target_kind"] == target_kind)
            & (detail_expanding["feature_set"] == "W0")
            & (detail_expanding["model"] == "random_forest")
        ]
        w0_mae = float(w0_rf["mae"].mean())
        w0_ridge = detail_expanding.loc[
            (detail_expanding["target_kind"] == target_kind)
            & (detail_expanding["feature_set"] == "W0")
            & (detail_expanding["model"] == "ridge")
        ]
        w1_aa = detail_expanding.loc[
            (detail_expanding["target_kind"] == target_kind)
            & (detail_expanding["feature_set"] == "W1_AA_CONTROL")
            & (detail_expanding["model"] == "random_forest")
        ]
        w2_aa = detail_expanding.loc[
            (detail_expanding["target_kind"] == target_kind)
            & (detail_expanding["feature_set"] == "W2_AA")
            & (detail_expanding["model"] == "random_forest")
        ]
        aa_delta = float(w2_aa["mae"].mean() - w1_aa["mae"].mean())
        main_ax.text(
            0.03,
            0.04,
            (
                f"W0 Ridge MAE = {w0_ridge['mae'].mean():.2f}\n"
                f"W0 RF MAE = {w0_mae:.2f}"
            ),
            transform=main_ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "pad": 3},
        )
        aa_ax.text(
            0.03,
            0.04,
            (
                f"W1 RF MAE = {w1_aa['mae'].mean():.2f}\n"
                f"W2 RF MAE = {w2_aa['mae'].mean():.2f}\n"
                f"Delta MAE (W2-W1) = {aa_delta:+.2f}"
            ),
            transform=aa_ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "pad": 3},
        )

    fig.suptitle(
        "R1B multicyle validation | observed true values are printed at every point",
        x=0.01,
        y=0.995,
        ha="left",
        fontsize=12,
    )
    fig.text(
        0.01,
        0.005,
        (
            "x label = forecast context → target solar cycle. Lines connect cycle-level peak predictions, not monthly SSN. "
            "RF bands show five-seed min–max. AA panels use rows 20–23 because earlier complete-aa features are unavailable."
        ),
        fontsize=7.5,
        color="#444444",
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.08)
    for suffix in (".png", ".pdf"):
        fig.savefig(
            OUT / f"r1b_multicycle_validation_curves{suffix}",
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
