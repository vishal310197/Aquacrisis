
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# AquaCrisis Figure 2: Native vs translated macro-F1 + 95% CI
#
# Inputs:
#   1) bootstrap_individual_macro_f1_ci.csv
#      -> use BASELINE rows only
#   2) llm_macro_f1_bootstrap_ci.csv
#      -> exact Table-3 LLM run, 10,000 stratified bootstrap
#
# Outputs:
#   figure2_task_a.pdf / .png
#   figure2_task_b.pdf / .png
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

BASELINE_CI = BASE_DIR / "bootstrap_individual_macro_f1_ci.csv"
LLM_CI = BASE_DIR / "llm_macro_f1_bootstrap_ci.csv"

OUT_DIR = BASE_DIR / "figure2_outputs"
OUT_DIR.mkdir(exist_ok=True)

# PDF text should remain vector/text where possible.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def load_figure_data():
    # --------------------------------------------------------
    # 1. Baseline confidence intervals
    # IMPORTANT:
    # This older file also contains LLM rows from a different run.
    # We explicitly keep ONLY the baseline systems needed for Fig. 2.
    # --------------------------------------------------------
    base = pd.read_csv(BASELINE_CI)

    base = base[
        (base["system_type"] == "baseline")
        & (base["model_key"].isin([
            "tfidf_logreg",
            "multilingual_embeddings",
        ]))
        & (base["text"].isin(["native", "translated"]))
        & (base["task_display"].isin(["Task A", "Task B"]))
    ].copy()

    baseline_name_map = {
        "tfidf_logreg": "TF-IDF + LR",
        "multilingual_embeddings": "mEmb + LR",
    }

    base["plot_system"] = base["model_key"].map(baseline_name_map)
    base["task_plot"] = base["task_display"]
    base["estimate"] = base["macro_f1"]
    base["ci_low"] = base["ci95_low"]
    base["ci_high"] = base["ci95_high"]

    base = base[
        [
            "plot_system", "text", "task_plot",
            "estimate", "ci_low", "ci_high"
        ]
    ]

    # --------------------------------------------------------
    # 2. Exact LLM confidence intervals
    # This file was produced from the exact row-level predictions
    # behind Table 3 using 10,000 stratified-bootstrap replicates.
    # --------------------------------------------------------
    llm = pd.read_csv(LLM_CI)

    llm = llm[
        llm["task"].isin(["Task A", "Task B"])
        & llm["text"].isin(["native", "translated"])
    ].copy()

    shot_suffix = {
        "zero": "0s",
        "five": "5s",
    }

    llm["plot_system"] = (
        llm["model"]
        + "-"
        + llm["shot"].map(shot_suffix)
    )

    llm["task_plot"] = llm["task"]
    llm["estimate"] = llm["macro_f1"]

    llm = llm[
        [
            "plot_system", "text", "task_plot",
            "estimate", "ci_low", "ci_high"
        ]
    ]

    dat = pd.concat([base, llm], ignore_index=True)

    # --------------------------------------------------------
    # 3. Structural checks
    # --------------------------------------------------------
    expected_systems = [
        "TF-IDF + LR",
        "mEmb + LR",
        "Gemma 4-0s",
        "Gemma 4-5s",
        "GPT-4.1-mini-0s",
        "GPT-4.1-mini-5s",
        "GPT-4o-mini-0s",
        "GPT-4o-mini-5s",
        "Qwen 3.5-0s",
        "Qwen 3.5-5s",
    ]

    for task in ["Task A", "Task B"]:
        d = dat[dat["task_plot"] == task]
        for system in expected_systems:
            for text in ["native", "translated"]:
                n = len(
                    d[
                        (d["plot_system"] == system)
                        & (d["text"] == text)
                    ]
                )
                assert n == 1, (
                    f"Expected exactly one row for "
                    f"{task} / {system} / {text}; found {n}"
                )

    return dat, expected_systems


def plot_task(dat, task, systems, filename_stem, show_legend):
    d = dat[dat["task_plot"] == task].copy()

    # Bottom-to-top order matches the original Figure 2.
    y = np.arange(len(systems))
    bar_h = 0.34
    offset = 0.18

    native = (
        d[d["text"] == "native"]
        .set_index("plot_system")
        .loc[systems]
    )
    translated = (
        d[d["text"] == "translated"]
        .set_index("plot_system")
        .loc[systems]
    )

    fig, ax = plt.subplots(figsize=(3.55, 4.15))

    # Asymmetric x-errors: estimate-low and high-estimate.
    native_xerr = np.vstack([
        native["estimate"].to_numpy() - native["ci_low"].to_numpy(),
        native["ci_high"].to_numpy() - native["estimate"].to_numpy(),
    ])

    translated_xerr = np.vstack([
        translated["estimate"].to_numpy() - translated["ci_low"].to_numpy(),
        translated["ci_high"].to_numpy() - translated["estimate"].to_numpy(),
    ])

    # Do not set explicit colors: matplotlib's default palette is used.
    ax.barh(
        y - offset,
        native["estimate"],
        height=bar_h,
        xerr=native_xerr,
        capsize=2.2,
        label="Native",
    )

    ax.barh(
        y + offset,
        translated["estimate"],
        height=bar_h,
        xerr=translated_xerr,
        capsize=2.2,
        label="Translated",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(systems, fontsize=7.2)
    ax.set_xlabel("Macro-F1", fontsize=8.5)

    panel = "(a)" if task == "Task A" else "(b)"
    ax.set_title(
        f"{panel} {task}",
        fontsize=9,
        fontweight="bold",
        pad=20 if show_legend else 8,
    )

    # Keep scales close to the current paper figure.
    if task == "Task A":
        ax.set_xlim(0.0, 0.75)
        tick_step = 0.1
    else:
        ax.set_xlim(0.0, 0.60)
        tick_step = 0.1

    xmax = ax.get_xlim()[1]
    ax.set_xticks(np.arange(0, xmax + 1e-9, tick_step))
    ax.tick_params(axis="x", labelsize=7.2)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    # Point-estimate labels are placed just beyond the upper CI endpoint.
    for yi, (_, row) in zip(y - offset, native.iterrows()):
        x = min(row["ci_high"] + 0.006, xmax - 0.038)
        ax.text(
            x,
            yi,
            f'{row["estimate"]:.3f}',
            va="center",
            ha="left",
            fontsize=6.2,
        )

    for yi, (_, row) in zip(y + offset, translated.iterrows()):
        x = min(row["ci_high"] + 0.006, xmax - 0.038)
        ax.text(
            x,
            yi,
            f'{row["estimate"]:.3f}',
            va="center",
            ha="left",
            fontsize=6.2,
        )

    if show_legend:
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.005),
            ncol=2,
            fontsize=7.2,
            frameon=False,
            borderaxespad=0.0,
        )

    fig.tight_layout()

    pdf_path = OUT_DIR / f"{filename_stem}.pdf"
    png_path = OUT_DIR / f"{filename_stem}.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", pdf_path)
    print("Saved:", png_path)


if __name__ == "__main__":
    if not BASELINE_CI.exists():
        raise FileNotFoundError(BASELINE_CI)

    if not LLM_CI.exists():
        raise FileNotFoundError(LLM_CI)

    dat, systems = load_figure_data()

    # Print the exact numbers used in the figure for auditability.
    audit = dat.sort_values(
        ["task_plot", "plot_system", "text"]
    ).reset_index(drop=True)

    audit.to_csv(
        OUT_DIR / "figure2_values_used.csv",
        index=False,
    )

    print(audit.to_string(index=False))
    print()

    plot_task(
        dat,
        task="Task A",
        systems=systems,
        filename_stem="figure2_task_a",
        show_legend=True,
    )

    plot_task(
        dat,
        task="Task B",
        systems=systems,
        filename_stem="figure2_task_b",
        show_legend=False,
    )
