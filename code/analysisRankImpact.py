#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single-proposal counterfactual selection analysis
with pooling across models and prompt templates.

Main analysis
-------------
- Reads three prompt-template CSVs defined in PROMPT_INPUTS
- Merges them with a prompt_template column
- Pools across models and prompt templates for the main outputs
- Produces per-model and per-template versions by restricting the input data
  to the relevant subset and recomputing the full analysis on that subset

Main outputs
------------
- exit_rate_vs_k.png
- mean_rank_loss_by_base_rank.png
- selection_summary_two_panel.png
- rank_delta_boxplot.png
- model_consistency_summary.csv
- model_consistency_mean_rank_delta.png
- prompt_consistency_summary.csv
- prompt_consistency_mean_rank_delta.png
- coherence_two_panel.png

Plus:
- per_model/<model_name>/...
- per_prompt/<prompt_name>/...

Notes
-----
- Pooling across models is intended as an LLM-panel analogue.
- Pooling across prompt templates treats prompt wording as nuisance variability.
- rank_delta does not depend on K, but exit-based metrics do.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def save_latex_table(
    df: pd.DataFrame,
    path: Path,
    caption: str,
    label: str,
    float_cols: List[str] | None = None,
    font_size: str = r"\scriptsize",
    col_sep_pt: str = "4pt",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table_df = df.copy()

    if float_cols is None:
        float_cols = table_df.select_dtypes(include=[np.number]).columns.tolist()

    for col in float_cols:
        if col in table_df.columns:
            table_df[col] = pd.to_numeric(table_df[col], errors="coerce")

    latex = table_df.to_latex(
        index=False,
        escape=False,
        na_rep="",
        float_format=lambda x: f"{x:.3f}",
        caption=caption,
        label=label,
    )

    if font_size:
        latex = latex.replace(r"\begin{tabular}", f"{font_size}\n\\begin{{tabular}}", 1)

    if col_sep_pt:
        sep = f"\\setlength{{\\tabcolsep}}{{{col_sep_pt}}}"
        latex = latex.replace(r"\begin{tabular}", f"{sep}\\begin{{tabular}}", 1)

    path.write_text(latex, encoding="utf-8")
    print(f"Saved: {path}")


DEFAULT_CONSISTENCY_LATEX_COLUMN_LABELS = {
    "effect_label": "Effect",
    "pooled_mean_rank_delta": "Pooled",
    "rank_delta_sign_consistency": "S.C.",
    "rank_delta_sd_across_models": "SD",
    "rank_delta_sd_across_prompts": "SD",
    "rank_delta_min": "Min",
    "rank_delta_max": "Max",
    "rank_delta_mean_abs_dev_from_pooled": "MADp",
}


def prepare_consistency_table_for_latex(
    df: pd.DataFrame,
    column_label_map: Optional[Dict[str, str]] = None,
    drop_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    out = df.copy()

    if drop_cols is None:
        drop_cols = ["effect", "flip_label", "n_models"]

    existing_drop_cols = [col for col in drop_cols if col in out.columns]
    if existing_drop_cols:
        out = out.drop(columns=existing_drop_cols)

    if column_label_map:
        rename_map = {col: label for col, label in column_label_map.items() if col in out.columns}
        if rename_map:
            out = out.rename(columns=rename_map)

    return out


PROMPT_INPUTS: Dict[str, str] = {
    "prompt_A": "proposals/analysis_v1/all_scores_wide.csv",
    "prompt_B": "proposals/analysis_v2/all_scores_wide.csv",
    "prompt_C": "proposals/analysis_v3/all_scores_wide.csv",
}

PROMPT_COL = "prompt_template"

EFFECT_SPECS = [
    {
        "effect": "pi_name_group",
        "family": "pi_main",
        "level_a": "F_N",
        "level_b": "M_N",
        "other_factors": ["pi_inst_tier", "pi_metric_level"],
        "label": "Name: Female → Male",
    },
    {
        "effect": "pi_inst_tier",
        "family": "pi_main",
        "level_a": "TOP_I",
        "level_b": "LOW_I",
        "other_factors": ["pi_name_group", "pi_metric_level"],
        "label": "Institute: Top-tier → Low-tier",
    },
    {
        "effect": "pi_metric_level",
        "family": "pi_main",
        "level_a": "HIGH_B",
        "level_b": "LOW_B",
        "other_factors": ["pi_name_group", "pi_inst_tier"],
        "label": "Profile: Higher → Lower",
    },
    {
        "effect": "ai_flag",
        "family": "ai_main",
        "level_a": "AI_N",
        "level_b": "AI_Y",
        "other_factors": [],
        "label": "AI usage: No → Yes",
    },
]

EFFECT_ORDER = [spec["label"] for spec in EFFECT_SPECS]


CONSISTENCY_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*", "h", "8"]
EFFECT_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*", "h", "8"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-proposal counterfactual top-K penalty analysis pooled across models and prompts."
    )
    parser.add_argument(
        "--score_col",
        type=str,
        default="q06",
        help="Score column to analyze (default: q06)",
    )
    parser.add_argument(
        "--k_list",
        type=int,
        nargs="+",
        default=[4, 5, 6, 7, 8, 9, 10],
        help="Funding thresholds K to analyze",
    )
    parser.add_argument(
        "--aggregate",
        type=str,
        choices=["mean", "median"],
        default="mean",
        help="How to aggregate iterations within each proposal-condition-model-prompt cell",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="proposals/analysis_MERGED_selection",
        help="Output directory",
    )

    parser.add_argument(
        "--model",
        action="store_true",
        help="compute also per model",
    )

    parser.add_argument(
        "--prompt",
        action="store_true",
        help="compute also per prompt",
    )
    parser.add_argument(
        "--bootstrap_iters",
        type=int,
        default=1000,
        help="Number of proposal-level bootstrap resamples for selection confidence intervals",
    )
    parser.add_argument(
        "--bootstrap_seed",
        type=int,
        default=42,
        help="Random seed for bootstrap resampling",
    )
    return parser.parse_args()


def ensure_required_columns(df: pd.DataFrame, score_col: str) -> None:
    required = {
        "text_id",
        "model",
        PROMPT_COL,
        "condition_family",
        "condition_id",
        "iteration",
        score_col,
        "pi_name_group",
        "pi_inst_tier",
        "pi_metric_level",
        "ai_flag",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def load_prompt_data(score_col: str) -> pd.DataFrame:
    frames = []
    for prompt_label, path_str in PROMPT_INPUTS.items():
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found for '{prompt_label}': {path}")
        df = pd.read_csv(path)
        df[PROMPT_COL] = prompt_label
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    ensure_required_columns(merged, score_col)
    return merged


def aggregate_scores(df: pd.DataFrame, score_col: str, aggregate: str) -> pd.DataFrame:
    group_cols = [
        "text_id",
        "model",
        PROMPT_COL,
        "condition_family",
        "condition_id",
        "pi_name_group",
        "pi_inst_tier",
        "pi_metric_level",
        "ai_flag",
    ]

    tmp = df.copy()
    tmp[score_col] = pd.to_numeric(tmp[score_col], errors="coerce")

    if aggregate == "mean":
        agg_df = (
            tmp.groupby(group_cols, dropna=False, as_index=False)[score_col]
            .mean()
            .rename(columns={score_col: "score"})
        )
    else:
        agg_df = (
            tmp.groupby(group_cols, dropna=False, as_index=False)[score_col]
            .median()
            .rename(columns={score_col: "score"})
        )

    return agg_df


def build_context_key(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    if not cols:
        return pd.Series(["__NO_CONTEXT__"] * len(df), index=df.index)

    parts = [df[c].astype(str).fillna("NA") for c in cols]
    key = parts[0].copy()
    for s in parts[1:]:
        key = key + "|" + s
    return key


def deterministic_rank(scores: pd.Series, text_ids: pd.Series) -> Dict[str, int]:
    tmp = pd.DataFrame({"text_id": text_ids.values, "score": scores.values})
    tmp = tmp.sort_values(["score", "text_id"], ascending=[False, True]).reset_index(drop=True)
    tmp["rank"] = np.arange(1, len(tmp) + 1)
    return dict(zip(tmp["text_id"], tmp["rank"]))


def summarize_group(g: pd.DataFrame) -> pd.Series:
    out = {}
    out["n_records"] = len(g)
    out["n_models"] = g["model"].nunique() if "model" in g.columns else np.nan
    out["n_prompts"] = g[PROMPT_COL].nunique() if PROMPT_COL in g.columns else np.nan
    out["n_contexts"] = g["context_key"].nunique() if "context_key" in g.columns else np.nan
    out["n_proposals"] = g["text_id"].nunique() if "text_id" in g.columns else np.nan

    out["mean_score_delta"] = g["score_delta"].mean()
    out["median_score_delta"] = g["score_delta"].median()

    out["mean_rank_delta"] = g["rank_delta"].mean()
    out["median_rank_delta"] = g["rank_delta"].median()

    harmed = g["rank_delta"] > 0
    out["mean_rank_delta_if_harmed"] = g.loc[harmed, "rank_delta"].mean() if harmed.any() else 0.0
    out["p_any_rank_loss"] = harmed.mean()
    out["p_any_rank_gain"] = (g["rank_delta"] < 0).mean()

    out["p_exit_unconditional"] = g["exit_funding"].mean()

    funded_base = g["funded_base"] == 1
    out["p_exit_given_funded_base"] = (
        g.loc[funded_base, "exit_funding"].mean() if funded_base.any() else np.nan
    )

    return pd.Series(out)


def compute_single_flip_records(
    agg_df: pd.DataFrame,
    effect_spec: Dict,
    k_list: List[int],
) -> pd.DataFrame:
    family = effect_spec["family"]
    effect = effect_spec["effect"]
    level_a = effect_spec["level_a"]
    level_b = effect_spec["level_b"]
    other_factors = effect_spec["other_factors"]
    label = effect_spec["label"]

    sub = agg_df.loc[agg_df["condition_family"] == family].copy()
    if sub.empty:
        return pd.DataFrame()

    sub = sub.loc[sub[effect].notna()].copy()
    if sub.empty:
        return pd.DataFrame()

    sub["context_key"] = build_context_key(sub, other_factors)

    pivot_index = ["model", PROMPT_COL, "context_key", "text_id"]
    piv = (
        sub.pivot_table(
            index=pivot_index,
            columns=effect,
            values="score",
            aggfunc="first",
        )
        .reset_index()
    )

    if level_a not in piv.columns or level_b not in piv.columns:
        return pd.DataFrame()

    piv = piv.loc[piv[level_a].notna() & piv[level_b].notna()].copy()
    if piv.empty:
        return pd.DataFrame()

    records = []

    for (model, prompt_name, context_key), block in piv.groupby(
        ["model", PROMPT_COL, "context_key"], dropna=False
    ):
        block = block[["text_id", level_a, level_b]].copy().reset_index(drop=True)
        n_props = len(block)
        if n_props == 0:
            continue

        base_scores = block[level_a].copy()
        text_ids = block["text_id"].copy()
        base_rank_map = deterministic_rank(base_scores, text_ids)
        idx_map = {tid: idx for idx, tid in enumerate(block["text_id"].tolist())}

        for k in k_list:
            for row in block.itertuples(index=False):
                tid = row.text_id
                score_a = getattr(row, level_a)
                score_b = getattr(row, level_b)

                cf_scores = base_scores.copy()
                cf_scores.iloc[idx_map[tid]] = score_b
                cf_rank_map = deterministic_rank(cf_scores, text_ids)

                rank_base = base_rank_map[tid]
                rank_flip = cf_rank_map[tid]
                funded_base = int(rank_base <= k)
                funded_flip = int(rank_flip <= k)

                records.append(
                    {
                        "effect": effect,
                        "effect_label": label,
                        "family": family,
                        "level_a": level_a,
                        "level_b": level_b,
                        "flip_label": f"{level_a} -> {level_b}",
                        "model": model,
                        PROMPT_COL: prompt_name,
                        "context_key": context_key,
                        "text_id": tid,
                        "K": k,
                        "score_base": score_a,
                        "score_flip": score_b,
                        "score_delta": score_b - score_a,
                        "rank_base": rank_base,
                        "rank_flip": rank_flip,
                        "rank_delta": rank_flip - rank_base,
                        "funded_base": funded_base,
                        "funded_flip": funded_flip,
                        "exit_funding": int(funded_base == 1 and funded_flip == 0),
                        "enter_funding": int(funded_base == 0 and funded_flip == 1),
                        "n_proposals_in_context": n_props,
                    }
                )

    return pd.DataFrame.from_records(records)


def bootstrap_selection_curves(
    proposal_level: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    text_ids = sorted(proposal_level["text_id"].dropna().astype(str).unique().tolist())
    if not text_ids:
        return pd.DataFrame(), pd.DataFrame()

    rng = np.random.default_rng(seed)
    boot_effect_k_frames = []
    boot_rank_frames = []

    for b in range(n_boot):
        sampled_ids = rng.choice(text_ids, size=len(text_ids), replace=True)
        boot_parts = []
        for i, tid in enumerate(sampled_ids):
            part = proposal_level.loc[proposal_level["text_id"].astype(str) == tid].copy()
            if part.empty:
                continue
            part["bootstrap_text_id"] = f"{tid}__boot{i}"
            boot_parts.append(part)

        if not boot_parts:
            continue

        boot_df = pd.concat(boot_parts, ignore_index=True)

        effect_k = (
            boot_df.groupby(["effect", "effect_label", "flip_label", "K"], dropna=False)
            .apply(summarize_group, include_groups=False)
            .reset_index()
        )
        effect_k["bootstrap_iter"] = b
        boot_effect_k_frames.append(effect_k)

        rank_df = make_rank_plot_slice(boot_df.rename(columns={"bootstrap_text_id": "text_id_boot"}))
        if "text_id_boot" in rank_df.columns:
            rank_df = rank_df.rename(columns={"text_id": "orig_text_id", "text_id_boot": "text_id"})
        rank_prof = (
            rank_df.groupby(["effect", "effect_label", "flip_label", "rank_base"], dropna=False)["rank_delta"]
            .mean()
            .reset_index()
        )
        rank_prof["bootstrap_iter"] = b
        boot_rank_frames.append(rank_prof)

    if boot_effect_k_frames:
        boot_effect_k = pd.concat(boot_effect_k_frames, ignore_index=True)
        effect_k_ci = (
            boot_effect_k.groupby(["effect", "effect_label", "flip_label", "K"], dropna=False)
            .agg(
                p_exit_given_funded_base_ci_low=("p_exit_given_funded_base", lambda s: float(np.nanquantile(s, 0.025))),
                p_exit_given_funded_base_ci_high=("p_exit_given_funded_base", lambda s: float(np.nanquantile(s, 0.975))),
                p_exit_unconditional_ci_low=("p_exit_unconditional", lambda s: float(np.nanquantile(s, 0.025))),
                p_exit_unconditional_ci_high=("p_exit_unconditional", lambda s: float(np.nanquantile(s, 0.975))),
                mean_rank_delta_ci_low=("mean_rank_delta", lambda s: float(np.nanquantile(s, 0.025))),
                mean_rank_delta_ci_high=("mean_rank_delta", lambda s: float(np.nanquantile(s, 0.975))),
            )
            .reset_index()
        )
    else:
        boot_effect_k = pd.DataFrame()
        effect_k_ci = pd.DataFrame()

    if boot_rank_frames:
        boot_rank = pd.concat(boot_rank_frames, ignore_index=True)
        rank_ci = (
            boot_rank.groupby(["effect", "effect_label", "flip_label", "rank_base"], dropna=False)
            .agg(
                rank_delta_ci_low=("rank_delta", lambda s: float(np.nanquantile(s, 0.025))),
                rank_delta_ci_high=("rank_delta", lambda s: float(np.nanquantile(s, 0.975))),
            )
            .reset_index()
        )
    else:
        boot_rank = pd.DataFrame()
        rank_ci = pd.DataFrame()

    return effect_k_ci, rank_ci


def merge_bootstrap_cis(
    proposal_level: pd.DataFrame,
    summary_effect_k: pd.DataFrame,
    rank_plot_df: pd.DataFrame,
    bootstrap_iters: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    effect_k_ci, rank_ci = bootstrap_selection_curves(
        proposal_level=proposal_level,
        n_boot=bootstrap_iters,
        seed=bootstrap_seed,
    )
    if not effect_k_ci.empty:
        summary_effect_k = summary_effect_k.merge(
            effect_k_ci,
            on=["effect", "effect_label", "flip_label", "K"],
            how="left",
        )
    if not rank_ci.empty:
        rank_plot_df = rank_plot_df.merge(
            rank_ci,
            on=["effect", "effect_label", "flip_label", "rank_base"],
            how="left",
        )
    return summary_effect_k, rank_plot_df, effect_k_ci, rank_ci


def get_effect_marker_map(effect_labels: List[str]) -> Dict[str, str]:
    ordered = list(dict.fromkeys(effect_labels))
    return {
        eff: EFFECT_MARKERS[i % len(EFFECT_MARKERS)]
        for i, eff in enumerate(ordered)
    }


def apply_effect_order(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "effect_label" in out.columns:
        out["effect_label"] = pd.Categorical(
            out["effect_label"],
            categories=EFFECT_ORDER,
            ordered=True,
        )
        out = out.sort_values("effect_label")
    return out

def style_top_legend(
    ax: plt.Axes,
    n_items: int,
    include_title: Optional[str] = None,
    y_anchor: float = 1.02,
) -> None:
    if n_items <= 0:
        return

    ncol = max(1, n_items)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, y_anchor),
        ncol=ncol,
        frameon=False,
        title=include_title,
        handletextpad=0.5,
        columnspacing=1.2,
        borderaxespad=0.2,
    )


def save_line_plot_by_effect(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    outpath: Path,
) -> None:
    df = apply_effect_order(df)

    fig, ax = plt.subplots(figsize=(9, 5))

    effect_labels = [e for e in EFFECT_ORDER if e in df["effect_label"].astype(str).unique()]
    marker_map = get_effect_marker_map(EFFECT_ORDER)

    for effect_label, g in df.groupby("effect_label", dropna=False, sort=False):
        g = g.sort_values(x_col)
        ax.plot(
            g[x_col],
            g[y_col],
            marker=marker_map.get(str(effect_label), "o"),
            label=str(effect_label),
        )

    ax.set_title(title, pad=24)
    ax.set_xlabel(x_col)
    ax.set_ylabel(ylabel)
    style_top_legend(ax, n_items=len(effect_labels), y_anchor=1.01)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_boxplot(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    title: str,
    ylabel: str,
    outpath: Path,
    rotation: int = 25,
) -> None:
    order = (
        df.groupby(category_col)[value_col]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    data = [df.loc[df[category_col] == cat, value_col].values for cat in order]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, tick_labels=order, showfliers=False)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotation)
    ax.axhline(0, linewidth=1)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def save_rank_profile_plot(
    df: pd.DataFrame,
    y_col: str,
    title: str,
    ylabel: str,
    outpath: Path,
    ci_low_col: Optional[str] = "rank_delta_ci_low",
    ci_high_col: Optional[str] = "rank_delta_ci_high",
) -> None:
    df = apply_effect_order(df)

    fig, ax = plt.subplots(figsize=(9, 5))

    effect_labels = [e for e in EFFECT_ORDER if e in df["effect_label"].astype(str).unique()]
    marker_map = get_effect_marker_map(EFFECT_ORDER)

    for effect_label, g in df.groupby("effect_label", dropna=False, sort=False):
        agg_dict = {y_col: "mean"}
        if ci_low_col and ci_low_col in g.columns:
            agg_dict[ci_low_col] = "first"
        if ci_high_col and ci_high_col in g.columns:
            agg_dict[ci_high_col] = "first"
        gg = g.groupby("rank_base", as_index=False).agg(agg_dict).sort_values("rank_base")
        ax.plot(
            gg["rank_base"],
            gg[y_col],
            marker=marker_map.get(str(effect_label), "o"),
            label=str(effect_label),
        )
        if ci_low_col and ci_high_col and ci_low_col in gg.columns and ci_high_col in gg.columns:
            lo = pd.to_numeric(gg[ci_low_col], errors="coerce")
            hi = pd.to_numeric(gg[ci_high_col], errors="coerce")
            if lo.notna().any() and hi.notna().any():
                ax.fill_between(gg["rank_base"], lo, hi, alpha=0.15)

    ax.axhline(0.0, color="grey", linestyle="--", linewidth=1.0)
    ax.set_title(title, pad=24)
    ax.set_xlabel("Base rank")
    ax.set_ylabel(ylabel)
    style_top_legend(ax, n_items=len(effect_labels), y_anchor=1.01)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_exit_rate_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str,
    ylabel: str,
    ci_low_col: Optional[str] = "p_exit_given_funded_base_ci_low",
    ci_high_col: Optional[str] = "p_exit_given_funded_base_ci_high",
) -> None:
    df = apply_effect_order(df)

    effect_labels = [e for e in EFFECT_ORDER if e in df["effect_label"].astype(str).unique()]
    marker_map = get_effect_marker_map(EFFECT_ORDER)

    for effect_label, g in df.groupby("effect_label", dropna=False, sort=False):
        g = g.sort_values("K")
        ax.plot(
            g["K"],
            g["p_exit_given_funded_base"],
            marker=marker_map.get(str(effect_label), "o"),
            label=str(effect_label),
        )
        if ci_low_col and ci_high_col and ci_low_col in g.columns and ci_high_col in g.columns:
            lo = pd.to_numeric(g[ci_low_col], errors="coerce")
            hi = pd.to_numeric(g[ci_high_col], errors="coerce")
            if lo.notna().any() and hi.notna().any():
                ax.fill_between(g["K"], lo, hi, alpha=0.15)

    ax.set_title(title, pad=36)
    ax.set_xlabel("K")
    ax.set_ylabel(ylabel)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.20),
            ncol=2,
            frameon=False,
            handletextpad=0.5,
            columnspacing=1.2,
            borderaxespad=0.2,
        )


def plot_rank_profile_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    y_col: str,
    title: str,
    ylabel: str,
    show_ylabel: bool = True,
    ci_low_col: Optional[str] = "rank_delta_ci_low",
    ci_high_col: Optional[str] = "rank_delta_ci_high",
) -> None:
    df = apply_effect_order(df)

    effect_labels = [e for e in EFFECT_ORDER if e in df["effect_label"].astype(str).unique()]
    marker_map = get_effect_marker_map(EFFECT_ORDER)

    for effect_label, g in df.groupby("effect_label", dropna=False, sort=False):
        agg_dict = {y_col: "mean"}
        if ci_low_col and ci_low_col in g.columns:
            agg_dict[ci_low_col] = "first"
        if ci_high_col and ci_high_col in g.columns:
            agg_dict[ci_high_col] = "first"
        gg = g.groupby("rank_base", as_index=False).agg(agg_dict).sort_values("rank_base")
        ax.plot(
            gg["rank_base"],
            gg[y_col],
            marker=marker_map.get(str(effect_label), "o"),
            label=str(effect_label),
        )
        if ci_low_col and ci_high_col and ci_low_col in gg.columns and ci_high_col in gg.columns:
            lo = pd.to_numeric(gg[ci_low_col], errors="coerce")
            hi = pd.to_numeric(gg[ci_high_col], errors="coerce")
            if lo.notna().any() and hi.notna().any():
                ax.fill_between(gg["rank_base"], lo, hi, alpha=0.15)

    ax.axhline(0.0, color="grey", linestyle="--", linewidth=1.0)
    ax.set_title(title, pad=36)
    ax.set_xlabel("Base rank")
    if show_ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel("")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5,  1.20),
            ncol=2,
            frameon=False,
            handletextpad=0.5,
            columnspacing=1.2,
            borderaxespad=0.2,
        )


def save_selection_two_panel_plot(
    summary_effect_k: pd.DataFrame,
    rank_plot_df: pd.DataFrame,
    outpath: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax_left, ax_right = axes

    plot_exit_rate_panel(
        ax=ax_left,
        df=summary_effect_k,
        title="Exit probability vs funding threshold K",
        ylabel="P(exit funding | funded in base)",
    )

    plot_rank_profile_panel(
        ax=ax_right,
        df=rank_plot_df,
        y_col="rank_delta",
        title="Mean rank loss by original rank",
        ylabel="Mean rank change",
        show_ylabel=True,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_model_consistency_plot(
    df: pd.DataFrame,
    pooled_df: pd.DataFrame,
    unit_col: str,
    value_col: str,
    title: str,
    xlabel: str,
    outpath: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    plot_consistency_panel(
        ax=ax,
        df=df,
        pooled_df=pooled_df,
        unit_col=unit_col,
        value_col=value_col,
        title=title,
        xlabel=xlabel,
        show_ylabel=True,
        show_legend=True,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_consistency_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    pooled_df: pd.DataFrame,
    unit_col: str,
    value_col: str,
    title: str,
    xlabel: str,
    show_ylabel: bool = True,
    show_legend: bool = True,
) -> None:
    pooled_df = apply_effect_order(pooled_df)
    df = apply_effect_order(df)

    effects = [e for e in EFFECT_ORDER if e in pooled_df["effect_label"].astype(str).unique()]
    y_positions = np.arange(len(effects))[::-1]
    y_map = {eff: y for eff, y in zip(effects, y_positions)}

    # Thin horizontal separators between effect rows
    for y_top, y_bottom in zip(y_positions[:-1], y_positions[1:]):
        y_sep = (y_top + y_bottom) / 2.0
        ax.axhline(
            y=y_sep,
            color="0.85",
            linewidth=0.6,
            zorder=0,
        )

    units = sorted(df[unit_col].dropna().astype(str).unique().tolist())
    marker_map = {
        u: CONSISTENCY_MARKERS[i % len(CONSISTENCY_MARKERS)]
        for i, u in enumerate(units)
    }

    for unit_name, g_unit in df.groupby(unit_col, dropna=False):
        unit_name = str(unit_name)
        marker = marker_map[unit_name]
        xs = []
        ys = []
        for _, row in g_unit.iterrows():
            eff = str(row["effect_label"])
            if eff not in y_map:
                continue
            xs.append(float(row[value_col]))
            ys.append(float(y_map[eff]))
        if xs:
            ax.scatter(xs, ys, marker=marker, s=46, alpha=0.9, label=unit_name)

    for _, row in pooled_df.iterrows():
        eff = str(row["effect_label"])
        if eff not in y_map:
            continue
        x = float(row[value_col])
        y = float(y_map[eff])
        ax.vlines(x, y - 0.25, y + 0.25, color="black", linewidth=2.0, zorder=3)
        ax.scatter([x], [y], marker="s", s=18, color="black", zorder=4)

    ax.axvline(0.0, color="grey", linestyle="--", linewidth=1.0)
    ax.set_yticks(y_positions)
    if show_ylabel:
        ax.set_yticklabels(effects)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=32)

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        uniq_handles = []
        uniq_labels = []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l)
                uniq_handles.append(h)
                uniq_labels.append(l)

        if unit_col == "model":
            legend_ncol = max(1, int(np.ceil(len(uniq_labels) / 2)))
        else:
            legend_ncol = max(1, len(uniq_labels))

        ax.legend(
            uniq_handles,
            uniq_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.35),
            ncol=legend_ncol,
            frameon=False,
            handletextpad=0.5,
            columnspacing=1.2,
            borderaxespad=0.2,
        )


def save_coherence_two_panel_plot(
    prompt_df: pd.DataFrame,
    pooled_prompt_df: pd.DataFrame,
    model_df: pd.DataFrame,
    pooled_model_df: pd.DataFrame,
    outpath: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.0), sharey=False)
    ax_left, ax_right = axes

    plot_consistency_panel(
        ax=ax_left,
        df=prompt_df,
        pooled_df=pooled_prompt_df,
        unit_col=PROMPT_COL,
        value_col="prompt_mean_rank_delta",
        title="Prompt coherence for mean rank loss",
        xlabel="Mean rank change",
        show_ylabel=True,
        show_legend=True,
    )

    plot_consistency_panel(
        ax=ax_right,
        df=model_df,
        pooled_df=pooled_model_df,
        unit_col="model",
        value_col="model_mean_rank_delta",
        title="Model coherence for mean rank loss",
        xlabel="Mean rank change",
        show_ylabel=False,
        show_legend=True,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_rank_plot_slice(proposal_level: pd.DataFrame) -> pd.DataFrame:
    subset_cols = [
        "effect",
        "effect_label",
        "flip_label",
        "model",
        PROMPT_COL,
        "context_key",
        "text_id",
        "score_base",
        "score_flip",
        "score_delta",
        "rank_base",
        "rank_flip",
        "rank_delta",
    ]
    return proposal_level.drop_duplicates(subset=subset_cols).copy()


def compute_model_consistency_summary(
    rank_plot_df: pd.DataFrame,
) -> pd.DataFrame:
    pooled_rank = (
        rank_plot_df.groupby(["effect", "effect_label", "flip_label"], dropna=False)["rank_delta"]
        .mean()
        .reset_index(name="pooled_mean_rank_delta")
    )

    model_rank = (
        rank_plot_df.groupby(["effect", "effect_label", "flip_label", "model"], dropna=False)["rank_delta"]
        .mean()
        .reset_index(name="model_mean_rank_delta")
    )

    rows = []

    for keys, sub in model_rank.groupby(["effect", "effect_label", "flip_label"], dropna=False):
        effect, effect_label, flip_label = keys

        pooled_rank_val = pooled_rank.loc[
            (pooled_rank["effect"] == effect) &
            (pooled_rank["effect_label"] == effect_label) &
            (pooled_rank["flip_label"] == flip_label),
            "pooled_mean_rank_delta"
        ]
        pooled_rank_val = float(pooled_rank_val.iloc[0]) if not pooled_rank_val.empty else np.nan

        rank_vals = pd.to_numeric(sub["model_mean_rank_delta"], errors="coerce").dropna()
        same_sign_rank = (
            int((np.sign(rank_vals) == np.sign(pooled_rank_val)).sum())
            if np.isfinite(pooled_rank_val)
            else np.nan
        )

        rows.append({
            "effect": effect,
            "effect_label": effect_label,
            "flip_label": flip_label,
            "n_models": int(sub["model"].nunique()),
            "pooled_mean_rank_delta": pooled_rank_val,
            "rank_delta_sign_consistency": (
                f"{same_sign_rank}/{int(sub['model'].nunique())}"
                if pd.notna(same_sign_rank)
                else np.nan
            ),
            "rank_delta_sd_across_models": float(rank_vals.std(ddof=1)) if len(rank_vals) > 1 else np.nan,
            "rank_delta_min": float(rank_vals.min()) if len(rank_vals) else np.nan,
            "rank_delta_max": float(rank_vals.max()) if len(rank_vals) else np.nan,
            "rank_delta_mean_abs_dev_from_pooled": (
                float(np.mean(np.abs(rank_vals - pooled_rank_val))) if len(rank_vals) else np.nan
            ),
        })

    return pd.DataFrame(rows)


def compute_prompt_consistency_summary(
    rank_plot_df: pd.DataFrame,
) -> pd.DataFrame:
    pooled_rank = (
        rank_plot_df.groupby(["effect", "effect_label", "flip_label"], dropna=False)["rank_delta"]
        .mean()
        .reset_index(name="pooled_mean_rank_delta")
    )

    prompt_rank = (
        rank_plot_df.groupby(["effect", "effect_label", "flip_label", PROMPT_COL], dropna=False)["rank_delta"]
        .mean()
        .reset_index(name="prompt_mean_rank_delta")
    )

    rows = []

    for keys, sub in prompt_rank.groupby(["effect", "effect_label", "flip_label"], dropna=False):
        effect, effect_label, flip_label = keys

        pooled_rank_val = pooled_rank.loc[
            (pooled_rank["effect"] == effect) &
            (pooled_rank["effect_label"] == effect_label) &
            (pooled_rank["flip_label"] == flip_label),
            "pooled_mean_rank_delta"
        ]
        pooled_rank_val = float(pooled_rank_val.iloc[0]) if not pooled_rank_val.empty else np.nan

        rank_vals = pd.to_numeric(sub["prompt_mean_rank_delta"], errors="coerce").dropna()
        same_sign_rank = (
            int((np.sign(rank_vals) == np.sign(pooled_rank_val)).sum())
            if np.isfinite(pooled_rank_val)
            else np.nan
        )

        rows.append({
            "effect": effect,
            "effect_label": effect_label,
            "flip_label": flip_label,
            "n_prompts": int(sub[PROMPT_COL].nunique()),
            "pooled_mean_rank_delta": pooled_rank_val,
            "rank_delta_sign_consistency": (
                f"{same_sign_rank}/{int(sub[PROMPT_COL].nunique())}"
                if pd.notna(same_sign_rank)
                else np.nan
            ),
            "rank_delta_sd_across_prompts": float(rank_vals.std(ddof=1)) if len(rank_vals) > 1 else np.nan,
            "rank_delta_min": float(rank_vals.min()) if len(rank_vals) else np.nan,
            "rank_delta_max": float(rank_vals.max()) if len(rank_vals) else np.nan,
            "rank_delta_mean_abs_dev_from_pooled": (
                float(np.mean(np.abs(rank_vals - pooled_rank_val))) if len(rank_vals) else np.nan
            ),
        })

    return pd.DataFrame(rows)


def run_analysis_on_subset(
    df_subset: pd.DataFrame,
    score_col: str,
    aggregate: str,
    k_list: List[int],
    outdir: Path,
    make_consistency: bool = False,
    bootstrap_iters: int = 1000,
    bootstrap_seed: int = 42,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    ensure_required_columns(df_subset, score_col)

    agg_df = aggregate_scores(df_subset, score_col, aggregate)
    agg_df.to_csv(outdir / "aggregated_scores_per_condition.csv", index=False)

    all_records = []
    for spec in EFFECT_SPECS:
        rec = compute_single_flip_records(
            agg_df=agg_df,
            effect_spec=spec,
            k_list=k_list,
        )
        if not rec.empty:
            all_records.append(rec)

    if not all_records:
        raise RuntimeError(f"No counterfactual records were generated for subset: {outdir}")

    proposal_level = pd.concat(all_records, ignore_index=True)
    proposal_level.to_csv(outdir / "proposal_level_single_flip_records.csv", index=False)

    group_cols_1 = ["effect", "effect_label", "flip_label", "K"]
    summary_effect_k = (
        proposal_level
        .groupby(group_cols_1, dropna=False)
        .apply(summarize_group, include_groups=False)
        .reset_index()
    )
    summary_effect_k.to_csv(outdir / "summary_by_effect_and_k.csv", index=False)

    group_cols_2 = ["effect", "effect_label", "flip_label", "K", "model"]
    summary_effect_k_model = (
        proposal_level
        .groupby(group_cols_2, dropna=False)
        .apply(summarize_group, include_groups=False)
        .reset_index()
    )
    summary_effect_k_model.to_csv(outdir / "summary_by_effect_k_and_model.csv", index=False)

    group_cols_3 = ["effect", "effect_label", "flip_label", "K", PROMPT_COL]
    summary_effect_k_prompt = (
        proposal_level
        .groupby(group_cols_3, dropna=False)
        .apply(summarize_group, include_groups=False)
        .reset_index()
    )
    summary_effect_k_prompt.to_csv(outdir / "summary_by_effect_k_and_prompt.csv", index=False)

    rank_plot_df = make_rank_plot_slice(proposal_level)

    summary_effect_k, rank_plot_df, effect_k_ci, rank_ci = merge_bootstrap_cis(
        proposal_level=proposal_level,
        summary_effect_k=summary_effect_k,
        rank_plot_df=rank_plot_df,
        bootstrap_iters=bootstrap_iters,
        bootstrap_seed=bootstrap_seed,
    )
    summary_effect_k.to_csv(outdir / "summary_by_effect_and_k.csv", index=False)
    rank_plot_df.to_csv(outdir / "rank_plot_slice.csv", index=False)
    effect_k_ci.to_csv(outdir / "bootstrap_exit_rate_ci_by_effect_and_k.csv", index=False)
    rank_ci.to_csv(outdir / "bootstrap_rank_loss_ci_by_effect_and_rank.csv", index=False)

    save_line_plot_by_effect(
        df=summary_effect_k,
        x_col="K",
        y_col="p_exit_given_funded_base",
        title="Exit probability vs funding threshold K",
        ylabel="P(exit funding | funded in base)",
        outpath=outdir / "exit_rate_vs_k.png",
    )

    save_rank_profile_plot(
        df=rank_plot_df,
        y_col="rank_delta",
        title="Mean rank loss by original rank",
        ylabel="Mean rank change",
        outpath=outdir / "mean_rank_loss_by_base_rank.png",
    )

    save_selection_two_panel_plot(
        summary_effect_k=summary_effect_k,
        rank_plot_df=rank_plot_df,
        outpath=outdir / "selection_summary_two_panel.png",
    )

    save_boxplot(
        df=rank_plot_df,
        category_col="effect_label",
        value_col="rank_delta",
        title="Distribution of rank penalties after single-proposal flip",
        ylabel="Rank change (positive = positions lost)",
        outpath=outdir / "rank_delta_boxplot.png",
    )

    if make_consistency:
        model_consistency_df = compute_model_consistency_summary(
            rank_plot_df=rank_plot_df,
        )
        model_consistency_df.to_csv(outdir / "model_consistency_summary.csv", index=False)
        save_latex_table(
            df=prepare_consistency_table_for_latex(
                model_consistency_df,
                column_label_map=DEFAULT_CONSISTENCY_LATEX_COLUMN_LABELS,
                drop_cols=["effect", "flip_label", "n_models"],
            ),
            path=outdir / "model_consistency_summary.tex",
            caption="Model-level heterogeneity summary for mean rank-change effects in the counterfactual selection analysis. Each row reports the pooled mean rank effect for one counterfactual manipulation together with its dispersion across the eight evaluator models. \\emph{S.C.} (sign consistency) indicates how many model-specific estimates share the sign of the pooled estimate; \\emph{SD} is the standard deviation across model-specific estimates; \\emph{Min} and \\emph{Max} report the range of model-level mean rank effects; and \\emph{MADp} is the mean absolute deviation from the pooled estimate.",
            label="tab:model_consistency_summary",
            col_sep_pt="2.5pt",
            
        )

        model_rank_effect_df = (
            rank_plot_df.groupby(["effect", "effect_label", "flip_label", "model"], dropna=False)["rank_delta"]
            .mean()
            .reset_index(name="model_mean_rank_delta")
        )
        pooled_rank_effect_df = (
            rank_plot_df.groupby(["effect", "effect_label", "flip_label"], dropna=False)["rank_delta"]
            .mean()
            .reset_index(name="model_mean_rank_delta")
        )

        save_model_consistency_plot(
            df=model_rank_effect_df,
            pooled_df=pooled_rank_effect_df,
            unit_col="model",
            value_col="model_mean_rank_delta",
            title="Model consistency for mean rank loss",
            xlabel="Mean rank change",
            outpath=outdir / "model_consistency_mean_rank_delta.png",
        )

        prompt_consistency_df = compute_prompt_consistency_summary(
            rank_plot_df=rank_plot_df,
        )
        prompt_consistency_df.to_csv(outdir / "prompt_consistency_summary.csv", index=False)
        save_latex_table(
            df=prepare_consistency_table_for_latex(
                prompt_consistency_df,
                column_label_map=DEFAULT_CONSISTENCY_LATEX_COLUMN_LABELS,
                drop_cols=["effect", "flip_label", "n_prompts"],
            ),
            path=outdir / "prompt_consistency_summary.tex",
            caption="Prompt-level heterogeneity summary for mean rank-change effects in the counterfactual selection analysis. Each row reports the pooled mean rank effect for one counterfactual manipulation together with its dispersion across the three prompt templates. \\emph{S.C.} (sign consistency) indicates how many prompt-specific estimates share the sign of the pooled estimate; \\emph{SD} is the standard deviation across prompt-specific estimates; \\emph{Min} and \\emph{Max} report the range of prompt-level mean rank effects; and \\emph{MADp} is the mean absolute deviation from the pooled estimate.",
            label="tab:prompt_consistency_summary",
            col_sep_pt="2.5pt",
        )

        prompt_rank_effect_df = (
            rank_plot_df.groupby(["effect", "effect_label", "flip_label", PROMPT_COL], dropna=False)["rank_delta"]
            .mean()
            .reset_index(name="prompt_mean_rank_delta")
        )
        pooled_prompt_rank_effect_df = (
            rank_plot_df.groupby(["effect", "effect_label", "flip_label"], dropna=False)["rank_delta"]
            .mean()
            .reset_index(name="prompt_mean_rank_delta")
        )

        save_model_consistency_plot(
            df=prompt_rank_effect_df,
            pooled_df=pooled_prompt_rank_effect_df,
            unit_col=PROMPT_COL,
            value_col="prompt_mean_rank_delta",
            title="Prompt consistency for mean rank loss",
            xlabel="Mean rank change",
            outpath=outdir / "prompt_consistency_mean_rank_delta.png",
        )

        save_coherence_two_panel_plot(
            prompt_df=prompt_rank_effect_df,
            pooled_prompt_df=pooled_prompt_rank_effect_df,
            model_df=model_rank_effect_df,
            pooled_model_df=pooled_rank_effect_df,
            outpath=outdir / "coherence_two_panel.png",
        )

    with open(outdir / "README_results.txt", "w", encoding="utf-8") as f:
        f.write(
            "Generated files\n"
            "---------------\n"
            "aggregated_scores_per_condition.csv\n"
            "proposal_level_single_flip_records.csv\n"
            "summary_by_effect_and_k.csv\n"
            "summary_by_effect_k_and_model.csv\n"
            "summary_by_effect_k_and_prompt.csv\n"
            "rank_plot_slice.csv\n\n"
            "Plots\n"
            "-----\n"
            "exit_rate_vs_k.png\n"
            "mean_rank_loss_by_base_rank.png\n"
            "selection_summary_two_panel.png\n"
            "rank_delta_boxplot.png\n"
            "model_consistency_mean_rank_delta.png\n"
            "model_consistency_summary.tex\n"
            "prompt_consistency_mean_rank_delta.png\n"
            "prompt_consistency_summary.tex\n"
            "coherence_two_panel.png\n\n"
            "If this is the pooled main analysis, it pools across both models and prompt templates.\n"
        )

    print(f"Saved analysis to: {outdir}")


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_prompt_data(args.score_col)
    df.to_csv(outdir / "merged_input_with_prompts.csv", index=False)
    print(f"Loaded merged data: {len(df):,} rows")

    run_analysis_on_subset(
        df_subset=df,
        score_col=args.score_col,
        aggregate=args.aggregate,
        k_list=args.k_list,
        outdir=outdir,
        make_consistency=True,
        bootstrap_iters=args.bootstrap_iters,
        bootstrap_seed=args.bootstrap_seed,
    )

    if args.model:
        per_model_dir = outdir / "per_model"
        per_model_dir.mkdir(parents=True, exist_ok=True)
        for model_name in sorted(df["model"].dropna().astype(str).unique()):
            safe_model = model_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
            model_dir = per_model_dir / safe_model
            df_model = df.loc[df["model"].astype(str) == model_name].copy()
            run_analysis_on_subset(
                df_subset=df_model,
                score_col=args.score_col,
                aggregate=args.aggregate,
                k_list=args.k_list,
                outdir=model_dir,
                make_consistency=False,
                bootstrap_iters=args.bootstrap_iters,
                bootstrap_seed=args.bootstrap_seed,
            )

    if args.prompt:
        per_prompt_dir = outdir / "per_prompt"
        per_prompt_dir.mkdir(parents=True, exist_ok=True)
        for prompt_name in sorted(df[PROMPT_COL].dropna().astype(str).unique()):
            safe_prompt = prompt_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
            prompt_dir = per_prompt_dir / safe_prompt
            df_prompt = df.loc[df[PROMPT_COL].astype(str) == prompt_name].copy()
            run_analysis_on_subset(
                df_subset=df_prompt,
                score_col=args.score_col,
                aggregate=args.aggregate,
                k_list=args.k_list,
                outdir=prompt_dir,
                make_consistency=False,
                bootstrap_iters=args.bootstrap_iters,
                bootstrap_seed=args.bootstrap_seed,
            )

    print("\nDone.")
    print(f"Main pooled outputs: {outdir.resolve()}")
    print(f"Per-model outputs: {(outdir / 'per_model').resolve()}")
    print(f"Per-prompt outputs: {(outdir / 'per_prompt').resolve()}")


if __name__ == "__main__":
    main()