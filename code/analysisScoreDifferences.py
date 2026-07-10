#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import statsmodels.formula.api as smf


def holm_adjust_pvalues(p_values: pd.Series | List[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(p_values, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)

    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return out

    finite_vals = arr[finite_mask]
    m = len(finite_vals)
    order = np.argsort(finite_vals)
    sorted_p = finite_vals[order]

    adjusted_sorted = np.empty(m, dtype=float)
    running_max = 0.0
    for i, p in enumerate(sorted_p):
        adj = (m - i) * p
        running_max = max(running_max, adj)
        adjusted_sorted[i] = min(running_max, 1.0)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    out[finite_mask] = adjusted
    return out


QUESTION_IDS = ["q01", "q02", "q03", "q04", "q05", "q06"]

# QUESTION_LABELS = {
#     "q01": "q01\nscientific\nchallenges",
#     "q02": "q02\nambition/\nbeyond SOTA",
#     "q03": "q03\nfeasibility",
#     "q04": "q04\nmethodology/\narrangements",
# #    "q05": "q05 resources / timescales / PI commitment",
#     "q05": "q05\nresources/\ntimescales/\nPI commitment",
#     "q06": "q06\noverall\nscore",
# }

QUESTION_LABELS = {
    "q01": "q01\nscientific\nchallenges",
    "q02": "q02\nambition/\nbeyond SOTA",
    "q03": "q03\nfeasibility",
    "q04": "q04\nmethodology/\narrangements",
#    "q05": "q05 resources / timescales / PI commitment",
    "q05": "q05\nresources/time/\nPI commitment",
    "q06": "q06\noverall score",
}

QUESTION_LABELS = {
    "q01": "q01",
    "q02": "q02",
    "q03": "q03",
    "q04": "q04",
    "q05": "q05",
    "q06": "q06",
}

QUESTION_LABELS_LONG = {
    "q01": "q01 scientific challenges",
    "q02": "q02 ambition/beyond SOTA",
    "q03": "q03 feasibility",
    "q04": "q04 methodology/arrangements",
    "q05": "q05 resources/timescales/PI commitment",
    "q06": "q06 overall score",
}

PROMPT_INPUTS: Dict[str, str] = {
    "prompt_A": "proposals/analysis_v1/all_scores_wide.csv",
    "prompt_B": "proposals/analysis_v2/all_scores_wide.csv",
    "prompt_C": "proposals/analysis_v3/all_scores_wide.csv",
}

PROMPT_COL = "prompt_template"

CONTRASTS: List[Dict[str, str]] = [
    {"label": "Male (name)", "group": "PI_NAME", "run_mode": "pi-only", "pi_name_group": "M_N"},
    {"label": "Female (name)", "group": "PI_NAME", "run_mode": "pi-only", "pi_name_group": "F_N"},
    {"label": "Low-tier (inst.)", "group": "PI_INST", "run_mode": "pi-only", "pi_inst_tier": "LOW_I"},
    {"label": "Top-tier (inst.)", "group": "PI_INST", "run_mode": "pi-only", "pi_inst_tier": "TOP_I"},
    {"label": "Lower (profile)", "group": "PI_BIB", "run_mode": "pi-only", "pi_metric_level": "LOW_B"},
    {"label": "Higher (profile)", "group": "PI_BIB", "run_mode": "pi-only", "pi_metric_level": "HIGH_B"},
    {"label": "Yes (AI)", "group": "AI", "run_mode": "ai-only", "ai_flag": "AI_Y"},
    {"label": "No (AI)", "group": "AI", "run_mode": "ai-only", "ai_flag": "AI_N"},
]

CONTRAST_ORDER = [
    "Male (name)",
    "Female (name)",
    "Low-tier (inst.)",
    "Top-tier (inst.)",
    "Lower (profile)",
    "Higher (profile)",
    "Yes (AI)",
    "No (AI)",
]

STYLE_MAP = {
    "Male (name)": {"color": "C0", "marker": "o", "filled": True},
    "Female (name)": {"color": "C0", "marker": "o", "filled": False},
    "Low-tier (inst.)": {"color": "C1", "marker": "s", "filled": True},
    "Top-tier (inst.)": {"color": "C1", "marker": "s", "filled": False},
    "Lower (profile)": {"color": "C2", "marker": "D", "filled": True},
    "Higher (profile)": {"color": "C2", "marker": "D", "filled": False},
    "Yes (AI)": {"color": "C3", "marker": "^", "filled": True},
    "No (AI)": {"color": "C3", "marker": "^", "filled": False},
}

PAIR_SPECS = [
    ("Male (name)", "Female (name)"),
    ("Low-tier (inst.)", "Top-tier (inst.)"),
    ("Lower (profile)", "Higher (profile)"),
    ("Yes (AI)", "No (AI)"),
]

PAIR_Y_OFFSETS = {
    "Male (name)": 0.24,
    "Female (name)": 0.24,
    "Low-tier (inst.)": 0.08,
    "Top-tier (inst.)": 0.08,
    "Lower (profile)": -0.08,
    "Higher (profile)": -0.08,
    "Yes (AI)": -0.24,
    "No (AI)": -0.24,
}

X_DODGE = {
    "Male (name)": -0.002,
    "Female (name)": 0.002,
    "Low-tier (inst.)": -0.002,
    "Top-tier (inst.)": 0.002,
    "Lower (profile)": -0.002,
    "Higher (profile)": 0.002,
    "Yes (AI)": -0.002,
    "No (AI)": 0.002,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute baseline-referenced question effects from multiple prompt-template files, "
            "produce pooled main plots, compact robustness summaries, and supplementary "
            "per-model/per-template plots using subset-recompute logic."
        )
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="proposals/analysis_MERGED",
        help="Directory for CSVs and plots",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute cached CSVs and plots even if already present",
    )
    parser.add_argument(
        "--xlim_min",
        type=float,
        default=-0.32,
        help="Minimum x-limit for pooled/main effect plots",
    )
    parser.add_argument(
        "--xlim_max",
        type=float,
        default=0.06,
        help="Maximum x-limit for pooled/main effect plots",
    )
    parser.add_argument(
        "--coherence_question",
        type=str,
        default="q06",
        choices=QUESTION_IDS,
        help="Question used for compact coherence plots (default: q06)",
    )
    return parser.parse_args()


def validate_input_columns(df: pd.DataFrame) -> None:
    required = {
        "text_id",
        "model",
        "run_mode",
        "condition_id",
        PROMPT_COL,
        *QUESTION_IDS,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def add_numeric_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for q in QUESTION_IDS:
        out[q] = pd.to_numeric(out[q], errors="coerce")
    return out


def summarize_series(s: pd.Series) -> Dict[str, float]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    n = int(len(s))
    mean = float(s.mean()) if n else np.nan
    sd = float(s.std(ddof=1)) if n > 1 else np.nan
    se = float(sd / np.sqrt(n)) if n > 1 else np.nan
    ci95 = float(1.96 * se) if n > 1 else np.nan
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci95": ci95,
    }


def apply_contrast_filter(df: pd.DataFrame, contrast: Dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    for key, value in contrast.items():
        if key in {"label", "group"}:
            continue
        if key not in out.columns:
            raise ValueError(f"Contrast filter uses missing column: {key}")
        out = out[out[key].astype(str) == str(value)]
    return out


def pair_label(left_label: str, right_label: str) -> str:
    return f"{left_label} - {right_label}"


def load_and_merge_prompt_files() -> pd.DataFrame:
    frames = []

    for prompt_label, path_str in PROMPT_INPUTS.items():
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found for '{prompt_label}': {path}")

        df = pd.read_csv(path)
        print(f"{prompt_label}: {df.shape[0]} rows, {df.shape[1]} cols from {path}")
        df[PROMPT_COL] = prompt_label
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    print(f"merged: {merged.shape[0]} rows, {merged.shape[1]} cols")
    print(merged[PROMPT_COL].value_counts(dropna=False).sort_index())

    return merged


def build_text_level_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    blind = df[df["run_mode"] == "blind"].copy()
    target = df[df["run_mode"] != "blind"].copy()

    if blind.empty:
        raise ValueError("No blind rows found (run_mode == 'blind').")
    if target.empty:
        raise ValueError("No non-blind rows found.")

    blind_group_cols = ["model", "text_id", PROMPT_COL]
    blind_text = (
        blind.groupby(blind_group_cols, dropna=False)[QUESTION_IDS]
        .mean()
        .reset_index()
    )

    target_group_cols = [
        "model",
        "text_id",
        "run_mode",
        "condition_id",
        PROMPT_COL,
    ]
    optional_design_cols = [
        "pi_name_group",
        "pi_inst_tier",
        "pi_metric_level",
        "ai_flag",
        "condition_family",
    ]
    target_group_cols += [c for c in optional_design_cols if c in target.columns]

    target_text = (
        target.groupby(target_group_cols, dropna=False)[QUESTION_IDS]
        .mean()
        .reset_index()
    )

    return blind_text, target_text


def collapse_text_level_effects(text_df: pd.DataFrame) -> pd.DataFrame:
    if text_df.empty:
        return pd.DataFrame()

    agg_dict = {}
    for q in QUESTION_IDS:
        if f"blind_{q}" in text_df.columns:
            agg_dict[f"blind_{q}"] = (f"blind_{q}", "mean")
        if q in text_df.columns:
            agg_dict[q] = (q, "mean")
        if f"effect_{q}" in text_df.columns:
            agg_dict[f"effect_{q}"] = (f"effect_{q}", "mean")

    keep_cols = ["contrast_label", "contrast_group", "model", "text_id", PROMPT_COL]

    collapsed = (
        text_df.groupby(keep_cols, dropna=False)
        .agg(**agg_dict)
        .reset_index()
    )
    return collapsed


def compute_effect_tables(
    blind_text: pd.DataFrame,
    target_text: pd.DataFrame,
    contrasts: List[Dict[str, str]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merge_keys = ["model", "text_id", PROMPT_COL]

    merged = target_text.merge(
        blind_text.rename(columns={q: f"blind_{q}" for q in QUESTION_IDS}),
        on=merge_keys,
        how="left",
        validate="many_to_one",
    )

    missing_blind = merged["blind_q01"].isna().sum()
    if missing_blind > 0:
        print(f"WARNING: {missing_blind} target text rows could not be matched to a blind baseline.")

    text_rows = []
    model_prompt_question_rows = []

    for contrast in contrasts:
        label = contrast["label"]
        group = contrast.get("group", "OTHER")

        sub = apply_contrast_filter(merged, contrast).copy()
        if sub.empty:
            print(f"WARNING: contrast '{label}' matched no rows.")
            continue

        for q in QUESTION_IDS:
            sub[f"effect_{q}"] = sub[q] - sub[f"blind_{q}"]

        sub.insert(0, "contrast_label", label)
        sub.insert(1, "contrast_group", group)
        text_rows.append(sub)

    text_df = pd.concat(text_rows, ignore_index=True) if text_rows else pd.DataFrame()
    collapsed_text_df = collapse_text_level_effects(text_df)

    if not collapsed_text_df.empty:
        group_cols = ["contrast_label", "contrast_group", "model", PROMPT_COL]

        for keys, sub_df in collapsed_text_df.groupby(group_cols, dropna=False):
            key_map = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))

            for q in QUESTION_IDS:
                sq = summarize_series(sub_df[f"effect_{q}"])
                model_prompt_question_rows.append({
                    "contrast_label": key_map["contrast_label"],
                    "contrast_group": key_map["contrast_group"],
                    "model": key_map["model"],
                    PROMPT_COL: key_map[PROMPT_COL],
                    "question_id": q,
                    "blind_mean": float(pd.to_numeric(sub_df[f"blind_{q}"], errors="coerce").mean()),
                    "target_mean": float(pd.to_numeric(sub_df[q], errors="coerce").mean()),
                    "mean_effect": sq["mean"],
                    "sd_effect": sq["sd"],
                    "se_effect": sq["se"],
                    "ci95": sq["ci95"],
                    "n_texts": sq["n"],
                })

    model_prompt_question_df = pd.DataFrame(model_prompt_question_rows)
    pooled_question_df = compute_pooled_question_effects(
        text_df=text_df,
        model_prompt_question_df=model_prompt_question_df,
    )

    return text_df, model_prompt_question_df, pooled_question_df


def compute_pooled_question_effects(
    text_df: pd.DataFrame,
    model_prompt_question_df: pd.DataFrame,
) -> pd.DataFrame:
    pooled_rows = []

    if model_prompt_question_df.empty or text_df.empty:
        return pd.DataFrame()

    collapsed_text_df = collapse_text_level_effects(text_df)

    for (contrast_label, contrast_group, question_id), sub_unit in model_prompt_question_df.groupby(
        ["contrast_label", "contrast_group", "question_id"], dropna=False
    ):
        sub_text = collapsed_text_df[
            collapsed_text_df["contrast_label"] == contrast_label
        ].copy()

        unit_effects = pd.to_numeric(sub_unit["mean_effect"], errors="coerce").dropna()
        text_effects = pd.to_numeric(sub_text[f"effect_{question_id}"], errors="coerce").dropna()

        blind_means = pd.to_numeric(sub_unit["blind_mean"], errors="coerce")
        target_means = pd.to_numeric(sub_unit["target_mean"], errors="coerce")

        n_model_prompt_cells = int(len(unit_effects))
        n_models = int(sub_unit["model"].nunique())
        n_prompts = int(sub_unit[PROMPT_COL].nunique())
        n_texts = int(len(text_effects))

        mean_effect = float(unit_effects.mean()) if n_model_prompt_cells else np.nan
        sd_text = float(text_effects.std(ddof=1)) if n_texts > 1 else np.nan
        se_text = float(sd_text / np.sqrt(n_texts)) if n_texts > 1 else np.nan
        ci95 = float(1.96 * se_text) if n_texts > 1 else np.nan

        pooled_rows.append({
            "contrast_label": contrast_label,
            "contrast_group": contrast_group,
            "question_id": question_id,
            "blind_mean": float(blind_means.mean()) if len(blind_means) else np.nan,
            "target_mean": float(target_means.mean()) if len(target_means) else np.nan,
            "mean_effect": mean_effect,
            "sd_text": sd_text,
            "se_text": se_text,
            "ci95": ci95,
            "n_models": n_models,
            "n_prompts": n_prompts,
            "n_model_prompt_cells": n_model_prompt_cells,
            "n_texts": n_texts,
        })

    return pd.DataFrame(pooled_rows)


def build_paired_difference_table(
    df: pd.DataFrame,
    value_col: str,
    unit_cols: List[str],
) -> pd.DataFrame:
    """
    Build paired contrast deltas:
      pair_value = value(left contrast) - value(right contrast)

    Expected input columns:
      unit_cols + ["contrast_label", "question_id", value_col]
    """
    rows = []
    if df.empty:
        return pd.DataFrame()

    needed = set(unit_cols + ["contrast_label", "question_id", value_col])
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for paired-difference table: {sorted(missing)}")

    for left_label, right_label in PAIR_SPECS:
        left = (
            df[df["contrast_label"] == left_label][unit_cols + ["question_id", value_col]]
            .rename(columns={value_col: "left_value"})
            .copy()
        )
        right = (
            df[df["contrast_label"] == right_label][unit_cols + ["question_id", value_col]]
            .rename(columns={value_col: "right_value"})
            .copy()
        )

        paired = left.merge(
            right,
            on=unit_cols + ["question_id"],
            how="inner",
            validate="one_to_one",
        ).dropna(subset=["left_value", "right_value"])

        if paired.empty:
            continue

        paired["pair_label"] = pair_label(left_label, right_label)
        paired["left_label"] = left_label
        paired["right_label"] = right_label
        paired["pair_mean_diff"] = (
            pd.to_numeric(paired["left_value"], errors="coerce")
            - pd.to_numeric(paired["right_value"], errors="coerce")
        )

        rows.append(paired)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    ordered_cols = (
        ["pair_label", "left_label", "right_label"]
        + unit_cols
        + ["question_id", "left_value", "right_value", "pair_mean_diff"]
    )
    return out[ordered_cols]


def p_to_stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def compute_pairwise_significance(text_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    if text_df.empty:
        return pd.DataFrame()

    collapsed_text_df = collapse_text_level_effects(text_df)
    merge_keys = ["model", "text_id", PROMPT_COL]

    for q in QUESTION_IDS:
        effect_col = f"effect_{q}"

        for left_label, right_label in PAIR_SPECS:
            left = (
                collapsed_text_df[collapsed_text_df["contrast_label"] == left_label]
                [merge_keys + [effect_col]]
                .rename(columns={effect_col: "left_effect"})
            )

            right = (
                collapsed_text_df[collapsed_text_df["contrast_label"] == right_label]
                [merge_keys + [effect_col]]
                .rename(columns={effect_col: "right_effect"})
            )

            paired = left.merge(
                right,
                on=merge_keys,
                how="inner",
                validate="one_to_one",
            ).dropna()

            n = len(paired)
            if n >= 2:
                stat = ttest_rel(
                    paired["left_effect"].to_numpy(dtype=float),
                    paired["right_effect"].to_numpy(dtype=float),
                    nan_policy="omit",
                )
                t_statistic = float(stat.statistic) if np.isfinite(stat.statistic) else np.nan
                p_value = float(stat.pvalue) if np.isfinite(stat.pvalue) else np.nan
            else:
                t_statistic = np.nan
                p_value = np.nan

            rows.append({
                "question_id": q,
                "left_label": left_label,
                "right_label": right_label,
                "pair_label": pair_label(left_label, right_label),
                "n_pairs": n,
                "t_statistic": t_statistic,
                "p_value": p_value,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["p_value_holm"] = holm_adjust_pvalues(out["p_value"].to_numpy(dtype=float))
    out["stars"] = out["p_value"].apply(p_to_stars)
    out["stars_holm"] = out["p_value_holm"].apply(p_to_stars)
    return out




def compute_pairwise_permutation_significance(
    text_df: pd.DataFrame,
    n_permutations: int = 10000,
    seed: int = 0,
) -> pd.DataFrame:
    rows = []

    if text_df.empty:
        return pd.DataFrame()

    collapsed_text_df = collapse_text_level_effects(text_df)
    merge_keys = ["model", "text_id", PROMPT_COL]
    rng = np.random.default_rng(seed)

    for q in QUESTION_IDS:
        effect_col = f"effect_{q}"

        for left_label, right_label in PAIR_SPECS:
            left = (
                collapsed_text_df[collapsed_text_df["contrast_label"] == left_label]
                [merge_keys + [effect_col]]
                .rename(columns={effect_col: "left_effect"})
            )
            right = (
                collapsed_text_df[collapsed_text_df["contrast_label"] == right_label]
                [merge_keys + [effect_col]]
                .rename(columns={effect_col: "right_effect"})
            )

            paired = left.merge(
                right,
                on=merge_keys,
                how="inner",
                validate="one_to_one",
            ).dropna()

            diffs = (
                pd.to_numeric(paired["left_effect"], errors="coerce")
                - pd.to_numeric(paired["right_effect"], errors="coerce")
            ).dropna().to_numpy(dtype=float)
            n = int(len(diffs))

            if n >= 2:
                observed_mean = float(diffs.mean())
                signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, n), replace=True)
                permuted_means = (signs * diffs).mean(axis=1)
                p_value = float((np.sum(np.abs(permuted_means) >= abs(observed_mean)) + 1) / (n_permutations + 1))
            else:
                observed_mean = np.nan
                p_value = np.nan

            rows.append({
                "question_id": q,
                "left_label": left_label,
                "right_label": right_label,
                "pair_label": pair_label(left_label, right_label),
                "n_pairs": n,
                "observed_mean_diff": observed_mean,
                "p_value_permutation": p_value,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["p_value_permutation_holm"] = holm_adjust_pvalues(out["p_value_permutation"].to_numpy(dtype=float))
    out["stars_permutation"] = out["p_value_permutation"].apply(p_to_stars)
    out["stars_permutation_holm"] = out["p_value_permutation_holm"].apply(p_to_stars)
    return out


def compute_mixed_effects_q06_main_effects(text_df: pd.DataFrame) -> pd.DataFrame:
    if text_df.empty:
        return pd.DataFrame()

    collapsed_text_df = collapse_text_level_effects(text_df)
    fit_df = collapsed_text_df[["contrast_label", "contrast_group", "model", "text_id", PROMPT_COL, "effect_q06"]].copy()
    fit_df["effect_q06"] = pd.to_numeric(fit_df["effect_q06"], errors="coerce")
    fit_df = fit_df.dropna(subset=["effect_q06", "contrast_label", "text_id", "model", PROMPT_COL])
    if fit_df.shape[0] < 2:
        return pd.DataFrame()

    fit_df["contrast_label"] = pd.Categorical(fit_df["contrast_label"], categories=CONTRAST_ORDER, ordered=True)
    fit_df = fit_df.dropna(subset=["contrast_label"]).copy()
    if fit_df.empty:
        return pd.DataFrame()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.mixedlm(
                "effect_q06 ~ 0 + C(contrast_label)",
                data=fit_df,
                groups=fit_df["text_id"],
                re_formula="1",
                vc_formula={
                    "model": "0 + C(model)",
                    "prompt": f"0 + C({PROMPT_COL})",
                },
            )
            result = model.fit(reml=False, method="lbfgs", disp=False)

        conf_int = result.conf_int()
        rows = []
        for contrast_label in CONTRAST_ORDER:
            term = f"C(contrast_label)[{contrast_label}]"
            estimate = float(result.params.get(term, np.nan))
            se = float(result.bse.get(term, np.nan))
            z_value = float(result.tvalues.get(term, np.nan))
            p_value = float(result.pvalues.get(term, np.nan))
            if term in conf_int.index:
                ci_low, ci_high = conf_int.loc[term].tolist()
                ci_low = float(ci_low)
                ci_high = float(ci_high)
            else:
                ci_low = np.nan
                ci_high = np.nan
            group_vals = fit_df.loc[fit_df["contrast_label"] == contrast_label, "contrast_group"].dropna().astype(str).unique()
            contrast_group = group_vals[0] if len(group_vals) else np.nan
            rows.append({
                "question_id": "q06",
                "contrast_label": contrast_label,
                "contrast_group": contrast_group,
                "n_obs": int((fit_df["contrast_label"] == contrast_label).sum()),
                "n_texts": int(fit_df.loc[fit_df["contrast_label"] == contrast_label, "text_id"].nunique()),
                "n_models": int(fit_df.loc[fit_df["contrast_label"] == contrast_label, "model"].nunique()),
                "n_prompts": int(fit_df.loc[fit_df["contrast_label"] == contrast_label, PROMPT_COL].nunique()),
                "estimate": estimate,
                "se": se,
                "z_value": z_value,
                "p_value": p_value,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "converged": bool(getattr(result, "converged", True)),
            })
    except Exception as exc:
        print(f"WARNING: mixed-effects main-effect fit failed for q06: {exc}")
        rows = []
        for contrast_label in CONTRAST_ORDER:
            sub = fit_df[fit_df["contrast_label"] == contrast_label].copy()
            vals = pd.to_numeric(sub["effect_q06"], errors="coerce").dropna()
            rows.append({
                "question_id": "q06",
                "contrast_label": contrast_label,
                "contrast_group": sub["contrast_group"].dropna().astype(str).iloc[0] if not sub.empty else np.nan,
                "n_obs": int(len(vals)),
                "n_texts": int(sub["text_id"].nunique()),
                "n_models": int(sub["model"].nunique()),
                "n_prompts": int(sub[PROMPT_COL].nunique()),
                "estimate": float(vals.mean()) if len(vals) else np.nan,
                "se": np.nan,
                "z_value": np.nan,
                "p_value": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "converged": False,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["p_value_holm"] = holm_adjust_pvalues(out["p_value"].to_numpy(dtype=float))
    out["stars"] = out["p_value"].apply(p_to_stars)
    out["stars_holm"] = out["p_value_holm"].apply(p_to_stars)
    return out


def compute_mixed_effects_q06(text_df: pd.DataFrame) -> pd.DataFrame:
    if text_df.empty:
        return pd.DataFrame()

    collapsed_text_df = collapse_text_level_effects(text_df)
    collapsed_text_df = collapsed_text_df.copy()
    collapsed_text_df["question_id"] = "q06"
    pair_df = build_paired_difference_table(
        df=collapsed_text_df,
        value_col="effect_q06",
        unit_cols=["model", "text_id", PROMPT_COL],
    )
    if pair_df.empty:
        return pd.DataFrame()

    rows = []
    for pair_name, sub in pair_df.groupby("pair_label", dropna=False):
        fit_df = sub.copy()
        fit_df["pair_mean_diff"] = pd.to_numeric(fit_df["pair_mean_diff"], errors="coerce")
        fit_df = fit_df.dropna(subset=["pair_mean_diff", "text_id", "model", PROMPT_COL])
        if fit_df.shape[0] < 2:
            continue

        left_label = str(fit_df["left_label"].iloc[0])
        right_label = str(fit_df["right_label"].iloc[0])

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = smf.mixedlm(
                    "pair_mean_diff ~ 1",
                    data=fit_df,
                    groups=fit_df["text_id"],
                    re_formula="1",
                    vc_formula={
                        "model": "0 + C(model)",
                        "prompt": f"0 + C({PROMPT_COL})",
                    },
                )
                result = model.fit(reml=False, method="lbfgs", disp=False)

            estimate = float(result.params.get("Intercept", np.nan))
            se = float(result.bse.get("Intercept", np.nan))
            z_value = float(result.tvalues.get("Intercept", np.nan))
            p_value = float(result.pvalues.get("Intercept", np.nan))
            ci_low, ci_high = result.conf_int().loc["Intercept"].tolist()
            ci_low = float(ci_low)
            ci_high = float(ci_high)
            converged = bool(getattr(result, "converged", True))
        except Exception as exc:
            estimate = float(fit_df["pair_mean_diff"].mean())
            se = np.nan
            z_value = np.nan
            p_value = np.nan
            ci_low = np.nan
            ci_high = np.nan
            converged = False
            print(f"WARNING: mixed-effects fit failed for {pair_name}: {exc}")

        rows.append({
            "question_id": "q06",
            "pair_label": pair_name,
            "left_label": left_label,
            "right_label": right_label,
            "n_obs": int(fit_df.shape[0]),
            "n_texts": int(fit_df["text_id"].nunique()),
            "n_models": int(fit_df["model"].nunique()),
            "n_prompts": int(fit_df[PROMPT_COL].nunique()),
            "estimate": estimate,
            "se": se,
            "z_value": z_value,
            "p_value": p_value,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "converged": converged,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["p_value_holm"] = holm_adjust_pvalues(out["p_value"].to_numpy(dtype=float))
    out["stars"] = out["p_value"].apply(p_to_stars)
    out["stars_holm"] = out["p_value_holm"].apply(p_to_stars)
    return out



def compute_within_cell_run_variability(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()

    df = add_numeric_scores(raw_df)

    group_cols = [
        "model",
        "text_id",
        PROMPT_COL,
        "run_mode",
        "condition_id",
    ]
    optional_cols = [
        "pi_name_group",
        "pi_inst_tier",
        "pi_metric_level",
        "ai_flag",
        "condition_family",
    ]
    group_cols += [c for c in optional_cols if c in df.columns]

    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for within-cell variability: {missing}")

    cell_rows = []
    for q in QUESTION_IDS:
        tmp = (
            df.groupby(group_cols, dropna=False)[q]
            .agg(n_runs="count", run_mean="mean", run_sd="std")
            .reset_index()
        )
        tmp["question_id"] = q
        cell_rows.append(tmp)

    cell_df = pd.concat(cell_rows, ignore_index=True)
    cell_df["run_sd"] = pd.to_numeric(cell_df["run_sd"], errors="coerce").fillna(0.0)

    summary = (
        cell_df.groupby("question_id", dropna=False)
        .agg(
            n_cells=("run_sd", "size"),
            mean_within_cell_sd=("run_sd", "mean"),
            median_within_cell_sd=("run_sd", "median"),
            p95_within_cell_sd=("run_sd", lambda s: float(np.percentile(s, 95))),
            max_within_cell_sd=("run_sd", "max"),
            pct_zero_sd=("run_sd", lambda s: float((s == 0).mean() * 100.0)),
            mean_n_runs=("n_runs", "mean"),
            min_n_runs=("n_runs", "min"),
            max_n_runs=("n_runs", "max"),
        )
        .reset_index()
    )
    return summary


def compute_question_group_effects(text_df: pd.DataFrame) -> pd.DataFrame:
    if text_df.empty:
        return pd.DataFrame()

    collapsed = collapse_text_level_effects(text_df)
    if collapsed.empty:
        return pd.DataFrame()

    group_specs = {
        "project_focused_q01_q04": ["q01", "q02", "q03", "q04"],
        "capacity_global_q05_q06": ["q05", "q06"],
    }

    rows = []
    for group_name, questions in group_specs.items():
        effect_cols = [f"effect_{q}" for q in questions]
        missing = [c for c in effect_cols if c not in collapsed.columns]
        if missing:
            raise ValueError(f"Missing effect columns for question grouping: {missing}")

        tmp = collapsed.copy()
        tmp["group_effect"] = tmp[effect_cols].mean(axis=1)

        for (contrast_label, contrast_group), sub in tmp.groupby(
            ["contrast_label", "contrast_group"], dropna=False
        ):
            vals = pd.to_numeric(sub["group_effect"], errors="coerce").dropna()
            rows.append({
                "question_group": group_name,
                "contrast_label": contrast_label,
                "contrast_group": contrast_group,
                "mean_effect": float(vals.mean()) if len(vals) else np.nan,
                "sd": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                "se": float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else np.nan,
                "ci95": float(1.96 * vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else np.nan,
                "n": int(len(vals)),
                "n_texts": int(sub["text_id"].nunique()),
                "n_models": int(sub["model"].nunique()),
                "n_prompts": int(sub[PROMPT_COL].nunique()),
            })

    return pd.DataFrame(rows)


def compute_q06_consistency(text_df: pd.DataFrame) -> pd.DataFrame:
    if text_df.empty:
        return pd.DataFrame()

    collapsed = collapse_text_level_effects(text_df).copy()
    if collapsed.empty:
        return pd.DataFrame()

    score_cols = ["q01", "q02", "q03", "q04", "q05"]
    effect_cols = [f"effect_{q}" for q in score_cols]
    required = score_cols + ["q06"] + effect_cols + ["effect_q06"]
    missing = [c for c in required if c not in collapsed.columns]
    if missing:
        raise ValueError(f"Missing columns for q06 consistency analysis: {missing}")

    collapsed["mean_q01_q05"] = collapsed[score_cols].mean(axis=1)
    collapsed["mean_effect_q01_q05"] = collapsed[effect_cols].mean(axis=1)

    rows = []
    grouping_specs = {
        "pooled": [],
        "by_model": ["model"],
        "by_prompt": [PROMPT_COL],
        "by_contrast": ["contrast_label"],
    }

    for level, group_cols in grouping_specs.items():
        grouped = collapsed.groupby(group_cols, dropna=False) if group_cols else [((), collapsed)]
        for keys, sub in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {"level": level}
            for col, val in zip(group_cols, keys):
                row[col] = val

            score_pair = sub[["q06", "mean_q01_q05"]].dropna()
            effect_pair = sub[["effect_q06", "mean_effect_q01_q05"]].dropna()

            row["n_score_cells"] = int(len(score_pair))
            row["corr_q06_mean_q01_q05"] = (
                float(score_pair["q06"].corr(score_pair["mean_q01_q05"]))
                if len(score_pair) >= 2 else np.nan
            )
            row["n_effect_cells"] = int(len(effect_pair))
            row["corr_effect_q06_mean_effect_q01_q05"] = (
                float(effect_pair["effect_q06"].corr(effect_pair["mean_effect_q01_q05"]))
                if len(effect_pair) >= 2 else np.nan
            )
            rows.append(row)

    return pd.DataFrame(rows)


def compute_pi_interactions_q06(text_df: pd.DataFrame) -> pd.DataFrame:
    if text_df.empty:
        return pd.DataFrame()

    needed = {
        "run_mode",
        "condition_id",
        "model",
        "text_id",
        PROMPT_COL,
        "pi_name_group",
        "pi_inst_tier",
        "pi_metric_level",
        "effect_q06",
    }
    missing = needed - set(text_df.columns)
    if missing:
        raise ValueError(f"Missing columns for PI interaction analysis: {sorted(missing)}")

    pi_df = text_df[text_df["run_mode"] == "pi-only"].copy()
    pi_df = pi_df.drop_duplicates(subset=["model", "text_id", PROMPT_COL, "condition_id"])
    pi_df["effect_q06"] = pd.to_numeric(pi_df["effect_q06"], errors="coerce")
    pi_df = pi_df.dropna(
        subset=[
            "effect_q06",
            "pi_name_group",
            "pi_inst_tier",
            "pi_metric_level",
            "model",
            "text_id",
            PROMPT_COL,
        ]
    )

    if pi_df.empty:
        return pd.DataFrame()

    interaction_specs = [
        ("name_x_profile", "C(pi_name_group) * C(pi_metric_level)"),
        ("name_x_institution", "C(pi_name_group) * C(pi_inst_tier)"),
        ("institution_x_profile", "C(pi_inst_tier) * C(pi_metric_level)"),
        (
            "full_two_way_model",
            "C(pi_name_group) * C(pi_metric_level) + "
            "C(pi_name_group) * C(pi_inst_tier) + "
            "C(pi_inst_tier) * C(pi_metric_level)",
        ),
    ]

    rows = []
    for model_label, formula_rhs in interaction_specs:
        formula = f"effect_q06 ~ {formula_rhs}"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = smf.mixedlm(
                    formula,
                    data=pi_df,
                    groups=pi_df["text_id"],
                    re_formula="1",
                    vc_formula={
                        "model": "0 + C(model)",
                        "prompt": f"0 + C({PROMPT_COL})",
                    },
                )
                result = model.fit(reml=False, method="lbfgs", disp=False)

            conf = result.conf_int()
            for term in result.params.index:
                rows.append({
                    "model_label": model_label,
                    "term": term,
                    "estimate": float(result.params.get(term, np.nan)),
                    "se": float(result.bse.get(term, np.nan)),
                    "z_value": float(result.tvalues.get(term, np.nan)),
                    "p_value": float(result.pvalues.get(term, np.nan)),
                    "ci_low": float(conf.loc[term, 0]) if term in conf.index else np.nan,
                    "ci_high": float(conf.loc[term, 1]) if term in conf.index else np.nan,
                    "n_obs": int(pi_df.shape[0]),
                    "n_texts": int(pi_df["text_id"].nunique()),
                    "n_models": int(pi_df["model"].nunique()),
                    "n_prompts": int(pi_df[PROMPT_COL].nunique()),
                    "converged": bool(getattr(result, "converged", True)),
                })
        except Exception as exc:
            print(f"WARNING: PI interaction model failed for {model_label}: {exc}")

    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_value_holm"] = holm_adjust_pvalues(out["p_value"].to_numpy(dtype=float))
    return out


def build_proposal_level_q06_pair_differences(text_df: pd.DataFrame) -> pd.DataFrame:
    if text_df.empty:
        return pd.DataFrame()

    collapsed = collapse_text_level_effects(text_df).copy()
    if collapsed.empty:
        return pd.DataFrame()

    proposal_level = (
        collapsed.groupby(["text_id", "contrast_label"], dropna=False)["effect_q06"]
        .mean()
        .reset_index()
    )
    proposal_level["question_id"] = "q06"

    pair_df = build_paired_difference_table(
        df=proposal_level,
        value_col="effect_q06",
        unit_cols=["text_id"],
    )
    return pair_df


def compute_proposal_level_q06_robustness(
    text_df: pd.DataFrame,
    n_bootstrap: int = 10000,
    seed: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_df = build_proposal_level_q06_pair_differences(text_df)
    if pair_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    rng = np.random.default_rng(seed)
    sign_rows = []
    loo_rows = []
    boot_rows = []

    for pair_name, sub in pair_df.groupby("pair_label", dropna=False):
        vals_df = sub[["text_id", "pair_mean_diff"]].dropna().copy()
        vals = pd.to_numeric(vals_df["pair_mean_diff"], errors="coerce").to_numpy(dtype=float)
        text_ids = vals_df["text_id"].to_numpy()
        n = len(vals)
        if n == 0:
            continue

        mean_diff = float(vals.mean())
        n_negative = int((vals < 0).sum())
        n_positive = int((vals > 0).sum())
        n_zero = int((vals == 0).sum())

        sign_rows.append({
            "pair_label": pair_name,
            "n_proposals": int(n),
            "mean_diff": mean_diff,
            "n_negative": n_negative,
            "n_positive": n_positive,
            "n_zero": n_zero,
            "pct_negative": float(n_negative / n * 100.0),
        })

        if n > 1:
            for i in range(n):
                keep = np.delete(vals, i)
                loo_rows.append({
                    "pair_label": pair_name,
                    "left_out_text_id": text_ids[i],
                    "loo_mean_diff": float(keep.mean()),
                })
        else:
            loo_rows.append({
                "pair_label": pair_name,
                "left_out_text_id": text_ids[0],
                "loo_mean_diff": np.nan,
            })

        boot_means = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            boot_means.append(float(vals[idx].mean()))
        boot_means = np.asarray(boot_means, dtype=float)

        boot_rows.append({
            "pair_label": pair_name,
            "n_proposals": int(n),
            "observed_mean_diff": mean_diff,
            "bootstrap_ci_low": float(np.percentile(boot_means, 2.5)),
            "bootstrap_ci_high": float(np.percentile(boot_means, 97.5)),
            "bootstrap_p05": float(np.percentile(boot_means, 5)),
            "bootstrap_p95": float(np.percentile(boot_means, 95)),
            "n_bootstrap": int(n_bootstrap),
        })

    return pd.DataFrame(sign_rows), pd.DataFrame(loo_rows), pd.DataFrame(boot_rows)


def export_additional_diagnostics(text_df: pd.DataFrame, output_dir: Path) -> Dict[str, pd.DataFrame]:
    outputs: Dict[str, pd.DataFrame] = {}

    merged_input_path = output_dir / "merged_input_with_prompts.csv"
    if merged_input_path.exists():
        raw_df = pd.read_csv(merged_input_path)
    else:
        raw_df = load_and_merge_prompt_files()
        validate_input_columns(raw_df)
        raw_df = add_numeric_scores(raw_df)

    within_cell_sd_df = compute_within_cell_run_variability(raw_df)
    path = output_dir / "within_cell_run_variability.csv"
    within_cell_sd_df.to_csv(path, index=False)
    print(f"Saved: {path}")
    outputs["within_cell_run_variability"] = within_cell_sd_df

    question_group_df = compute_question_group_effects(text_df)
    path = output_dir / "question_group_effects_q01q04_vs_q05q06.csv"
    question_group_df.to_csv(path, index=False)
    print(f"Saved: {path}")
    outputs["question_group_effects"] = question_group_df

    q06_consistency_df = compute_q06_consistency(text_df)
    path = output_dir / "q06_consistency_with_q01_q05.csv"
    q06_consistency_df.to_csv(path, index=False)
    print(f"Saved: {path}")
    outputs["q06_consistency"] = q06_consistency_df

    pi_interactions_df = compute_pi_interactions_q06(text_df)
    path = output_dir / "pi_interactions_q06.csv"
    pi_interactions_df.to_csv(path, index=False)
    print(f"Saved: {path}")
    outputs["pi_interactions_q06"] = pi_interactions_df

    proposal_sign_df, proposal_loo_df, proposal_boot_df = compute_proposal_level_q06_robustness(
        text_df=text_df,
        n_bootstrap=10000,
        seed=0,
    )

    path = output_dir / "proposal_level_q06_sign_consistency.csv"
    proposal_sign_df.to_csv(path, index=False)
    print(f"Saved: {path}")
    outputs["proposal_level_q06_sign_consistency"] = proposal_sign_df

    path = output_dir / "proposal_level_q06_leave_one_out.csv"
    proposal_loo_df.to_csv(path, index=False)
    print(f"Saved: {path}")
    outputs["proposal_level_q06_leave_one_out"] = proposal_loo_df

    path = output_dir / "proposal_level_q06_bootstrap.csv"
    proposal_boot_df.to_csv(path, index=False)
    print(f"Saved: {path}")
    outputs["proposal_level_q06_bootstrap"] = proposal_boot_df

    return outputs

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
    path.write_text(latex)
    print(f"Saved: {path}")


def _sort_effect_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "contrast_label" not in df.columns:
        return df
    out = df.copy()
    out["contrast_label"] = pd.Categorical(
        out["contrast_label"], categories=CONTRAST_ORDER, ordered=True
    )
    sort_cols = [c for c in ["question_id", "contrast_label"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def _sort_pair_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "pair_label" not in df.columns:
        return df
    pair_order = [pair_label(left, right) for left, right in PAIR_SPECS]
    out = df.copy()
    out["pair_label"] = pd.Categorical(
        out["pair_label"], categories=pair_order, ordered=True
    )
    sort_cols = [c for c in ["question_id", "pair_label"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def format_pvalue_for_table(x: float) -> str:
    if pd.isna(x):
        return ""
    x = float(x)
    if x < 0.001:
        return "<0.001"
    return f"{x:.3f}"


def export_pairwise_significance_latex(sig_df: pd.DataFrame, output_dir: Path) -> None:
    if sig_df.empty:
        return

    pair_order = [pair_label(left, right) for left, right in PAIR_SPECS]
    pretty = sig_df.copy()
    pretty["question_id"] = pd.Categorical(pretty["question_id"], categories=QUESTION_IDS, ordered=True)
    pretty["pair_label"] = pd.Categorical(pretty["pair_label"], categories=pair_order, ordered=True)
    pretty = pretty.sort_values(["question_id", "pair_label"]).copy()

    pretty["Question"] = pretty["question_id"].map(QUESTION_LABELS)
    pretty["Contrast pair"] = pretty["pair_label"].astype(str)
    pretty["$N$"] = pretty["n_pairs"].map(lambda x: f"{int(x)}" if pd.notna(x) else "")
    pretty["$t$"] = pd.to_numeric(pretty["t_statistic"], errors="coerce").map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    #pretty["$p$"] = pd.to_numeric(pretty["p_value"], errors="coerce").map(format_pvalue_for_table)
    pretty["$p_{Hc}$"] = pd.to_numeric(pretty["p_value_holm"], errors="coerce").map(format_pvalue_for_table)
    #pretty["Sig."] = pretty["stars"].fillna("")
    #pretty["Sig. Holm"] = pretty["stars_holm"].fillna("")

    pretty = pretty[[
        #"Question", "Contrast pair", "$N$", "$t$", "$p$", "$p_{Holm}$", "Sig.", "Sig. Holm"
        "Question", "Contrast pair", "$N$", "$t$", "$p_{Hc}$"
    ]]

    save_latex_table(
        pretty,
        output_dir / "pairwise_significance.tex",
        caption=(
            "Pairwise paired $t$-test significance for left-minus-right differences between matched contextual contrasts, "
            "reported separately for each question."
        ),
        label="tab:pairwise_significance",
        float_cols=[],
    )



def export_pairwise_permutation_latex(perm_df: pd.DataFrame, output_dir: Path) -> None:
    if perm_df.empty:
        return

    pair_order = [pair_label(left, right) for left, right in PAIR_SPECS]
    pretty = perm_df.copy()
    pretty["question_id"] = pd.Categorical(pretty["question_id"], categories=QUESTION_IDS, ordered=True)
    pretty["pair_label"] = pd.Categorical(pretty["pair_label"], categories=pair_order, ordered=True)
    pretty = pretty.sort_values(["question_id", "pair_label"]).copy()

    pretty["Question"] = pretty["question_id"].map(QUESTION_LABELS)
    pretty["Contrast pair"] = pretty["pair_label"].astype(str)
    pretty["$N$"] = pretty["n_pairs"].map(lambda x: f"{int(x)}" if pd.notna(x) else "")
    pretty["Mean diff."] = pd.to_numeric(pretty["observed_mean_diff"], errors="coerce").map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    #pretty["$p_{perm}$"] = pd.to_numeric(pretty["p_value_permutation"], errors="coerce").map(format_pvalue_for_table)
    pretty["$p_{p-Hc}$"] = pd.to_numeric(pretty["p_value_permutation_holm"], errors="coerce").map(format_pvalue_for_table)
    #pretty["Sig."] = pretty["stars_permutation"].fillna("")
    #pretty["Sig. Holm"] = pretty["stars_permutation_holm"].fillna("")

    pretty = pretty[[
        #"Question", "Contrast pair", "$N$", "Mean diff.", "$p_{perm}$", "$p_{perm,Holm}$", "Sig.", "Sig. Holm"
        "Question", "Contrast pair", "$N$", "Mean diff.", "$p_{p-Hc}$"
    ]]

    save_latex_table(
        pretty,
        output_dir / "pairwise_significance_permutation.tex",
        caption=(
            "Pairwise permutation-test significance for left-minus-right differences between matched contextual contrasts, "
            "reported separately for each question."
        ),
        label="tab:pairwise_significance_permutation",
        float_cols=[],
    )


def export_mixed_effects_outputs(text_df: pd.DataFrame, output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    main_out = compute_mixed_effects_q06_main_effects(text_df)
    main_csv_path = output_dir / "mixed_effects_q06.csv"
    main_out.to_csv(main_csv_path, index=False)
    print(f"Saved: {main_csv_path}")

    if not main_out.empty:
        pretty = main_out[[
            "contrast_label", "estimate", "ci_low", "ci_high", "p_value_holm"
        ]].copy()

        pretty["Estimate"] = pretty["estimate"].map(lambda x: f"{x:.3f}")
        pretty["95\\% CI"] = pretty.apply(
            lambda r: f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]",
            axis=1,
        )
        pretty["$p_{Holm}$"] = pretty["p_value_holm"].map(
            lambda x: "<0.001" if pd.notna(x) and x < 0.001 else (f"{x:.3f}" if pd.notna(x) else "")
        )

        pretty = pretty[["contrast_label", "Estimate", "95\\% CI", "$p_{Holm}$"]].rename(columns={
            "contrast_label": "Contrast",
        })

        save_latex_table(
            pretty,
            output_dir / "mixed_effects_q06.tex",
            caption=(
                "Mixed-effects estimates for q06 blind-referenced contextual effects. "
                "Each coefficient estimates the average effect of a contextual condition "
                "relative to the blind baseline. We report estimate, 95\\% confidence interval, "
                "and Holm-adjusted $p$-value."
            ),
            label="tab:mixed_effects_q06",
        )

    pair_out = compute_mixed_effects_q06(text_df)
    pair_csv_path = output_dir / "mixed_effects_q06_pair_differences.csv"
    pair_out.to_csv(pair_csv_path, index=False)
    print(f"Saved: {pair_csv_path}")

    if not pair_out.empty:
        pretty = pair_out[[
            "pair_label", "estimate", "ci_low", "ci_high", "p_value_holm"
        ]].copy()

        pretty["Estimate"] = pretty["estimate"].map(lambda x: f"{x:.3f}")
        pretty["95\\% CI"] = pretty.apply(
            lambda r: f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]",
            axis=1,
        )
        pretty["$p_{Holm}$"] = pretty["p_value_holm"].map(
            lambda x: "<0.001" if pd.notna(x) and x < 0.001 else (f"{x:.3f}" if pd.notna(x) else "")
        )

        pretty = pretty[["pair_label", "Estimate", "95\\% CI", "$p_{Holm}$"]].rename(columns={
            "pair_label": "Contrast",
        })

        save_latex_table(
            pretty,
            output_dir / "mixed_effects_q06_pair_differences.tex",
            caption=(
                "Mixed-effects estimates for paired q06 effect differences. "
                "Each row reports the left-minus-right blind-referenced contrast difference, "
                "with estimate, 95\\% confidence interval, and Holm-adjusted $p$-value."
            ),
            label="tab:mixed_effects_q06_pair_differences",
        )

    return main_out, pair_out


def export_permutation_outputs(text_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    out = compute_pairwise_permutation_significance(text_df=text_df)
    csv_path = output_dir / "pairwise_significance_permutation.csv"
    out.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    export_pairwise_permutation_latex(perm_df=out, output_dir=output_dir)

    q06 = out[out["question_id"] == "q06"].copy()
    if not q06.empty:
        pretty = q06[[
            "pair_label", "observed_mean_diff", "p_value_permutation", "p_value_permutation_holm", "stars_permutation_holm", "n_pairs"
        ]].rename(columns={
            "pair_label": "Contrast",
            "observed_mean_diff": "Mean diff.",
            "p_value_permutation": "$p_{perm}$",
            "p_value_permutation_holm": "$p_{perm,Holm}$",
            "stars_permutation_holm": "Sig.",
            "n_pairs": "$N$ pairs",
        })
        save_latex_table(
            pretty,
            output_dir / "pairwise_significance_permutation_q06.tex",
            caption="Permutation robustness checks for paired q06 effect differences.",
            label="tab:perm_q06",
        )
    return out


def export_heterogeneity_tables(
    model_summary: pd.DataFrame,
    prompt_summary: pd.DataFrame,
    model_pair_summary: pd.DataFrame,
    prompt_pair_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    q06_model = _sort_effect_summary(
        model_summary[model_summary["question_id"] == "q06"].copy()
    )
    q06_prompt = _sort_effect_summary(
        prompt_summary[prompt_summary["question_id"] == "q06"].copy()
    )
    q06_model_pair = _sort_pair_summary(
        model_pair_summary[model_pair_summary["question_id"] == "q06"].copy()
    )
    q06_prompt_pair = _sort_pair_summary(
        prompt_pair_summary[prompt_pair_summary["question_id"] == "q06"].copy()
    )

    if not q06_model.empty:
        pretty = q06_model[[
            #"contrast_label", "pooled_effect", "n_models", "model_sign_consistency", "model_sd", "model_min", "model_max", "model_mean_abs_dev_from_pooled"
            "contrast_label", "pooled_effect", "model_sign_consistency", "model_sd", "model_min", "model_max", "model_mean_abs_dev_from_pooled"
        ]].rename(columns={
            "contrast_label": "Effect",
            "pooled_effect": "Pooled",
#            "n_models": "$N$ models",
            "model_sign_consistency": "S.C.",
            "model_sd": "SD",
            "model_min": "Min",
            "model_max": "Max",
            "model_mean_abs_dev_from_pooled": "MADp",
        })
        save_latex_table(
            pretty,
            output_dir / "model_heterogeneity_q06.tex",
            caption="Model-level heterogeneity summary for q06 blind-referenced effects. Each row reports the pooled q06 effect for one contextual condition together with its dispersion across the eight evaluator models. \\emph{S.C.} (sign consistency) indicates how many model-specific estimates share the sign of the pooled estimate; \\emph{SD} is the standard deviation across model-specific estimates; \\emph{Min} and \\emph{Max} report the range of model-level effects; and \\emph{MADp} is the mean absolute deviation from the pooled estimate.",
            label="tab:model_heterogeneity_q06",
            col_sep_pt="5pt",
        )

    if not q06_prompt.empty:
        pretty = q06_prompt[[
            #"contrast_label", "pooled_effect", "n_prompts", "prompt_sign_consistency", "prompt_sd", "prompt_min", "prompt_max", "prompt_mean_abs_dev_from_pooled"
            "contrast_label", "pooled_effect", "prompt_sign_consistency", "prompt_sd", "prompt_min", "prompt_max", "prompt_mean_abs_dev_from_pooled"
        ]].rename(columns={
            "contrast_label": "Effect",
            "pooled_effect": "Pooled",
#            "n_prompts": "$N$ prompts",
            "prompt_sign_consistency": "S.C.",
            "prompt_sd": "SD",
            "prompt_min": "Min",
            "prompt_max": "Max",
            "prompt_mean_abs_dev_from_pooled": "MADp",
        })
        save_latex_table(
            pretty,
            output_dir / "prompt_heterogeneity_q06.tex",
            caption="Prompt-level heterogeneity summary for q06 blind-referenced effects. Each row reports the pooled q06 effect for one contextual condition together with its dispersion across the three prompt templates. \\emph{S.C.} (sign consistency) indicates how many prompt-specific estimates share the sign of the pooled estimate; \\emph{SD} is the standard deviation across prompt-specific estimates; \\emph{Min} and \\emph{Max} report the range of prompt-level effects; and \\emph{MADp} is the mean absolute deviation from the pooled estimate.",
            label="tab:prompt_heterogeneity_q06",
            col_sep_pt="5pt",
        )

    if not q06_model_pair.empty:
        pretty = q06_model_pair[[
            #"pair_label", "pooled_pair_diff", "n_models", "model_sign_consistency", "model_sd", "model_min", "model_max", "model_mean_abs_dev_from_pooled"
            "pair_label", "pooled_pair_diff", "model_sign_consistency", "model_sd", "model_min", "model_max", "model_mean_abs_dev_from_pooled"
        ]].rename(columns={
            "pair_label": "Contrast pair",
            "pooled_pair_diff": "Pooled",
#            "n_models": "$N$ models",
            "model_sign_consistency": "S.C.",
            "model_sd": "SD",
            "model_min": "Min",
            "model_max": "Max",
            "model_mean_abs_dev_from_pooled": "MADp",
        })
        save_latex_table(
            pretty,
            output_dir / "model_pair_heterogeneity_q06.tex",
            caption="Model-level heterogeneity summary for paired q06 contrast differences. Each row reports the pooled left-minus-right q06 difference for one paired contrast together with its dispersion across the eight evaluator models. \\emph{S.C.} (sign consistency) indicates how many model-specific paired differences share the sign of the pooled difference; \\emph{SD} is the standard deviation across model-specific paired differences; \\emph{Min} and \\emph{Max} report the range of model-level paired differences; and \\emph{MADp} is the mean absolute deviation from the pooled difference.",
            label="tab:model_pair_heterogeneity_q06",
            col_sep_pt="1.8pt",
        )

    if not q06_prompt_pair.empty:
        pretty = q06_prompt_pair[[
            #"pair_label", "pooled_pair_diff", "n_prompts", "prompt_sign_consistency", "prompt_sd", "prompt_min", "prompt_max", "prompt_mean_abs_dev_from_pooled"
            "pair_label", "pooled_pair_diff", "prompt_sign_consistency", "prompt_sd", "prompt_min", "prompt_max", "prompt_mean_abs_dev_from_pooled"
        ]].rename(columns={
            "pair_label": "Contrast pair",
            "pooled_pair_diff": "Pooled",
#            "n_prompts": "$N$ prompts",
            "prompt_sign_consistency": "S.C.",
            "prompt_sd": "SD",
            "prompt_min": "Min",
            "prompt_max": "Max",
            "prompt_mean_abs_dev_from_pooled": "MADp",
        })
        save_latex_table(
            pretty,
            output_dir / "prompt_pair_heterogeneity_q06.tex",
            caption="Prompt-level heterogeneity summary for paired q06 contrast differences. Each row reports the pooled left-minus-right q06 difference for one paired contrast together with its dispersion across the three prompt templates. \\emph{S.C.} (sign consistency) indicates how many prompt-specific paired differences share the sign of the pooled difference; \\emph{SD} is the standard deviation across prompt-specific paired differences; \\emph{Min} and \\emph{Max} report the range of prompt-level paired differences; and \\emph{MADp} is the mean absolute deviation from the pooled difference.",
            label="tab:prompt_pair_heterogeneity_q06",
            col_sep_pt="1.8pt",
        )

def load_or_compute_effects(
    output_dir: Path,
    force: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged_input_path = output_dir / "merged_input_with_prompts.csv"
    text_path = output_dir / "text_level_effects.csv"
    model_prompt_question_path = output_dir / "model_prompt_question_effects.csv"
    pooled_question_path = output_dir / "pooled_question_effects.csv"

    if (not force) and text_path.exists() and model_prompt_question_path.exists() and pooled_question_path.exists():
        print("Using cached effect CSVs.")
        return (
            pd.read_csv(text_path),
            pd.read_csv(model_prompt_question_path),
            pd.read_csv(pooled_question_path),
        )

    print("Computing effect CSVs from prompt input files.")
    df = load_and_merge_prompt_files()
    validate_input_columns(df)
    df = add_numeric_scores(df)

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(merged_input_path, index=False)
    print(f"Saved: {merged_input_path}")

    blind_text, target_text = build_text_level_tables(df)
    text_df, model_prompt_question_df, pooled_question_df = compute_effect_tables(
        blind_text=blind_text,
        target_text=target_text,
        contrasts=CONTRASTS,
    )

    text_df.to_csv(text_path, index=False)
    model_prompt_question_df.to_csv(model_prompt_question_path, index=False)
    pooled_question_df.to_csv(pooled_question_path, index=False)

    print(f"Saved: {text_path}")
    print(f"Saved: {model_prompt_question_path}")
    print(f"Saved: {pooled_question_path}")

    return text_df, model_prompt_question_df, pooled_question_df


def plot_baseline_referenced_question_effects(
    df: pd.DataFrame,
    output_path: Path,
    xlim_min: float,
    xlim_max: float,
    title: str,
    sig_df: pd.DataFrame | None = None,
    x_label: str = "Score difference relative to blind baseline",
) -> None:
    if df.empty:
        raise ValueError("No data available for plotting.")

    q_order = ["q01", "q02", "q03", "q04", "q05", "q06"]
    plot_df = df.copy()
    plot_df = plot_df[plot_df["question_id"].isin(q_order)]
    plot_df = plot_df[plot_df["contrast_label"].isin(CONTRAST_ORDER)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    base_y = np.arange(len(q_order))[::-1]
    point_lookup: Dict[Tuple[str, str], Dict[str, float]] = {}

    # Thin horizontal separators between question rows
    for y_top, y_bottom in zip(base_y[:-1], base_y[1:]):
        y_sep = (y_top + y_bottom) / 2.0
        ax.axhline(
            y=y_sep,
            color="0.85",
            linewidth=0.6,
            zorder=0,
        )

    for contrast_label in CONTRAST_ORDER:
        sub = plot_df[plot_df["contrast_label"] == contrast_label].copy()
        if sub.empty:
            continue

        style = STYLE_MAP[contrast_label]
        xs: List[float] = []
        ys: List[float] = []
        xerrs: List[float] = []

        for q in q_order:
            row = sub[sub["question_id"] == q]
            if row.empty:
                continue

            mean_effect = pd.to_numeric(row["mean_effect"], errors="coerce").iloc[0]
            ci95 = pd.to_numeric(row["ci95"], errors="coerce").iloc[0]

            y = base_y[q_order.index(q)] + PAIR_Y_OFFSETS[contrast_label]
            x = float(mean_effect) + X_DODGE[contrast_label]

            xs.append(x)
            ys.append(float(y))
            xerrs.append(float(ci95) if np.isfinite(ci95) else 0.0)

            point_lookup[(q, contrast_label)] = {
                "x": x,
                "y": float(y),
                "ci95": float(ci95) if np.isfinite(ci95) else np.nan,
            }

        mfc = style["color"] if style["filled"] else "white"
        ci_linestyle = "-" if style["filled"] else "--"

        for x, y, err in zip(xs, ys, xerrs):
            if np.isfinite(err) and err > 0:
                ax.hlines(
                    y=y,
                    xmin=x - err,
                    xmax=x + err,
                    colors=style["color"],
                    linewidth=1.1,
                    linestyles=ci_linestyle,
                    zorder=2,
                )
                ax.vlines(
                    x=[x - err, x + err],
                    ymin=y - 0.035,
                    ymax=y + 0.035,
                    colors=style["color"],
                    linewidth=1.0,
                    zorder=2,
                )

        ax.plot(
            xs,
            ys,
            linestyle="none",
            marker=style["marker"],
            color=style["color"],
            markerfacecolor=mfc,
            markeredgecolor=style["color"],
            markeredgewidth=1.2,
            markersize=5,
            label=contrast_label,
            zorder=3,
        )

    if sig_df is not None and not sig_df.empty:
        for _, row in sig_df.iterrows():
            stars = row.get("stars_holm", row.get("stars", ""))
            if not stars:
                continue

            q = row["question_id"]
            left_label = row["left_label"]
            right_label = row["right_label"]

            left_point = point_lookup.get((q, left_label))
            right_point = point_lookup.get((q, right_label))
            if left_point is None or right_point is None:
                continue

            x_star = max(left_point["x"], right_point["x"]) + 0.012
            y_star = left_point["y"]

            ax.text(
                x_star,
                y_star,
                stars,
                va="center",
                ha="left",
                fontsize=11,
                color="black",
                zorder=4,
            )

    ax.axvline(0.0, color="C0", linestyle="--", linewidth=1.1, zorder=1)
    ax.set_xlim(xlim_min, xlim_max)
    ax.set_yticks(base_y)
    ax.set_yticklabels([QUESTION_LABELS[q] for q in q_order])
    ax.set_xlabel(x_label)
    # ax.set_title(title, pad=36)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.45, 1.15),
        ncol=4,
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.0,
        borderaxespad=0.0,
    )

    ax.grid(False)
    fig.subplots_adjust(top=0.80, left=0.22, right=0.98, bottom=0.11)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def compute_robustness_summary(
    unit_df: pd.DataFrame,
    pooled_df: pd.DataFrame,
    unit_col: str,
) -> pd.DataFrame:
    rows = []

    if unit_df.empty or pooled_df.empty:
        return pd.DataFrame()

    pooled_lookup = pooled_df.set_index(["contrast_label", "question_id"])["mean_effect"].to_dict()

    for (contrast_label, question_id), sub in unit_df.groupby(
        ["contrast_label", "question_id"], dropna=False
    ):
        vals = pd.to_numeric(sub["mean_effect"], errors="coerce").dropna()
        if vals.empty:
            continue

        pooled_effect = pooled_lookup.get((contrast_label, question_id), np.nan)
        signs = np.sign(vals)
        pooled_sign = np.sign(pooled_effect) if np.isfinite(pooled_effect) else np.nan

        n_units = int(sub[unit_col].nunique())
        if np.isfinite(pooled_sign) and pooled_sign != 0:
            same_sign = int((signs == pooled_sign).sum())
        else:
            same_sign = int((signs == 0).sum())

        rows.append({
            "contrast_label": contrast_label,
            "question_id": question_id,
            "pooled_effect": float(pooled_effect) if np.isfinite(pooled_effect) else np.nan,
            f"n_{unit_col}s": n_units,
            f"{unit_col}_sign_consistency": f"{same_sign}/{n_units}",
            f"{unit_col}_sd": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
            f"{unit_col}_min": float(vals.min()),
            f"{unit_col}_max": float(vals.max()),
            f"{unit_col}_mean_abs_dev_from_pooled": float(np.mean(np.abs(vals - pooled_effect))),
        })

    return pd.DataFrame(rows)


def compute_paired_difference_robustness_summary(
    unit_pair_df: pd.DataFrame,
    pooled_pair_df: pd.DataFrame,
    unit_col: str,
) -> pd.DataFrame:
    rows = []

    if unit_pair_df.empty or pooled_pair_df.empty:
        return pd.DataFrame()

    pooled_lookup = pooled_pair_df.set_index(["pair_label", "question_id"])["pair_mean_diff"].to_dict()

    for (pair_name, question_id), sub in unit_pair_df.groupby(
        ["pair_label", "question_id"], dropna=False
    ):
        vals = pd.to_numeric(sub["pair_mean_diff"], errors="coerce").dropna()
        if vals.empty:
            continue

        pooled_diff = pooled_lookup.get((pair_name, question_id), np.nan)
        signs = np.sign(vals)
        pooled_sign = np.sign(pooled_diff) if np.isfinite(pooled_diff) else np.nan

        n_units = int(sub[unit_col].nunique())
        if np.isfinite(pooled_sign) and pooled_sign != 0:
            same_sign = int((signs == pooled_sign).sum())
        else:
            same_sign = int((signs == 0).sum())

        left_label = str(sub["left_label"].iloc[0]) if "left_label" in sub.columns else ""
        right_label = str(sub["right_label"].iloc[0]) if "right_label" in sub.columns else ""

        rows.append({
            "pair_label": pair_name,
            "left_label": left_label,
            "right_label": right_label,
            "question_id": question_id,
            "pooled_pair_diff": float(pooled_diff) if np.isfinite(pooled_diff) else np.nan,
            f"n_{unit_col}s": n_units,
            f"{unit_col}_sign_consistency": f"{same_sign}/{n_units}",
            f"{unit_col}_sd": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
            f"{unit_col}_min": float(vals.min()),
            f"{unit_col}_max": float(vals.max()),
            f"{unit_col}_mean_abs_dev_from_pooled": float(np.mean(np.abs(vals - pooled_diff))),
        })

    return pd.DataFrame(rows)


def _get_style_for_contrast(contrast_label: str) -> Dict[str, object]:
    return STYLE_MAP.get(
        contrast_label,
        {"color": "black", "marker": "o", "filled": True},
    )


def _unit_marker_map(units: List[str]) -> Dict[str, str]:
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*", "h", "8"]
    out = {}
    for i, u in enumerate(sorted(map(str, units))):
        out[u] = marker_cycle[i % len(marker_cycle)]
    return out


def _plot_compact_coherence_panel(
    ax: plt.Axes,
    unit_df: pd.DataFrame,
    pooled_df: pd.DataFrame,
    unit_col: str,
    question_id: str,
    panel_title: str,
    show_ylabels: bool = True,
) -> None:
    sub_unit = unit_df[unit_df["question_id"] == question_id].copy()
    sub_pooled = pooled_df[pooled_df["question_id"] == question_id].copy()

    if sub_unit.empty or sub_pooled.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(panel_title)
        ax.set_axis_off()
        return

    sub_unit["contrast_label"] = pd.Categorical(
        sub_unit["contrast_label"], categories=CONTRAST_ORDER, ordered=True
    )
    sub_pooled["contrast_label"] = pd.Categorical(
        sub_pooled["contrast_label"], categories=CONTRAST_ORDER, ordered=True
    )

    sub_unit = sub_unit.sort_values(["contrast_label", unit_col])
    sub_pooled = sub_pooled.sort_values("contrast_label")

    y_positions = np.arange(len(CONTRAST_ORDER))[::-1]
    y_map = {label: y for label, y in zip(CONTRAST_ORDER, y_positions)}

    # Thin horizontal separators between contrast rows
    for y_top, y_bottom in zip(y_positions[:-1], y_positions[1:]):
        y_sep = (y_top + y_bottom) / 2.0
        ax.axhline(
            y=y_sep,
            color="0.85",
            linewidth=0.6,
            zorder=0,
        )

    units = sorted(sub_unit[unit_col].dropna().astype(str).unique().tolist())
    marker_map = _unit_marker_map(units)

    for unit_name, g_unit in sub_unit.groupby(unit_col, dropna=False):
        unit_name = str(unit_name)
        marker = marker_map[unit_name]

        for contrast_label, g in g_unit.groupby("contrast_label", dropna=False, observed=False):
            if pd.isna(contrast_label):
                continue
            contrast_label = str(contrast_label)
            if contrast_label not in y_map:
                continue

            y = y_map[contrast_label]
            style = _get_style_for_contrast(contrast_label)
            xs = pd.to_numeric(g["mean_effect"], errors="coerce").to_numpy(dtype=float)
            xs = xs[np.isfinite(xs)]
            if xs.size == 0:
                continue

            mfc = style["color"] if style["filled"] else "white"

            ax.scatter(
                xs,
                np.full_like(xs, y, dtype=float),
                s=46,
                marker=marker,
                facecolors=mfc,
                edgecolors=style["color"],
                linewidths=1.2,
                alpha=0.9,
                zorder=2,
                label=unit_name,
            )

    for _, row in sub_pooled.iterrows():
        label = str(row["contrast_label"])
        if label not in y_map:
            continue

        x = pd.to_numeric(pd.Series([row["mean_effect"]]), errors="coerce").iloc[0]
        if not np.isfinite(x):
            continue

        y = y_map[label]
        style = _get_style_for_contrast(label)

        #ax.vlines(x, y - 0.38, y + 0.38, linewidth=2.4, color=style["color"], zorder=3)
        ax.vlines(x, y - 0.38, y + 0.38, linewidth=2.4, color="black", zorder=3)
        ax.scatter(
            [x],
            [y],
            s=20,
            marker="s",
            #facecolors=style["color"],
            facecolors="black",
            #edgecolors=style["color"],
            edgecolors="black",
            linewidths=0.8,
            zorder=4,
        )

    ax.axvline(0.0, color="grey", linestyle="--", linewidth=1.0)
    ax.set_yticks(y_positions)

    ax.set_yticks(y_positions)
    if show_ylabels:
        ax.tick_params(axis="y", left=True, labelleft=True)
        ax.set_yticklabels(CONTRAST_ORDER)
    else:
        ax.tick_params(axis="y", left=False, labelleft=False)

    # ax.set_xlabel(f"Effect on {QUESTION_LABELS[question_id]}")
    ax.set_title(panel_title, pad=34)
    ax.grid(False)

    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_handles = []
    uniq_labels = []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            uniq_handles.append(h)
            uniq_labels.append(l)

    ax.legend(
        uniq_handles,
        uniq_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.23),
        ncol=min(4, max(1, len(uniq_labels))),
        frameon=False,
        borderaxespad=0.2,
    )


def plot_combined_coherence(
    prompt_level_df: pd.DataFrame,
    model_level_df: pd.DataFrame,
    pooled_df: pd.DataFrame,
    question_id: str,
    output_path: Path,
    title: str,
) -> None:
    prompt_df = prompt_level_df.rename(columns={PROMPT_COL: "prompt"}).copy()
    model_df = model_level_df.copy()

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 3.2),
        sharey=True,
    )

    _plot_compact_coherence_panel(
        ax=axes[0],
        unit_df=prompt_df,
        pooled_df=pooled_df,
        unit_col="prompt",
        question_id=question_id,
        panel_title=f"Prompt coherence for \"{QUESTION_LABELS_LONG[question_id]}\"",
        show_ylabels=True,
    )

    _plot_compact_coherence_panel(
        ax=axes[1],
        unit_df=model_df,
        pooled_df=pooled_df,
        unit_col="model",
        question_id=question_id,
        panel_title=f"Model coherence for \"{QUESTION_LABELS_LONG[question_id]}\"",
        show_ylabels=False,
    )

    #fig.suptitle(title, y=0.95)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def build_model_level_summary(model_prompt_question_df: pd.DataFrame) -> pd.DataFrame:
    if model_prompt_question_df.empty:
        return pd.DataFrame()

    group_cols = ["contrast_label", "contrast_group", "model", "question_id"]
    return (
        model_prompt_question_df.groupby(group_cols, dropna=False)
        .agg(
            mean_effect=("mean_effect", "mean"),
            blind_mean=("blind_mean", "mean"),
            target_mean=("target_mean", "mean"),
            n_texts=("n_texts", "mean"),
        )
        .reset_index()
    )


def build_prompt_level_summary(model_prompt_question_df: pd.DataFrame) -> pd.DataFrame:
    if model_prompt_question_df.empty:
        return pd.DataFrame()

    group_cols = ["contrast_label", "contrast_group", PROMPT_COL, "question_id"]
    return (
        model_prompt_question_df.groupby(group_cols, dropna=False)
        .agg(
            mean_effect=("mean_effect", "mean"),
            blind_mean=("blind_mean", "mean"),
            target_mean=("target_mean", "mean"),
            n_texts=("n_texts", "mean"),
        )
        .reset_index()
    )


def recompute_subset_outputs(
    text_df: pd.DataFrame,
    subset_mask: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Recompute pooled plot table and significance from a filtered text_df subset
    using the same logic as the main merged analysis.
    """
    sub_text = text_df.loc[subset_mask].copy()
    if sub_text.empty:
        return pd.DataFrame(), pd.DataFrame()

    collapsed_sub = collapse_text_level_effects(sub_text)

    model_prompt_rows = []
    for keys, sub in collapsed_sub.groupby(
        ["contrast_label", "contrast_group", "model", PROMPT_COL], dropna=False
    ):
        contrast_label, contrast_group, model_name, prompt_name = keys

        for q in QUESTION_IDS:
            sq = summarize_series(sub[f"effect_{q}"])
            model_prompt_rows.append({
                "contrast_label": contrast_label,
                "contrast_group": contrast_group,
                "model": model_name,
                PROMPT_COL: prompt_name,
                "question_id": q,
                "blind_mean": float(pd.to_numeric(sub[f"blind_{q}"], errors="coerce").mean()),
                "target_mean": float(pd.to_numeric(sub[q], errors="coerce").mean()),
                "mean_effect": sq["mean"],
                "sd_effect": sq["sd"],
                "se_effect": sq["se"],
                "ci95": sq["ci95"],
                "n_texts": sq["n"],
            })

    sub_model_prompt_question_df = pd.DataFrame(model_prompt_rows)
    sub_pooled_question_df = compute_pooled_question_effects(
        text_df=sub_text,
        model_prompt_question_df=sub_model_prompt_question_df,
    )
    sub_sig_df = compute_pairwise_significance(sub_text)

    return sub_pooled_question_df, sub_sig_df


def make_main_question_plot(
    pooled_question_df: pd.DataFrame,
    text_df: pd.DataFrame,
    output_dir: Path,
    force: bool,
    xlim_min: float,
    xlim_max: float,
) -> None:
    plot_path = output_dir / "baseline_dumbbell_questions_pooled.png"
    sig_path = output_dir / "pairwise_significance.csv"
    sig_df = compute_pairwise_significance(text_df)
    sig_df.to_csv(sig_path, index=False)
    print(f"Saved: {sig_path}")
    export_pairwise_significance_latex(sig_df=sig_df, output_dir=output_dir)

    if plot_path.exists() and not force:
        print(f"Using existing plot: {plot_path}")
        return

    plot_baseline_referenced_question_effects(
        df=pooled_question_df,
        output_path=plot_path,
        xlim_min=xlim_min,
        xlim_max=xlim_max,
        title="Baseline-referenced contextual effects",
        sig_df=sig_df,
    )


def make_model_robustness_outputs(
    model_level_df: pd.DataFrame,
    pooled_question_df: pd.DataFrame,
    output_dir: Path,
    coherence_question: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if model_level_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    summary = _sort_effect_summary(
        compute_robustness_summary(
            unit_df=model_level_df,
            pooled_df=pooled_question_df,
            unit_col="model",
        )
    )
    summary.to_csv(output_dir / "model_robustness_summary.csv", index=False)
    print(f"Saved: {output_dir / 'model_robustness_summary.csv'}")

    model_pair_df = build_paired_difference_table(
        df=model_level_df,
        value_col="mean_effect",
        unit_cols=["model"],
    )
    pooled_pair_df = build_paired_difference_table(
        df=pooled_question_df,
        value_col="mean_effect",
        unit_cols=[],
    )

    model_pair_df.to_csv(output_dir / "model_pair_difference_effects.csv", index=False)
    print(f"Saved: {output_dir / 'model_pair_difference_effects.csv'}")

    pooled_pair_df.to_csv(output_dir / "pooled_pair_difference_effects.csv", index=False)
    print(f"Saved: {output_dir / 'pooled_pair_difference_effects.csv'}")

    pair_summary = _sort_pair_summary(
        compute_paired_difference_robustness_summary(
            unit_pair_df=model_pair_df,
            pooled_pair_df=pooled_pair_df,
            unit_col="model",
        )
    )
    pair_summary.to_csv(output_dir / "model_pair_robustness_summary.csv", index=False)
    print(f"Saved: {output_dir / 'model_pair_robustness_summary.csv'}")

    return model_pair_df, pooled_pair_df, pair_summary


def make_prompt_robustness_outputs(
    prompt_level_df: pd.DataFrame,
    pooled_question_df: pd.DataFrame,
    output_dir: Path,
    coherence_question: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if prompt_level_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    renamed = prompt_level_df.rename(columns={PROMPT_COL: "prompt"})
    summary = _sort_effect_summary(
        compute_robustness_summary(
            unit_df=renamed,
            pooled_df=pooled_question_df,
            unit_col="prompt",
        )
    )
    summary.to_csv(output_dir / "prompt_robustness_summary.csv", index=False)
    print(f"Saved: {output_dir / 'prompt_robustness_summary.csv'}")

    prompt_pair_df = build_paired_difference_table(
        df=renamed,
        value_col="mean_effect",
        unit_cols=["prompt"],
    )
    pooled_pair_df = build_paired_difference_table(
        df=pooled_question_df,
        value_col="mean_effect",
        unit_cols=[],
    )

    prompt_pair_df.to_csv(output_dir / "prompt_pair_difference_effects.csv", index=False)
    print(f"Saved: {output_dir / 'prompt_pair_difference_effects.csv'}")

    if not (output_dir / "pooled_pair_difference_effects.csv").exists():
        pooled_pair_df.to_csv(output_dir / "pooled_pair_difference_effects.csv", index=False)
        print(f"Saved: {output_dir / 'pooled_pair_difference_effects.csv'}")

    pair_summary = _sort_pair_summary(
        compute_paired_difference_robustness_summary(
            unit_pair_df=prompt_pair_df,
            pooled_pair_df=pooled_pair_df,
            unit_col="prompt",
        )
    )
    pair_summary.to_csv(output_dir / "prompt_pair_robustness_summary.csv", index=False)
    print(f"Saved: {output_dir / 'prompt_pair_robustness_summary.csv'}")

    return prompt_pair_df, pooled_pair_df, pair_summary


def make_per_model_question_plots(
    text_df: pd.DataFrame,
    output_dir: Path,
    force: bool,
    xlim_min: float,
    xlim_max: float,
) -> None:
    supp_dir = output_dir / "supplementary" / "per_model_question_plots"
    supp_dir.mkdir(parents=True, exist_ok=True)

    for model_name in sorted(text_df["model"].dropna().astype(str).unique()):
        safe_model = str(model_name).replace("/", "_")
        plot_path = supp_dir / f"baseline_dumbbell_questions_{safe_model}.png"

        if plot_path.exists() and not force:
            print(f"Using existing plot: {plot_path}")
            continue

        sub_pooled_df, sub_sig_df = recompute_subset_outputs(
            text_df=text_df,
            subset_mask=text_df["model"].astype(str) == str(model_name),
        )

        if sub_pooled_df.empty:
            print(f"Skipping model plot for {model_name}: no data.")
            continue

        plot_baseline_referenced_question_effects(
            df=sub_pooled_df,
            output_path=plot_path,
            xlim_min=xlim_min,
            xlim_max=xlim_max,
            title=f"Baseline-referenced contextual effects — {model_name}",
            sig_df=sub_sig_df,
        )


def make_per_prompt_question_plots(
    text_df: pd.DataFrame,
    output_dir: Path,
    force: bool,
    xlim_min: float,
    xlim_max: float,
) -> None:
    supp_dir = output_dir / "supplementary" / "per_prompt_question_plots"
    supp_dir.mkdir(parents=True, exist_ok=True)

    for prompt_name in sorted(text_df[PROMPT_COL].dropna().astype(str).unique()):
        safe_prompt = str(prompt_name).replace("/", "_")
        plot_path = supp_dir / f"baseline_dumbbell_questions_{safe_prompt}.png"

        if plot_path.exists() and not force:
            print(f"Using existing plot: {plot_path}")
            continue

        sub_pooled_df, sub_sig_df = recompute_subset_outputs(
            text_df=text_df,
            subset_mask=text_df[PROMPT_COL].astype(str) == str(prompt_name),
        )

        if sub_pooled_df.empty:
            print(f"Skipping prompt plot for {prompt_name}: no data.")
            continue

        plot_baseline_referenced_question_effects(
            df=sub_pooled_df,
            output_path=plot_path,
            xlim_min=xlim_min,
            xlim_max=xlim_max,
            title=f"Baseline-referenced contextual effects — {prompt_name}",
            sig_df=sub_sig_df,
        )


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text_df, model_prompt_question_df, pooled_question_df = load_or_compute_effects(
        output_dir=output_dir,
        force=args.force,
    )

    additional_diagnostics = export_additional_diagnostics(
        text_df=text_df,
        output_dir=output_dir,
    )

    make_main_question_plot(
        pooled_question_df=pooled_question_df,
        text_df=text_df,
        output_dir=output_dir,
        force=args.force,
        xlim_min=args.xlim_min,
        xlim_max=args.xlim_max,
    )

    model_level_df = build_model_level_summary(model_prompt_question_df)
    model_level_df.to_csv(output_dir / "model_level_effects_compact.csv", index=False)
    print(f"Saved: {output_dir / 'model_level_effects_compact.csv'}")

    prompt_level_df = build_prompt_level_summary(model_prompt_question_df)
    prompt_level_df.to_csv(output_dir / "prompt_level_effects_compact.csv", index=False)
    print(f"Saved: {output_dir / 'prompt_level_effects_compact.csv'}")

    model_pair_df, pooled_pair_df_model, model_pair_summary = make_model_robustness_outputs(
        model_level_df=model_level_df,
        pooled_question_df=pooled_question_df,
        output_dir=output_dir,
        coherence_question=args.coherence_question,
    )

    prompt_pair_df, pooled_pair_df_prompt, prompt_pair_summary = make_prompt_robustness_outputs(
        prompt_level_df=prompt_level_df,
        pooled_question_df=pooled_question_df,
        output_dir=output_dir,
        coherence_question=args.coherence_question,
    )

    mixed_effects_q06_df, mixed_effects_q06_pair_df = export_mixed_effects_outputs(text_df=text_df, output_dir=output_dir)
    permutation_df = export_permutation_outputs(text_df=text_df, output_dir=output_dir)
    export_heterogeneity_tables(
        model_summary=pd.read_csv(output_dir / "model_robustness_summary.csv"),
        prompt_summary=pd.read_csv(output_dir / "prompt_robustness_summary.csv"),
        model_pair_summary=model_pair_summary,
        prompt_pair_summary=prompt_pair_summary,
        output_dir=output_dir,
    )

    plot_combined_coherence(
        prompt_level_df=prompt_level_df,
        model_level_df=model_level_df,
        pooled_df=pooled_question_df,
        question_id=args.coherence_question,
        output_path=output_dir / f"coherence_{args.coherence_question}_prompt_vs_model.png",
        title=f"Coherence for {QUESTION_LABELS[args.coherence_question]}",
    )

    make_per_model_question_plots(
        text_df=text_df,
        output_dir=output_dir,
        force=args.force,
        xlim_min=args.xlim_min,
        xlim_max=args.xlim_max,
    )

    make_per_prompt_question_plots(
        text_df=text_df,
        output_dir=output_dir,
        force=args.force,
        xlim_min=args.xlim_min,
        xlim_max=args.xlim_max,
    )

    print("\nDone.")
    print(f"Text-level rows: {len(text_df):,}")
    print(f"Model-prompt-question rows: {len(model_prompt_question_df):,}")
    print(f"Pooled-question rows: {len(pooled_question_df):,}")
    print(f"Compact model rows: {len(model_level_df):,}")
    print(f"Compact prompt rows: {len(prompt_level_df):,}")
    print(f"Model pair-difference rows: {len(model_pair_df):,}")
    print(f"Prompt pair-difference rows: {len(prompt_pair_df):,}")
    print(f"Pooled pair-difference rows: {max(len(pooled_pair_df_model), len(pooled_pair_df_prompt)):,}")
    print(f"Mixed-effects q06 rows: {len(mixed_effects_q06_df):,}")
    print(f"Permutation rows: {len(permutation_df):,}")
    for name, df in additional_diagnostics.items():
        print(f"Additional diagnostic {name} rows: {len(df):,}")


if __name__ == "__main__":
    main()