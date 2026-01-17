from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd


# Scenario knobs are simulation/design parameters (NOT analysis/pipeline parameters).
SCENARIO_CANDIDATE_COLS: list[str] = [
    "n_genes",
    "guides_per_gene",
    "n_control",
    "n_treatment",
    "depth_log_sd",
    "n_batches",
    "batch_confounding_strength",
    "batch_depth_log_sd",
    "treatment_depth_multiplier",
    "frac_signal",
    "effect_sd",
    "guide_slope_sd",
    "guide_lambda_log_sd",
    "gene_lambda_log_sd",
    "gene_lambda_family",
    "gene_lambda_mix_pi_high",
    "gene_lambda_mix_delta_log_mean",
    "gene_lambda_power_alpha",
    "guide_lambda_family",
    "guide_lambda_dirichlet_alpha0",
    "offtarget_guide_frac",
    "offtarget_slope_sd",
    "nb_overdispersion",
]

_ALIASES: dict[str, str] = {
    "n_genes": "ng",
    "guides_per_gene": "gpg",
    "n_control": "n_ctrl",
    "n_treatment": "n_trt",
    "depth_log_sd": "depth_sd",
    "n_batches": "batches",
    "batch_confounding_strength": "batch_conf",
    "batch_depth_log_sd": "batch_depth_sd",
    "treatment_depth_multiplier": "tdm",
    "frac_signal": "fs",
    "effect_sd": "eff_sd",
    "guide_slope_sd": "guide_slope_sd",
    "guide_lambda_log_sd": "guide_ll_sd",
    "gene_lambda_log_sd": "gene_ll_sd",
    "gene_lambda_family": "gene_ll_fam",
    "gene_lambda_mix_pi_high": "mix_pi",
    "gene_lambda_mix_delta_log_mean": "mix_dlog",
    "gene_lambda_power_alpha": "pl_alpha",
    "guide_lambda_family": "guide_ll_fam",
    "guide_lambda_dirichlet_alpha0": "dir_a0",
    "offtarget_guide_frac": "ot_frac",
    "offtarget_slope_sd": "ot_sd",
    "nb_overdispersion": "nb_phi",
}

_CATEGORICAL_SCENARIO_COLS: set[str] = {
    "gene_lambda_family",
    "guide_lambda_family",
}


def _fmt_num(x: object) -> str:
    v = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    if not np.isfinite(v):
        return "NA"
    if float(v).is_integer():
        return str(int(v))
    return f"{float(v):g}"


def _row_hash(row: pd.Series, *, scenario_cols: list[str]) -> str:
    payload = {c: row.get(c) for c in scenario_cols}
    b = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha1(b).hexdigest()[:8]


def make_scenario_table(df: pd.DataFrame, *, exclude_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Build a scenario table (one row per unique simulated scenario), with a stable label.

    A "scenario" is defined only by simulation/design knobs (not analysis/pipeline knobs).
    The label includes only knobs that vary in the provided df, to keep names readable.
    """

    exclude = set([str(c) for c in (exclude_cols or [])])
    scenario_cols = [c for c in SCENARIO_CANDIDATE_COLS if (c in df.columns and c not in exclude)]
    if not scenario_cols:
        out = pd.DataFrame({"scenario": ["scenario"], "scenario_id": ["00000000"], "is_null": [False]})
        return out

    scenarios = df[scenario_cols].drop_duplicates().copy()
    for c in scenario_cols:
        if c in _CATEGORICAL_SCENARIO_COLS:
            scenarios[c] = scenarios[c].astype(str)
        else:
            scenarios[c] = pd.to_numeric(scenarios[c], errors="coerce")

    # Stable ordering by raw scenario params.
    sort_cols = [c for c in SCENARIO_CANDIDATE_COLS if c in scenarios.columns]
    scenarios = scenarios.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    fs = pd.to_numeric(scenarios.get("frac_signal", 0.0), errors="coerce").fillna(0.0)
    scenarios["is_null"] = fs == 0.0

    varying: list[str] = []
    for c in scenario_cols:
        if c in _CATEGORICAL_SCENARIO_COLS:
            if int(df[c].astype(str).nunique(dropna=False)) > 1:
                varying.append(c)
        else:
            s = pd.to_numeric(df[c], errors="coerce")
            if int(s.dropna().nunique()) > 1:
                varying.append(c)

    fs_nonzero_unique = pd.to_numeric(scenarios.loc[~scenarios["is_null"], "frac_signal"], errors="coerce").dropna().unique()
    include_fs = int(pd.to_numeric(scenarios.get("frac_signal", 0.0), errors="coerce").dropna().nunique()) > 2 or int(
        len(fs_nonzero_unique)
    ) > 1
    varying_for_label = [c for c in varying if c in _ALIASES and (c != "frac_signal" or include_fs)]

    baseline = {
        "treatment_depth_multiplier": 1.0,
        "n_batches": 1.0,
        "batch_confounding_strength": 0.0,
        "batch_depth_log_sd": 0.0,
        "offtarget_guide_frac": 0.0,
        "offtarget_slope_sd": 0.0,
        "nb_overdispersion": 0.0,
    }

    scenario_ids: list[str] = []
    labels: list[str] = []
    for r in scenarios.itertuples(index=False):
        row = pd.Series(r._asdict())
        scenario_id = _row_hash(row, scenario_cols=scenario_cols)
        scenario_ids.append(scenario_id)

        base = "null" if bool(row.get("is_null")) else "signal"
        parts = [base]

        n_batches = pd.to_numeric(row.get("n_batches", np.nan), errors="coerce")
        n_batches = float(n_batches) if np.isfinite(n_batches) else np.nan
        ot_frac = pd.to_numeric(row.get("offtarget_guide_frac", np.nan), errors="coerce")
        ot_frac = float(ot_frac) if np.isfinite(ot_frac) else np.nan

        for c in varying_for_label:
            val = row.get(c)
            if c in _CATEGORICAL_SCENARIO_COLS:
                parts.append(f"{_ALIASES[c]}={str(val)}")
                continue
            if c in baseline:
                v = pd.to_numeric(val, errors="coerce")
                v = float(v) if np.isfinite(v) else np.nan
                if np.isfinite(v) and np.isfinite(float(baseline[c])) and v == float(baseline[c]):
                    continue
            if c in {"batch_confounding_strength", "batch_depth_log_sd"} and (not np.isfinite(n_batches) or n_batches <= 1):
                continue
            if c == "offtarget_slope_sd" and (not np.isfinite(ot_frac) or ot_frac <= 0.0):
                continue
            parts.append(f"{_ALIASES[c]}={_fmt_num(val)}")
        labels.append("; ".join(parts))

    scenarios["scenario_id"] = scenario_ids
    scenarios["scenario"] = labels

    # Ensure scenario labels are unique (avoid ambiguous column names).
    if bool(scenarios["scenario"].duplicated().any()):
        dup = scenarios["scenario"].duplicated(keep=False)
        scenarios.loc[dup, "scenario"] = scenarios.loc[dup].apply(
            lambda r: f"{r['scenario']} [id={r['scenario_id']}]",
            axis=1,
        )

    return scenarios[scenario_cols + ["scenario_id", "is_null", "scenario"]]


def attach_scenarios(df: pd.DataFrame, *, exclude_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Return df with added columns: scenario_id, is_null, scenario.
    """

    scenarios = make_scenario_table(df, exclude_cols=exclude_cols)
    scenario_cols = [c for c in scenarios.columns if c not in {"scenario", "scenario_id", "is_null"}]
    return df.merge(scenarios, on=scenario_cols, how="left", validate="many_to_one")
