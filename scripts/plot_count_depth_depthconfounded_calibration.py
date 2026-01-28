from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from suite_paths import ReportPathResolver


def _require_matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _savefig(fig, out_path: str) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=r"^Tight layout not applied\..*")
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=r"^This figure includes Axes that are not compatible with tight_layout.*",
        )
        fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)


@dataclass(frozen=True)
class PipelineSpec:
    method: str  # meta|stouffer|lmm_lrt|lmm_wald
    depth_covariate_mode: str  # none|log_libsize
    label: str
    color: str
    linestyle: str


def _method_display(method: str) -> str:
    m = str(method)
    if m == "meta":
        return "Meta (RE)"
    if m == "stouffer":
        return "Stouffer (t)"
    if m == "lmm_lrt":
        return "LMM (LRT)"
    if m == "lmm_wald":
        return "LMM (Wald)"
    return m


def _depth_display(depth_covariate_mode: str) -> str:
    d = str(depth_covariate_mode)
    if d == "none":
        return "no depth cov"
    return f"depth cov: {d}"


def _pipeline_display(method: str, depth_covariate_mode: str) -> str:
    return f"{_method_display(method)} | {_depth_display(depth_covariate_mode)}"


def _load_truth(run_dir: Path) -> pd.DataFrame:
    truth = pd.read_csv(run_dir / "sim_truth_gene.tsv", sep="\t")
    for c in ["is_reference", "is_signal"]:
        if c in truth.columns:
            truth[c] = truth[c].astype(bool)
    if "gene_id" not in truth.columns:
        raise ValueError(f"missing gene_id in truth table: {run_dir!s}")
    return truth


def _load_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def _method_long(
    truth: pd.DataFrame,
    *,
    method_name: str,
    df: pd.DataFrame,
    p_col: str,
    p_adj_col: str,
) -> pd.DataFrame:
    if df.empty:
        out = truth[["gene_id", "is_reference", "is_signal"]].copy()
        out["method"] = str(method_name)
        out["p"] = np.nan
        out["p_adj"] = np.nan
        return out

    sub = df.copy()
    if "focal_var" in sub.columns:
        sub = sub.loc[sub["focal_var"].astype(str) == "treatment"].copy()

    cols = ["gene_id"]
    for c in [p_col, p_adj_col]:
        if c in sub.columns:
            cols.append(c)
    sub = sub[cols].copy()
    sub = sub.rename(columns={p_col: "p", p_adj_col: "p_adj"})

    out = truth.merge(sub, on="gene_id", how="left")
    out["method"] = str(method_name)
    out["p"] = pd.to_numeric(out.get("p"), errors="coerce")
    out["p_adj"] = pd.to_numeric(out.get("p_adj"), errors="coerce")
    return out


def _fdp_curve_one_run(
    df: pd.DataFrame,
    *,
    q_grid: np.ndarray,
) -> np.ndarray:
    qv = pd.to_numeric(df["p_adj"], errors="coerce").to_numpy(dtype=float)
    is_signal = df["is_signal"].to_numpy(dtype=bool)
    out = np.zeros_like(q_grid, dtype=float)
    for i, q in enumerate(q_grid):
        called = np.isfinite(qv) & (qv <= float(q))
        n_called = int(np.sum(called))
        if n_called <= 0:
            out[i] = 0.0
        else:
            fp = int(np.sum(called & (~is_signal)))
            out[i] = float(fp / n_called)
    return out


def _summary_one_run(df: pd.DataFrame, *, q: float) -> tuple[float, int]:
    qv = pd.to_numeric(df["p_adj"], errors="coerce").to_numpy(dtype=float)
    is_signal = df["is_signal"].to_numpy(dtype=bool)
    called = np.isfinite(qv) & (qv <= float(q))
    n_called = int(np.sum(called))
    if n_called <= 0:
        return 0.0, 0
    fp = int(np.sum(called & (~is_signal)))
    return float(fp / n_called), n_called


def _qq_arrays(p: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (expected, observed, lo95, hi95) in -log10 scale for QQ plots.
    """

    p = pd.to_numeric(p, errors="coerce").dropna()
    p = p[(p > 0.0) & (p <= 1.0)]
    if p.empty:
        return np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([])

    p_sorted = np.sort(p.to_numpy(dtype=float))
    n = int(p_sorted.size)
    exp = (np.arange(1, n + 1, dtype=float) / (n + 1.0)).astype(float)

    # Order-statistic envelope: U_(i) ~ Beta(i, n-i+1)
    try:
        from scipy.stats import beta as beta_dist

        i = np.arange(1, n + 1, dtype=float)
        lo = beta_dist.ppf(0.025, i, (n - i + 1.0))
        hi = beta_dist.ppf(0.975, i, (n - i + 1.0))
    except Exception:
        lo = np.full(n, np.nan)
        hi = np.full(n, np.nan)

    return -np.log10(exp), -np.log10(p_sorted), -np.log10(lo), -np.log10(hi)


def _color_for_method(method: str) -> str:
    m = str(method)
    return {
        "meta": "#1f77b4",
        "stouffer": "#ff7f0e",
        "lmm_lrt": "#2ca02c",
        "lmm_wald": "#d62728",
    }.get(m, "black")


def _ordered_methods() -> list[str]:
    return ["meta", "stouffer", "lmm_lrt", "lmm_wald"]


def _ordered_depthcov() -> list[str]:
    return ["none", "log_libsize"]


def _method_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "meta": run_dir / "PMD_std_res_gene_meta.tsv",
        "stouffer": run_dir / "PMD_std_res_gene_stouffer.tsv",
        "lmm": run_dir / "PMD_std_res_gene_lmm.tsv",
    }


def _stable_hash(obj: object) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:10]


def _fmt_num(x: object) -> str:
    v = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    if not np.isfinite(v):
        return "NA"
    if float(v).is_integer():
        return str(int(v))
    return f"{float(v):g}"


def _bucket_dirname(bucket: str) -> str:
    b = str(bucket)
    mapping = {
        "all_nonref": "all_nonref",
        "<1": "lt1",
        "1-<3": "1to3",
        "3-<5": "3to5",
        ">=5": "ge5",
        "NA": "NA",
    }
    return mapping.get(b, b.replace("<", "lt").replace(">=", "ge").replace(">", "gt").replace("=", "eq"))


def _load_expected_bucket(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "sim_gene_expected_counts.tsv"
    if not path.is_file():
        return pd.DataFrame({"gene_id": [], "expected_p10_mincond_bucket": []})
    df = pd.read_csv(path, sep="\t")
    if "gene_id" not in df.columns:
        return pd.DataFrame({"gene_id": [], "expected_p10_mincond_bucket": []})
    if "expected_p10_mincond_bucket" not in df.columns:
        df["expected_p10_mincond_bucket"] = np.nan
    return df[["gene_id", "expected_p10_mincond_bucket"]].copy()


def _abundance_signature(row: pd.Series) -> dict[str, object]:
    glf = str(row.get("gene_lambda_family", "lognormal"))
    guf = str(row.get("guide_lambda_family", "lognormal_noise"))

    sig: dict[str, object] = {
        "gene_lambda_family": glf,
        "gene_lambda_log_sd": _fmt_num(row.get("gene_lambda_log_sd")),
        "guide_lambda_family": guf,
        "guide_lambda_log_mean": _fmt_num(row.get("guide_lambda_log_mean")),
        "guide_lambda_log_sd": _fmt_num(row.get("guide_lambda_log_sd")),
    }

    if glf == "mixture_lognormal":
        sig["gene_lambda_mix_pi_high"] = _fmt_num(row.get("gene_lambda_mix_pi_high"))
        sig["gene_lambda_mix_delta_log_mean"] = _fmt_num(row.get("gene_lambda_mix_delta_log_mean"))
    if glf == "power_law":
        sig["gene_lambda_power_alpha"] = _fmt_num(row.get("gene_lambda_power_alpha"))
    if guf == "dirichlet_weights":
        sig["guide_lambda_dirichlet_alpha0"] = _fmt_num(row.get("guide_lambda_dirichlet_alpha0"))

    return sig


def _abundance_label(sig: dict[str, object]) -> str:
    glf = str(sig.get("gene_lambda_family", "lognormal"))
    if glf == "lognormal":
        glf_tag = "glf=ln"
    elif glf == "mixture_lognormal":
        glf_tag = f"glf=mln_pi={sig.get('gene_lambda_mix_pi_high','NA')}_dlog={sig.get('gene_lambda_mix_delta_log_mean','NA')}"
    elif glf == "power_law":
        glf_tag = f"glf=pl_a={sig.get('gene_lambda_power_alpha','NA')}"
    else:
        glf_tag = f"glf={glf}"

    guf = str(sig.get("guide_lambda_family", "lognormal_noise"))
    if guf == "lognormal_noise":
        guf_tag = "guf=lnn"
    elif guf == "dirichlet_weights":
        guf_tag = f"guf=dir_a0={sig.get('guide_lambda_dirichlet_alpha0','NA')}"
    else:
        guf_tag = f"guf={guf}"

    glm = sig.get("guide_lambda_log_mean", "NA")
    glsd = sig.get("gene_lambda_log_sd", "NA")
    gulsd = sig.get("guide_lambda_log_sd", "NA")
    return f"{glf_tag}__gene_sd={glsd}__{guf_tag}__base_logmean={glm}__guide_sd={gulsd}"


def _load_run_long_tables(run_dir: Path) -> dict[str, pd.DataFrame]:
    truth = _load_truth(run_dir)
    expected = _load_expected_bucket(run_dir)

    paths = _method_paths(run_dir)
    meta = _load_table(paths["meta"])
    stouffer = _load_table(paths["stouffer"])
    lmm = _load_table(paths["lmm"])

    out: dict[str, pd.DataFrame] = {}
    out["meta"] = _method_long(truth, method_name="meta", df=meta, p_col="p", p_adj_col="p_adj").merge(
        expected, on="gene_id", how="left"
    )
    out["stouffer"] = _method_long(truth, method_name="stouffer", df=stouffer, p_col="p", p_adj_col="p_adj").merge(
        expected, on="gene_id", how="left"
    )
    out["lmm_lrt"] = _method_long(truth, method_name="lmm_lrt", df=lmm, p_col="lrt_p", p_adj_col="lrt_p_adj").merge(
        expected, on="gene_id", how="left"
    )
    out["lmm_wald"] = _method_long(truth, method_name="lmm_wald", df=lmm, p_col="wald_p", p_adj_col="wald_p_adj").merge(
        expected, on="gene_id", how="left"
    )
    return out


def _plot_dashboard(
    *,
    out_path: str,
    title: str,
    pipelines: list[PipelineSpec],
    qq_null_p_by_pipeline: dict[str, pd.Series],
    fdp_curves_by_pipeline: dict[str, np.ndarray],
    fdp_summary_by_pipeline: dict[str, dict[str, float]],
    q_grid: np.ndarray,
) -> None:
    plt = _require_matplotlib()

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[2.2, 1.0], width_ratios=[1.3, 0.7], hspace=0.25, wspace=0.25)

    qq_gs = gs[0, :].subgridspec(nrows=len(_ordered_methods()), ncols=len(_ordered_depthcov()), hspace=0.25, wspace=0.20)
    ax_fdp = fig.add_subplot(gs[1, 0])
    ax_summary = fig.add_subplot(gs[1, 1])

    # QQ grid
    for r, method in enumerate(_ordered_methods()):
        for c, depth in enumerate(_ordered_depthcov()):
            ax = fig.add_subplot(qq_gs[r, c])
            key = f"{method}__{depth}"
            p = qq_null_p_by_pipeline.get(key, pd.Series(dtype=float))
            exp, obs, lo, hi = _qq_arrays(p)
            if exp.size:
                ax.plot(exp, obs, color=_color_for_method(method), linewidth=1.2)
                ax.plot([0, float(np.nanmax([exp.max(), obs.max()]))], [0, float(np.nanmax([exp.max(), obs.max()]))], color="0.6", linestyle="--", linewidth=1)
                if np.isfinite(lo).any() and np.isfinite(hi).any():
                    ax.fill_between(exp, lo, hi, color="0.85", alpha=0.8, linewidth=0)
            ax.set_xlabel("expected -log10(p)")
            ax.set_ylabel("observed -log10(p)")
            ax.set_title(_pipeline_display(method, depth), fontsize=10)

            # quick calibration text
            try:
                from benchmark_count_depth import _qq_stats  # noqa: PLC0415

                qq = _qq_stats(p) if not p.empty else {"n": 0, "lambda_gc": None}
                lam = qq.get("lambda_gc")
                n = int(qq.get("n") or 0)
            except Exception:
                lam = None
                n = int(pd.to_numeric(p, errors="coerce").dropna().shape[0])
            prop = float(np.mean(pd.to_numeric(p, errors="coerce").dropna().to_numpy(dtype=float) < 0.05)) if n > 0 else np.nan
            txt = f"n={n}"
            if lam is not None and np.isfinite(float(lam)):
                txt += f"  λGC={float(lam):.3g}"
            if np.isfinite(prop):
                txt += f"  P<0.05={prop:.3g}"
            ax.text(0.02, 0.98, txt, transform=ax.transAxes, ha="left", va="top", fontsize=9)

    # FDP calibration curve
    ax_fdp.plot([0, float(q_grid.max())], [0, float(q_grid.max())], color="0.6", linestyle="--", linewidth=1, label="ideal")
    for p in pipelines:
        k = f"{p.method}__{p.depth_covariate_mode}"
        curve = fdp_curves_by_pipeline.get(k)
        if curve is None:
            continue
        ax_fdp.plot(q_grid, curve, color=p.color, linestyle=p.linestyle, linewidth=2, label=p.label)
    ax_fdp.set_title("Empirical FDP vs nominal q (BH)")
    ax_fdp.set_xlabel("nominal q")
    ax_fdp.set_ylabel("realized FDP")
    ax_fdp.set_xlim(0.0, float(q_grid.max()))
    ax_fdp.set_ylim(0.0, 1.0)
    ax_fdp.legend(loc="upper left", fontsize=9, ncol=2)

    # Summary dots
    y = np.arange(len(pipelines), dtype=float)[::-1]
    ax_summary.set_yticks(y)
    ax_summary.set_yticklabels([p.label for p in pipelines][::-1], fontsize=9)
    ax_summary.set_xlim(0.0, 1.0)
    ax_summary.set_xlabel("FDP")
    ax_summary.set_title("FDP @ q and discoveries")

    for i, p in enumerate(pipelines[::-1]):
        k = f"{p.method}__{p.depth_covariate_mode}"
        s = fdp_summary_by_pipeline.get(k, {})
        fdp05 = float(s.get("fdp_q05", np.nan))
        fdp10 = float(s.get("fdp_q10", np.nan))
        n10 = int(s.get("n_called_q10", 0))
        yy = float(y[i])
        ax_summary.scatter([fdp05], [yy], color=p.color, marker="o", s=30, zorder=3)
        ax_summary.scatter([fdp10], [yy], color=p.color, marker="s", s=30, zorder=3)
        ax_summary.text(1.01, yy, f"n@0.10={n10}", ha="left", va="center", fontsize=9, transform=ax_summary.get_yaxis_transform())

    ax_summary.text(0.02, -0.18, "o: q=0.05   s: q=0.10", transform=ax_summary.transAxes, fontsize=9)

    fig.suptitle(title, fontsize=14)
    _savefig(fig, out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Treatment-depth calibration dashboards (QQ + FDP) for count-depth benchmarks (suite output consumer). "
            "Intended to diagnose depth confounding by sweeping treatment-depth multipliers."
        )
    )
    parser.add_argument("--grid-tsv", required=True, type=str, help="Suite-level count_depth_grid_summary.tsv (must include report_path).")
    parser.add_argument("--out-dir", required=True, type=str, help="Output directory for calibration dashboards.")
    parser.add_argument(
        "--treatment-depth-multiplier",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Treatment depth multipliers to plot (e.g. 2 10). "
            "If not set, uses all values present in the grid for the selected scenario."
        ),
    )
    parser.add_argument(
        "--frac-signal",
        type=float,
        default=0.2,
        help="frac_signal to use (default: 0.2).",
    )
    parser.add_argument(
        "--normalization-mode",
        type=str,
        default="none",
        help="Normalization mode filter for non-PMD response modes (default: none).",
    )
    parser.add_argument(
        "--logratio-mode",
        type=str,
        default="none",
        help="Log-ratio mode filter for non-PMD response modes (default: none).",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    grid = pd.read_csv(args.grid_tsv, sep="\t")
    if "report_path" not in grid.columns:
        raise ValueError("--grid-tsv must include report_path (suite output) to locate per-run TSVs")

    grid_tsv_path = Path(args.grid_tsv).resolve()
    fs = float(args.frac_signal)

    # Depth-confounded signal condition: no other scenario stressors enabled.
    sub = grid.copy()
    # Back-compat: older grid TSVs may not include newer scenario columns.
    # Treat missing columns as their "no additional stressor" defaults so the
    # calibration plots can still run on minimal suite summaries.
    defaults = {
        "n_batches": 1,
        "batch_confounding_strength": 0.0,
        "batch_depth_log_sd": 0.0,
        "offtarget_guide_frac": 0.0,
        "nb_overdispersion": 0.0,
    }
    for col, default in defaults.items():
        if col not in sub.columns:
            sub[col] = default
    sub["treatment_depth_multiplier"] = pd.to_numeric(sub["treatment_depth_multiplier"], errors="coerce")
    sub["frac_signal"] = pd.to_numeric(sub["frac_signal"], errors="coerce")
    sub["n_batches"] = pd.to_numeric(sub["n_batches"], errors="coerce")
    sub["batch_confounding_strength"] = pd.to_numeric(sub["batch_confounding_strength"], errors="coerce")
    sub["batch_depth_log_sd"] = pd.to_numeric(sub["batch_depth_log_sd"], errors="coerce")
    sub["offtarget_guide_frac"] = pd.to_numeric(sub["offtarget_guide_frac"], errors="coerce")
    sub["nb_overdispersion"] = pd.to_numeric(sub["nb_overdispersion"], errors="coerce")

    sub = sub.loc[
        (sub["frac_signal"] == fs)
        & (sub["n_batches"] == 1)
        & (sub["batch_confounding_strength"] == 0.0)
        & (sub["batch_depth_log_sd"] == 0.0)
        & (sub["offtarget_guide_frac"] == 0.0)
        & (sub["nb_overdispersion"] == 0.0)
    ].copy()

    if sub.empty:
        print("treatment-depth calibration: no matching runs; nothing to plot", file=sys.stderr)
        return

    available_tdms = sorted([float(x) for x in sub["treatment_depth_multiplier"].dropna().unique().tolist()])
    requested_tdms = args.treatment_depth_multiplier
    if requested_tdms is None:
        tdms = available_tdms
    else:
        tdms = []
        for req in [float(x) for x in requested_tdms]:
            hit = None
            for a in available_tdms:
                if np.isfinite(a) and np.isclose(a, req, rtol=0.0, atol=1e-9):
                    hit = a
                    break
            if hit is None:
                print(
                    f"treatment-depth calibration: skip tdm={req:g} (no matching runs in grid after filters)",
                    file=sys.stderr,
                )
                continue
            tdms.append(float(hit))
        tdms = sorted(tdms)

    resolver = ReportPathResolver.from_grid_tsv(grid_tsv_path)

    q_grid = np.linspace(0.0, 0.2, 41, dtype=float)  # 0.00..0.20 step 0.005
    buckets = ["all_nonref", ">=5", "3-<5", "1-<3", "<1"]
    manifest: dict[str, object] = {
        "grid_tsv": str(args.grid_tsv),
        "filters": {
            "treatment_depth_multipliers": [float(x) for x in (tdms or [])],
            "frac_signal": fs,
            "normalization_mode": str(args.normalization_mode),
            "logratio_mode": str(args.logratio_mode),
            "scenario_guards": {
                "n_batches": 1,
                "batch_confounding_strength": 0.0,
                "batch_depth_log_sd": 0.0,
                "offtarget_guide_frac": 0.0,
                "nb_overdispersion": 0.0,
            },
        },
        "abundance_groups": {},
        "outputs": {},
    }

    for tdm in tdms:
        tdm_sub = sub.loc[np.isclose(sub["treatment_depth_multiplier"], float(tdm), rtol=0.0, atol=1e-9)].copy()
        if tdm_sub.empty:
            continue
        # Compute abundance-group IDs (gene/guide lambda families + key params).
        abundance_specs: dict[str, dict[str, object]] = {}
        abundance_ids: list[str] = []
        abundance_labels: list[str] = []
        for row in tdm_sub.itertuples(index=False):
            row_s = pd.Series(row._asdict())
            sig = _abundance_signature(row_s)
            aid = _stable_hash(sig)
            if aid not in abundance_specs:
                abundance_specs[aid] = {"signature": sig, "label": _abundance_label(sig)}
            abundance_ids.append(aid)
            abundance_labels.append(str(abundance_specs[aid]["label"]))
        tdm_sub = tdm_sub.copy()
        tdm_sub["abundance_id"] = abundance_ids
        tdm_sub["abundance_label"] = abundance_labels

        # Record signatures once for the manifest.
        for aid, spec in abundance_specs.items():
            if aid not in manifest["abundance_groups"]:
                manifest["abundance_groups"][aid] = spec

        tdm_key = f"tdm={tdm:g}"
        manifest["outputs"][tdm_key] = {}

        for aid, gdf in tdm_sub.groupby("abundance_id", sort=True):
            spec = abundance_specs.get(str(aid), {})
            abundance_label = str(spec.get("label", "abundance"))
            group_key = f"abundance={abundance_label}__h={aid}"
            manifest["outputs"][tdm_key][group_key] = {}

            response_modes = sorted(gdf["response_mode"].astype(str).unique().tolist())
            for rm in response_modes:
                rm_sub = gdf.loc[gdf["response_mode"].astype(str) == str(rm)].copy()
                if rm != "pmd_std_res":
                    rm_sub = rm_sub.loc[
                        (rm_sub["normalization_mode"].astype(str) == str(args.normalization_mode))
                        & (rm_sub["logratio_mode"].astype(str) == str(args.logratio_mode))
                    ].copy()
                if rm_sub.empty:
                    continue

                # Predeclare pipeline visuals (method color, depthcov linestyle).
                pipelines: list[PipelineSpec] = []
                for method in _ordered_methods():
                    for depth_mode in _ordered_depthcov():
                        label = _pipeline_display(method, depth_mode)
                        pipelines.append(
                            PipelineSpec(
                                method=str(method),
                                depth_covariate_mode=str(depth_mode),
                                label=label,
                                color=_color_for_method(method),
                                linestyle="-" if depth_mode != "none" else "--",
                            )
                        )

                # Accumulate run-level data per bucket (avoid mixing across genes with no membership).
                qq_null_p: dict[str, dict[str, list[pd.Series]]] = {
                    b: {f"{m}__{d}": [] for m in _ordered_methods() for d in _ordered_depthcov()} for b in buckets
                }
                fdp_curves: dict[str, dict[str, list[np.ndarray]]] = {
                    b: {f"{m}__{d}": [] for m in _ordered_methods() for d in _ordered_depthcov()} for b in buckets
                }
                fdp_summary: dict[str, dict[str, list[dict[str, float]]]] = {
                    b: {f"{m}__{d}": [] for m in _ordered_methods() for d in _ordered_depthcov()} for b in buckets
                }

                for row in rm_sub.itertuples(index=False):
                    row_s = pd.Series(row._asdict())
                    report_path = resolver.resolve_report_path(str(row_s["report_path"]))
                    run_dir = report_path.parent
                    depth_mode = str(row_s.get("depth_covariate_mode", ""))
                    if depth_mode not in set(_ordered_depthcov()):
                        continue

                    try:
                        run_tables = _load_run_long_tables(run_dir)
                    except FileNotFoundError:
                        # Suite-level plots should be resilient to minimal grid TSVs
                        # used in smoke tests (report-only; no per-run artifacts).
                        continue
                    for method in _ordered_methods():
                        df_long = run_tables.get(method, pd.DataFrame())
                        if df_long.empty:
                            continue
                        df_long = df_long.loc[~df_long["is_reference"].astype(bool)].copy()
                        key = f"{method}__{depth_mode}"

                        for b in buckets:
                            if b == "all_nonref":
                                df_b = df_long
                            else:
                                df_b = df_long.loc[df_long["expected_p10_mincond_bucket"].astype(str) == str(b)].copy()
                            if df_b.empty:
                                continue

                            # QQ: null p-values (raw p only).
                            qq_null_p[b][key].append(df_b.loc[~df_b["is_signal"], "p"])

                            # FDP curves: per-run BH adjusted p-values.
                            fdp_curves[b][key].append(_fdp_curve_one_run(df_b, q_grid=q_grid))
                            fdp05, _ = _summary_one_run(df_b, q=0.05)
                            fdp10, n10 = _summary_one_run(df_b, q=0.10)
                            fdp_summary[b][key].append({"fdp_q05": float(fdp05), "fdp_q10": float(fdp10), "n_called_q10": float(n10)})

                out_base = Path(args.out_dir) / tdm_key / group_key / f"resp={rm}"
                out_base.mkdir(parents=True, exist_ok=True)
                manifest["outputs"][tdm_key][group_key][rm] = {"n_runs": int(rm_sub.shape[0]), "buckets": {}}

                for b in buckets:
                    # Reduce to pooled series/mean curves.
                    qq_null_p_final: dict[str, pd.Series] = {}
                    fdp_curves_final: dict[str, np.ndarray] = {}
                    fdp_summary_final: dict[str, dict[str, float]] = {}

                    has_any = False
                    for p in pipelines:
                        k = f"{p.method}__{p.depth_covariate_mode}"
                        pooled = (
                            pd.concat(qq_null_p[b].get(k, []), axis=0, ignore_index=True)
                            if qq_null_p[b].get(k)
                            else pd.Series(dtype=float)
                        )
                        qq_null_p_final[k] = pooled
                        if not pooled.empty:
                            has_any = True

                        curves = fdp_curves[b].get(k, [])
                        if curves:
                            m = np.nanmean(np.vstack(curves), axis=0)
                            fdp_curves_final[k] = m.astype(float)
                        else:
                            fdp_curves_final[k] = np.full_like(q_grid, np.nan, dtype=float)

                        summ = fdp_summary[b].get(k, [])
                        if summ:
                            s_df = pd.DataFrame(summ)
                            fdp_summary_final[k] = {
                                "fdp_q05": float(np.nanmean(pd.to_numeric(s_df["fdp_q05"], errors="coerce").to_numpy(dtype=float))),
                                "fdp_q10": float(np.nanmean(pd.to_numeric(s_df["fdp_q10"], errors="coerce").to_numpy(dtype=float))),
                                "n_called_q10": float(np.nanmean(pd.to_numeric(s_df["n_called_q10"], errors="coerce").to_numpy(dtype=float))),
                            }
                        else:
                            fdp_summary_final[k] = {"fdp_q05": np.nan, "fdp_q10": np.nan, "n_called_q10": 0.0}

                    if not has_any:
                        continue

                    bucket_dir = out_base / f"bucket={_bucket_dirname(b)}"
                    bucket_dir.mkdir(parents=True, exist_ok=True)

                    title = (
                        f"Treatment-depth calibration ({rm})\n"
                        f"{abundance_label}\n"
                        f"tdm={tdm:g}, bucket={b}, frac_signal={fs:g}, norm={str(args.normalization_mode)}, lr={str(args.logratio_mode)}"
                    )
                    out_path = bucket_dir / f"calibration_dashboard__fs={fs:g}__norm={str(args.normalization_mode)}__lr={str(args.logratio_mode)}.png"

                    _plot_dashboard(
                        out_path=str(out_path),
                        title=title,
                        pipelines=pipelines,
                        qq_null_p_by_pipeline=qq_null_p_final,
                        fdp_curves_by_pipeline=fdp_curves_final,
                        fdp_summary_by_pipeline=fdp_summary_final,
                        q_grid=q_grid,
                    )

                    manifest["outputs"][tdm_key][group_key][rm]["buckets"][str(b)] = {
                        "bucket_dir": str(bucket_dir),
                        "dashboard_png": str(out_path),
                    }

    manifest_path = Path(args.out_dir) / "calibration_depthconfounded_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(str(manifest_path))


if __name__ == "__main__":
    main()
