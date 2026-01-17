from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from datetime import timezone
from importlib import metadata
from pathlib import Path

import pandas as pd


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _script_path(name: str) -> str:
    return os.path.join(os.path.dirname(__file__), name)


def _dir_is_empty(path: str) -> bool:
    if not os.path.isdir(path):
        return True
    return not any(os.scandir(path))


def _try_git_info() -> dict[str, object]:
    def _git(args: list[str]) -> str:
        proc = subprocess.run(["git", *args], check=False, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "git failed")
        return proc.stdout.strip()

    info: dict[str, object] = {}
    try:
        info["head"] = _git(["rev-parse", "HEAD"])
        info["branch"] = _git(["rev-parse", "--abbrev-ref", "HEAD"])
        info["describe"] = _git(["describe", "--tags", "--always", "--dirty"])
        info["is_dirty"] = bool(_git(["status", "--porcelain"]))
    except Exception as e:  # noqa: BLE001 - best-effort capture only
        info["error"] = str(e)
    return info


def _package_versions(names: list[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except Exception:  # noqa: BLE001 - missing or non-standard install
            out[n] = None
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local count-depth benchmark suite (grid → aggregate → figures).")
    parser.add_argument("--out-dir", required=True, type=str, help="Root output directory for the suite.")
    parser.add_argument(
        "--preset",
        type=str,
        choices=["quick", "standard", "full", "abundance"],
        default=None,
        help="Optional named grid preset (applies only when the suite runs the grid).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="If >0, pass --jobs to the grid runner (default: 0 uses grid default).",
    )
    parser.add_argument(
        "--grid-tsv",
        default=None,
        type=str,
        help="Optional existing raw count_depth_grid_summary.tsv (must include report_path). If not set, the suite runs the grid.",
    )
    parser.add_argument(
        "--grid-out-dir",
        default=None,
        type=str,
        help="Where to write grid run directories (default: <out-dir>/grid_runs).",
    )
    parser.add_argument(
        "--grid-args",
        nargs="*",
        help=(
            "Extra args passed to scripts/run_count_depth_grid.py (excluding --out-dir). "
            "Any unrecognized args are also forwarded to the grid runner."
        ),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If enabled, reuse existing <grid-out-dir>/count_depth_grid_summary.tsv when present. "
            "Also allows writing into a non-empty --out-dir (default: disabled)."
        ),
    )
    parser.add_argument(
        "--force-resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, allow --resume across differing git HEADs (default: disabled).",
    )
    parser.add_argument(
        "--heatmaps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, generate heatmap panels (can produce many files; default: disabled).",
    )
    args, unknown = parser.parse_known_args()
    unknown = [a for a in unknown if a != "--"]
    preset_args: list[str] = []
    if args.preset == "quick":
        preset_args = [
            "--seeds",
            "1",
            "--n-genes",
            "80",
            "--frac-signal",
            "0.0",
            "0.2",
            "--treatment-depth-multiplier",
            "1.0",
            "--no-include-batch-covariate",
            "--no-include-depth-covariate",
            "--methods",
            "meta",
            "stouffer",
            "lmm",
            "qc",
            "--lmm-scope",
            "meta_or_het_fdr",
            "--lmm-max-genes-per-focal-var",
            "50",
            "--max-iter",
            "60",
            "--qq-plots",
        ]
    elif args.preset == "standard":
        preset_args = [
            "--seeds",
            "1",
            "2",
            "3",
            "--n-genes",
            "500",
            "--n-batches",
            "1",
            "--treatment-depth-multiplier",
            "1.0",
            "2.0",
            "10.0",
            "--frac-signal",
            "0.0",
            "0.2",
            "--effect-sd",
            "0.5",
            "--offtarget-guide-frac",
            "0.0",
            "--offtarget-slope-sd",
            "0.0",
            "--nb-overdispersion",
            "0.0",
            "--no-include-batch-covariate",
            "--methods",
            "meta",
            "stouffer",
            "lmm",
            "qc",
            "--lmm-scope",
            "meta_or_het_fdr",
            "--lmm-max-genes-per-focal-var",
            "200",
            "--qq-plots",
        ]
    elif args.preset == "full":
        preset_args = [
            "--seeds",
            "1",
            "2",
            "3",
            "--n-genes",
            "500",
            "--guides-per-gene",
            "4",
            "--n-control",
            "12",
            "--n-treatment",
            "12",
            "--methods",
            "meta",
            "stouffer",
            "lmm",
            "qc",
            "--lmm-scope",
            "meta_or_het_fdr",
            "--lmm-max-genes-per-focal-var",
            "500",
            "--qq-plots",
        ]
    elif args.preset == "abundance":
        # Focused stress-test of abundance regimes (heavy tails + within-gene dominance) without exploding the grid.
        preset_args = [
            "--seeds",
            "1",
            "2",
            "--n-genes",
            "500",
            "--guides-per-gene",
            "4",
            "--n-control",
            "12",
            "--n-treatment",
            "12",
            "--no-include-batch-covariate",
            "--methods",
            "meta",
            "stouffer",
            "lmm",
            "qc",
            "--lmm-scope",
            "meta_or_het_fdr",
            "--lmm-max-genes-per-focal-var",
            "500",
            "--gene-lambda-family",
            "lognormal",
            "mixture_lognormal",
            "power_law",
            "--gene-lambda-log-sd",
            "0.5",
            "--gene-lambda-mix-pi-high",
            "0.05",
            "0.5",
            "--gene-lambda-mix-delta-log-mean",
            "2.5",
            "--gene-lambda-power-alpha",
            "1.5",
            "--guide-lambda-family",
            "lognormal_noise",
            "dirichlet_weights",
            "--guide-lambda-log-sd",
            "0.8",
            "--guide-lambda-dirichlet-alpha0",
            "0.2",
            "--qq-plots",
        ]

    grid_args = preset_args + list(args.grid_args or []) + list(unknown)
    if int(args.jobs) > 0 and "--jobs" not in [str(x) for x in grid_args]:
        grid_args.extend(["--jobs", str(int(args.jobs))])
    if args.grid_tsv is not None and grid_args:
        raise ValueError("cannot use --grid-args (or extra grid flags) together with --grid-tsv")

    out_dir = str(args.out_dir)
    current_git = _try_git_info()
    if bool(args.resume) and not bool(args.force_resume):
        prev_manifest_path = os.path.join(out_dir, "suite_manifest.json")
        if os.path.isfile(prev_manifest_path):
            try:
                prev = json.loads(Path(prev_manifest_path).read_text(encoding="utf-8"))
            except Exception as e:  # noqa: BLE001 - surface as a clear resume error
                raise ValueError(f"--resume requires a readable suite_manifest.json; failed to read: {prev_manifest_path!r}: {e}") from e
            prev_head = (prev.get("git") or {}).get("head")
            cur_head = current_git.get("head")
            if prev_head and cur_head and str(prev_head) != str(cur_head):
                raise ValueError(
                    "refusing to --resume a suite created at a different git HEAD "
                    f"(prev={prev_head} cur={cur_head}); use --force-resume to override"
                )
    if os.path.exists(out_dir) and (not _dir_is_empty(out_dir)) and (not bool(args.resume)):
        raise ValueError(f"refusing to write into non-empty --out-dir (use a fresh path, or pass --resume): {out_dir!r}")
    os.makedirs(out_dir, exist_ok=True)

    grid_out_dir = args.grid_out_dir or os.path.join(out_dir, "grid_runs")
    os.makedirs(grid_out_dir, exist_ok=True)

    manifest: dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "out_dir": out_dir,
        "grid_out_dir": grid_out_dir,
        "preset": args.preset,
        "preset_args": preset_args,
        "grid_args": grid_args,
        "resume": bool(args.resume),
        "invocation": {"argv": list(sys.argv), "python_executable": sys.executable, "cwd": os.getcwd()},
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "packages": _package_versions(
                [
                    "numpy",
                    "pandas",
                    "scipy",
                    "statsmodels",
                    "percent_max_diff",
                    "matplotlib",
                ]
            ),
        },
        "git": current_git,
        "commands": {},
        "paths": {},
    }

    grid_tsv = args.grid_tsv
    if grid_tsv is None:
        # For some presets, run multiple response-mode grids and concatenate.
        grid_variants: list[dict[str, object]] = []
        if args.preset in {"standard", "full", "abundance"}:
            forbidden = {"--response-mode", "--pmd-n-boot", "--normalization-mode", "--logratio-mode", "--n-reference-genes"}
            bad = [a for a in grid_args if str(a) in forbidden]
            if bad:
                raise ValueError(
                    "multi-variant presets manage response/normalization/logratio flags internally; "
                    f"remove these from --grid-args: {sorted(set(bad))}"
                )

        if args.preset == "standard":
            grid_variants = [
                {
                    "name": "log_counts",
                    "extra_args": [
                        "--response-mode",
                        "log_counts",
                        "--normalization-mode",
                        "none",
                        "libsize_to_mean",
                        "cpm",
                        "median_ratio",
                        "--logratio-mode",
                        "none",
                        "clr_all",
                        "alr_refset",
                        "--n-reference-genes",
                        "50",
                    ],
                },
                {
                    "name": "guide_zscore_log_counts",
                    "extra_args": [
                        "--response-mode",
                        "guide_zscore_log_counts",
                        "--normalization-mode",
                        "none",
                        "libsize_to_mean",
                        "cpm",
                        "median_ratio",
                        "--logratio-mode",
                        "none",
                        "clr_all",
                        "alr_refset",
                        "--n-reference-genes",
                        "50",
                    ],
                },
                {"name": "pmd_std_res", "extra_args": ["--response-mode", "pmd_std_res", "--pmd-n-boot", "50"]},
            ]
        elif args.preset == "full":
            processing: list[dict[str, object]] = [
                {
                    "name": "log_counts",
                    "extra_args": [
                        "--response-mode",
                        "log_counts",
                        "--normalization-mode",
                        "none",
                        "libsize_to_mean",
                        "cpm",
                        "median_ratio",
                        "--logratio-mode",
                        "none",
                        "clr_all",
                        "alr_refset",
                        "--n-reference-genes",
                        "50",
                    ],
                },
                {
                    "name": "guide_zscore_log_counts",
                    "extra_args": [
                        "--response-mode",
                        "guide_zscore_log_counts",
                        "--normalization-mode",
                        "none",
                        "libsize_to_mean",
                        "cpm",
                        "median_ratio",
                        "--logratio-mode",
                        "none",
                        "clr_all",
                        "alr_refset",
                        "--n-reference-genes",
                        "50",
                    ],
                },
                {"name": "pmd_std_res", "extra_args": ["--response-mode", "pmd_std_res", "--pmd-n-boot", "100"]},
            ]

            scenarios: list[dict[str, object]] = [
                {
                    "name": "null_depth",
                    "extra_args": [
                        "--n-batches",
                        "1",
                        "--treatment-depth-multiplier",
                        "1.0",
                        "2.0",
                        "10.0",
                        "--frac-signal",
                        "0.0",
                        "--effect-sd",
                        "0.5",
                        "--offtarget-guide-frac",
                        "0.0",
                        "--offtarget-slope-sd",
                        "0.0",
                        "--nb-overdispersion",
                        "0.0",
                    ],
                },
                {
                    "name": "signal_depth",
                    "extra_args": [
                        "--n-batches",
                        "1",
                        "--treatment-depth-multiplier",
                        "1.0",
                        "2.0",
                        "10.0",
                        "--frac-signal",
                        "0.2",
                        "--effect-sd",
                        "0.5",
                        "--offtarget-guide-frac",
                        "0.0",
                        "--offtarget-slope-sd",
                        "0.0",
                        "--nb-overdispersion",
                        "0.0",
                    ],
                },
                {
                    "name": "null_batch",
                    "extra_args": [
                        "--n-batches",
                        "2",
                        "--batch-confounding-strength",
                        "0.7",
                        "--batch-depth-log-sd",
                        "0.5",
                        "--treatment-depth-multiplier",
                        "1.0",
                        "--frac-signal",
                        "0.0",
                        "--effect-sd",
                        "0.5",
                        "--offtarget-guide-frac",
                        "0.0",
                        "--offtarget-slope-sd",
                        "0.0",
                        "--nb-overdispersion",
                        "0.0",
                    ],
                },
                {
                    "name": "signal_batch",
                    "extra_args": [
                        "--n-batches",
                        "2",
                        "--batch-confounding-strength",
                        "0.7",
                        "--batch-depth-log-sd",
                        "0.5",
                        "--treatment-depth-multiplier",
                        "1.0",
                        "--frac-signal",
                        "0.2",
                        "--effect-sd",
                        "0.5",
                        "--offtarget-guide-frac",
                        "0.0",
                        "--offtarget-slope-sd",
                        "0.0",
                        "--nb-overdispersion",
                        "0.0",
                    ],
                },
                {
                    "name": "signal_offtarget",
                    "extra_args": [
                        "--n-batches",
                        "1",
                        "--treatment-depth-multiplier",
                        "1.0",
                        "--frac-signal",
                        "0.2",
                        "--effect-sd",
                        "0.5",
                        "--offtarget-guide-frac",
                        "0.25",
                        "--offtarget-slope-sd",
                        "0.2",
                        "--nb-overdispersion",
                        "0.0",
                    ],
                },
                {
                    "name": "signal_overdispersed",
                    "extra_args": [
                        "--n-batches",
                        "1",
                        "--treatment-depth-multiplier",
                        "1.0",
                        "--frac-signal",
                        "0.2",
                        "--effect-sd",
                        "0.5",
                        "--offtarget-guide-frac",
                        "0.0",
                        "--offtarget-slope-sd",
                        "0.0",
                        "--nb-overdispersion",
                        "0.05",
                    ],
                },
            ]

            grid_variants = []
            for s in scenarios:
                for p in processing:
                    grid_variants.append(
                        {
                            "name": f"{s['name']}__{p['name']}",
                            "extra_args": [*p["extra_args"], *s["extra_args"]],
                        }
                    )
        elif args.preset == "abundance":
            processing: list[dict[str, object]] = [
                {
                    "name": "log_counts",
                    "extra_args": [
                        "--response-mode",
                        "log_counts",
                        "--normalization-mode",
                        "none",
                        "cpm",
                        "--logratio-mode",
                        "none",
                        "clr_all",
                        "--n-reference-genes",
                        "0",
                    ],
                },
                {
                    "name": "guide_zscore_log_counts",
                    "extra_args": [
                        "--response-mode",
                        "guide_zscore_log_counts",
                        "--normalization-mode",
                        "none",
                        "cpm",
                        "--logratio-mode",
                        "none",
                        "clr_all",
                        "--n-reference-genes",
                        "0",
                    ],
                },
                {"name": "pmd_std_res", "extra_args": ["--response-mode", "pmd_std_res", "--pmd-n-boot", "100"]},
            ]

            scenarios: list[dict[str, object]] = [
                {
                    "name": "null_depth",
                    "extra_args": [
                        "--n-batches",
                        "1",
                        "--treatment-depth-multiplier",
                        "1.0",
                        "10.0",
                        "--frac-signal",
                        "0.0",
                        "--effect-sd",
                        "0.5",
                        "--offtarget-guide-frac",
                        "0.0",
                        "--offtarget-slope-sd",
                        "0.0",
                        "--nb-overdispersion",
                        "0.0",
                    ],
                },
                {
                    "name": "signal_depth",
                    "extra_args": [
                        "--n-batches",
                        "1",
                        "--treatment-depth-multiplier",
                        "1.0",
                        "10.0",
                        "--frac-signal",
                        "0.2",
                        "--effect-sd",
                        "0.5",
                        "--offtarget-guide-frac",
                        "0.0",
                        "--offtarget-slope-sd",
                        "0.0",
                        "--nb-overdispersion",
                        "0.0",
                    ],
                },
            ]

            grid_variants = []
            for s in scenarios:
                for p in processing:
                    grid_variants.append(
                        {
                            "name": f"{s['name']}__{p['name']}",
                            "extra_args": [*p["extra_args"], *s["extra_args"]],
                        }
                    )

        if not grid_variants:
            grid_variants = [{"name": "grid", "extra_args": []}]

        grid_tsv_paths: list[str] = []
        for v in grid_variants:
            name = str(v["name"])
            variant_out_dir = os.path.join(grid_out_dir, f"rm={name}")
            os.makedirs(variant_out_dir, exist_ok=True)

            existing = os.path.join(variant_out_dir, "count_depth_grid_summary.tsv")

            cmd = [sys.executable, _script_path("run_count_depth_grid.py"), "--out-dir", variant_out_dir]
            if bool(args.resume):
                cmd.append("--resume")
            cmd.extend([str(x) for x in v.get("extra_args", [])])
            if grid_args:
                cmd.extend(grid_args)
            manifest["commands"][f"grid_{name}"] = cmd
            _run(cmd)
            if not os.path.isfile(existing):
                raise RuntimeError(f"grid runner finished but did not write expected TSV: {existing!r}")
            grid_tsv_paths.append(existing)

        if len(grid_tsv_paths) == 1:
            grid_tsv = grid_tsv_paths[0]
        else:
            combined = os.path.join(out_dir, "count_depth_grid_summary.tsv")
            dfs = [pd.read_csv(p, sep="\t") for p in grid_tsv_paths]
            pd.concat(dfs, axis=0, ignore_index=True, sort=False).to_csv(combined, sep="\t", index=False)
            grid_tsv = combined

        manifest["paths"]["grid_variant_tsvs"] = grid_tsv_paths
    manifest["paths"]["grid_tsv"] = grid_tsv

    agg_tsv = os.path.join(out_dir, "count_depth_grid_summary_agg.tsv")
    cmd = [
        sys.executable,
        _script_path("aggregate_count_depth_grid_summary.py"),
        "--grid-tsv",
        str(grid_tsv),
        "--out-tsv",
        agg_tsv,
    ]
    manifest["commands"]["aggregate"] = cmd
    _run(cmd)
    manifest["paths"]["grid_tsv_agg"] = agg_tsv

    fig_root = os.path.join(out_dir, "figures")
    os.makedirs(fig_root, exist_ok=True)

    fig_summary = os.path.join(fig_root, "summary")
    os.makedirs(fig_summary, exist_ok=True)
    cmd = [sys.executable, _script_path("plot_count_depth_grid_summary.py"), "--grid-tsv", grid_tsv, "--out-dir", fig_summary]
    manifest["commands"]["plot_summary"] = cmd
    _run(cmd)
    manifest["paths"]["figures_summary_dir"] = fig_summary

    fig_scorecards = os.path.join(fig_root, "scorecards")
    os.makedirs(fig_scorecards, exist_ok=True)
    cmd = [sys.executable, _script_path("plot_count_depth_scorecards.py"), "--grid-tsv", grid_tsv, "--out-dir", fig_scorecards]
    manifest["commands"]["plot_scorecards"] = cmd
    _run(cmd)
    manifest["paths"]["figures_scorecards_dir"] = fig_scorecards

    fig_confounding = os.path.join(fig_root, "confounding")
    os.makedirs(fig_confounding, exist_ok=True)
    cmd = [
        sys.executable,
        _script_path("plot_count_depth_confounding_diagnostics.py"),
        "--grid-tsv",
        grid_tsv,
        "--out-dir",
        fig_confounding,
    ]
    manifest["commands"]["plot_confounding"] = cmd
    _run(cmd)
    manifest["paths"]["figures_confounding_dir"] = fig_confounding

    fig_p_hist = os.path.join(fig_root, "p_hist")
    os.makedirs(fig_p_hist, exist_ok=True)
    cmd = [sys.executable, _script_path("plot_count_depth_p_histograms.py"), "--grid-tsv", grid_tsv, "--out-dir", fig_p_hist]
    manifest["commands"]["plot_p_hist"] = cmd
    _run(cmd)
    manifest["paths"]["figures_p_hist_dir"] = fig_p_hist

    if bool(args.heatmaps):
        fig_heatmaps = os.path.join(fig_root, "heatmaps")
        os.makedirs(fig_heatmaps, exist_ok=True)
        cmd = [sys.executable, _script_path("plot_count_depth_grid_heatmaps.py"), "--grid-tsv", grid_tsv, "--out-dir", fig_heatmaps]
        if bool(args.resume):
            cmd.append("--skip-existing")
        manifest["commands"]["plot_heatmaps"] = cmd
        _run(cmd)
        manifest["paths"]["figures_heatmaps_dir"] = fig_heatmaps

    manifest_path = os.path.join(out_dir, "suite_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(manifest_path)


if __name__ == "__main__":
    main()
