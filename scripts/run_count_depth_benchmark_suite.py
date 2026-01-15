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


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    return proc.stdout


def _run_and_last_line(cmd: list[str]) -> str:
    out = _run(cmd).strip().splitlines()
    if not out:
        raise RuntimeError(f"expected a path on stdout; got empty stdout: {' '.join(cmd)}")
    return out[-1].strip()


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
        choices=["quick", "standard", "full"],
        default=None,
        help="Optional named grid preset (applies only when the suite runs the grid).",
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
            "--no-qq-plots",
        ]
    elif args.preset == "standard":
        preset_args = [
            "--seeds",
            "1",
            "2",
            "3",
            "--n-genes",
            "500",
            "--frac-signal",
            "0.0",
            "0.2",
            "--effect-sd",
            "0.2",
            "0.5",
            "--offtarget-guide-frac",
            "0.0",
            "0.25",
            "--nb-overdispersion",
            "0.0",
            "0.05",
            "--methods",
            "meta",
            "stouffer",
            "lmm",
            "qc",
            "--lmm-scope",
            "meta_or_het_fdr",
            "--lmm-max-genes-per-focal-var",
            "200",
            "--no-qq-plots",
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
            "--depth-log-sd",
            "0.5",
            "1.0",
            "--treatment-depth-multiplier",
            "1.0",
            "2.0",
            "--n-batches",
            "1",
            "2",
            "--batch-confounding-strength",
            "0.0",
            "0.7",
            "--batch-depth-log-sd",
            "0.0",
            "0.5",
            "--frac-signal",
            "0.0",
            "0.2",
            "--effect-sd",
            "0.2",
            "0.5",
            "--offtarget-guide-frac",
            "0.0",
            "0.25",
            "--offtarget-slope-sd",
            "0.0",
            "0.2",
            "--nb-overdispersion",
            "0.0",
            "0.05",
            "--response-mode",
            "log_counts",
            "--normalization-mode",
            "none",
            "libsize_to_mean",
            "median_ratio",
            "--logratio-mode",
            "none",
            "clr_all",
            "--methods",
            "meta",
            "stouffer",
            "lmm",
            "qc",
            "--lmm-scope",
            "meta_or_het_fdr",
            "--lmm-max-genes-per-focal-var",
            "500",
            "--no-qq-plots",
        ]

    grid_args = preset_args + list(args.grid_args or []) + list(unknown)
    if args.grid_tsv is not None and grid_args:
        raise ValueError("cannot use --grid-args (or extra grid flags) together with --grid-tsv")

    out_dir = str(args.out_dir)
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
        "git": _try_git_info(),
        "commands": {},
        "paths": {},
    }

    grid_tsv = args.grid_tsv
    if grid_tsv is None:
        existing_grid_tsv = os.path.join(grid_out_dir, "count_depth_grid_summary.tsv")
        if bool(args.resume) and os.path.isfile(existing_grid_tsv):
            if grid_args:
                raise ValueError(
                    "cannot reuse an existing grid TSV while also passing grid args "
                    "(remove --grid-args/extra grid flags, or use a fresh --out-dir)"
                )
            grid_tsv = existing_grid_tsv
        else:
            cmd = [sys.executable, _script_path("run_count_depth_grid.py"), "--out-dir", grid_out_dir]
            if grid_args:
                cmd.extend(grid_args)
            manifest["commands"]["grid"] = cmd
            grid_tsv = _run_and_last_line(cmd)
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
        manifest["commands"]["plot_heatmaps"] = cmd
        _run(cmd)
        manifest["paths"]["figures_heatmaps_dir"] = fig_heatmaps

    manifest_path = os.path.join(out_dir, "suite_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(manifest_path)


if __name__ == "__main__":
    main()
