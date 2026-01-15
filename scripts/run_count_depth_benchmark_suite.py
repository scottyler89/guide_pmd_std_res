from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from datetime import timezone


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local count-depth benchmark suite (grid → aggregate → figures).")
    parser.add_argument("--out-dir", required=True, type=str, help="Root output directory for the suite.")
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
            "Suite flags (e.g. --heatmaps) will not be forwarded."
        ),
    )
    parser.add_argument(
        "--heatmaps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, generate heatmap panels (can produce many files; default: disabled).",
    )
    args = parser.parse_args()

    out_dir = str(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    grid_out_dir = args.grid_out_dir or os.path.join(out_dir, "grid_runs")
    os.makedirs(grid_out_dir, exist_ok=True)

    manifest: dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "out_dir": out_dir,
        "grid_out_dir": grid_out_dir,
        "grid_args": args.grid_args or [],
        "paths": {},
    }

    grid_tsv = args.grid_tsv
    if grid_tsv is None:
        cmd = [sys.executable, _script_path("run_count_depth_grid.py"), "--out-dir", grid_out_dir]
        if args.grid_args:
            cmd.extend(args.grid_args)
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
    _run(cmd)
    manifest["paths"]["grid_tsv_agg"] = agg_tsv

    fig_root = os.path.join(out_dir, "figures")
    os.makedirs(fig_root, exist_ok=True)

    fig_summary = os.path.join(fig_root, "summary")
    os.makedirs(fig_summary, exist_ok=True)
    _run([sys.executable, _script_path("plot_count_depth_grid_summary.py"), "--grid-tsv", grid_tsv, "--out-dir", fig_summary])
    manifest["paths"]["figures_summary_dir"] = fig_summary

    fig_scorecards = os.path.join(fig_root, "scorecards")
    os.makedirs(fig_scorecards, exist_ok=True)
    _run([sys.executable, _script_path("plot_count_depth_scorecards.py"), "--grid-tsv", grid_tsv, "--out-dir", fig_scorecards])
    manifest["paths"]["figures_scorecards_dir"] = fig_scorecards

    fig_confounding = os.path.join(fig_root, "confounding")
    os.makedirs(fig_confounding, exist_ok=True)
    _run([sys.executable, _script_path("plot_count_depth_confounding_diagnostics.py"), "--grid-tsv", grid_tsv, "--out-dir", fig_confounding])
    manifest["paths"]["figures_confounding_dir"] = fig_confounding

    fig_p_hist = os.path.join(fig_root, "p_hist")
    os.makedirs(fig_p_hist, exist_ok=True)
    _run([sys.executable, _script_path("plot_count_depth_p_histograms.py"), "--grid-tsv", grid_tsv, "--out-dir", fig_p_hist])
    manifest["paths"]["figures_p_hist_dir"] = fig_p_hist

    if bool(args.heatmaps):
        fig_heatmaps = os.path.join(fig_root, "heatmaps")
        os.makedirs(fig_heatmaps, exist_ok=True)
        _run([sys.executable, _script_path("plot_count_depth_grid_heatmaps.py"), "--grid-tsv", grid_tsv, "--out-dir", fig_heatmaps])
        manifest["paths"]["figures_heatmaps_dir"] = fig_heatmaps

    manifest_path = os.path.join(out_dir, "suite_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(manifest_path)


if __name__ == "__main__":
    main()
