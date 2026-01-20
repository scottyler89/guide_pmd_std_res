from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _unique_paths(paths: list[Path]) -> tuple[Path, ...]:
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        out.append(rp)
    return tuple(out)


def _grid_runs_suffix(path: Path) -> Path | None:
    parts = list(path.parts)
    if "grid_runs" not in parts:
        return None
    i = parts.index("grid_runs")
    return Path(*parts[i:])


@dataclass(frozen=True)
class ReportPathResolver:
    """
    Resolve report paths stored in suite-level TSVs (e.g. count_depth_grid_summary.tsv).

    Problem:
    - `report_path` is often recorded as a workspace-relative path (e.g. ".tmp/...").
    - Suites are frequently copied/relocated (e.g. moved under ".tmp/suites/...").
    - Consumers should be able to find per-run directories even after relocation.

    Strategy:
    - Try raw report_path under `cwd` and each ancestor of the grid TSV.
    - If still missing, try a suffix match anchored at "grid_runs/...".
    """

    grid_tsv_path: Path
    candidate_roots: tuple[Path, ...]

    @classmethod
    def from_grid_tsv(cls, grid_tsv: str | Path) -> "ReportPathResolver":
        grid_tsv_path = Path(grid_tsv).resolve()
        roots = [Path.cwd().resolve(), *grid_tsv_path.parents]
        return cls(grid_tsv_path=grid_tsv_path, candidate_roots=_unique_paths(roots))

    def resolve_report_path(self, report_path: str | Path) -> Path:
        rp = Path(report_path)
        if rp.is_absolute() and rp.is_file():
            return rp

        for root in self.candidate_roots:
            cand = root / rp
            if cand.is_file():
                return cand

        suffix = _grid_runs_suffix(rp) if (not rp.is_absolute()) else None
        if suffix is not None:
            for root in self.candidate_roots:
                cand = root / suffix
                if cand.is_file():
                    return cand

        raise FileNotFoundError(
            "could not resolve report_path relative to cwd or grid TSV ancestors "
            f"(grid_tsv={self.grid_tsv_path!s}, report_path={str(report_path)!r})"
        )

    def resolve_run_dir(self, report_path: str | Path) -> Path:
        return self.resolve_report_path(report_path).parent

