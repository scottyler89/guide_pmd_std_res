from __future__ import annotations

import numpy as np
import pandas as pd


def pipeline_label(row: pd.Series, *, method: str) -> str:
    """
    Canonical benchmark pipeline label (used across reporting/plots).

    Includes:
      - method family (meta / stouffer / lmm_lrt / lmm_wald)
      - response pipeline knobs (response/norm/logratio/depthcov/batchcov)
      - LMM selection knobs when applicable
    """
    rm = str(row.get("response_mode", ""))
    norm = str(row.get("normalization_mode", ""))
    lr = str(row.get("logratio_mode", ""))
    depth = str(row.get("depth_covariate_mode", ""))
    batch_cov = int(bool(row.get("include_batch_covariate", False)))

    parts = [str(method), f"resp={rm}", f"norm={norm}", f"lr={lr}", f"depthcov={depth}", f"batchcov={batch_cov}"]
    if str(method).startswith("lmm_"):
        scope = str(row.get("lmm_scope", ""))
        cap = row.get("lmm_max_genes_per_focal_var", None)
        cap_num = pd.to_numeric(pd.Series([cap]), errors="coerce").iloc[0]
        cap_s = "0" if (not np.isfinite(cap_num) or float(cap_num) <= 0.0) else str(int(cap_num))
        parts.append(f"scope={scope}")
        parts.append(f"lmm_cap={cap_s}")
    return " | ".join(parts)
