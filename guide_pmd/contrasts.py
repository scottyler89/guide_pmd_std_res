from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd


_NAME_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class ParsedContrast:
    name: str
    expression: str
    weights: dict[str, float]


def _replace_backticked_names(expr: str) -> tuple[str, dict[str, str]]:
    """
    Replace `...` substrings with safe identifiers so we can parse with `ast`.

    Returns:
      - transformed expression
      - placeholder -> original column name mapping
    """
    out: list[str] = []
    mapping: dict[str, str] = {}
    i = 0
    counter = 0
    while i < len(expr):
        if expr[i] != "`":
            out.append(expr[i])
            i += 1
            continue
        j = expr.find("`", i + 1)
        if j == -1:
            raise ValueError("unclosed backtick in contrast expression")
        name = expr[i + 1 : j]
        if name == "":
            raise ValueError("empty backticked name in contrast expression")
        placeholder = f"__c{counter}__"
        counter += 1
        mapping[placeholder] = name
        out.append(placeholder)
        i = j + 1
    return "".join(out), mapping


def _split_named_contrast(spec: str) -> tuple[str | None, str]:
    """
    Accept either:
      - "expr"
      - "NAME=expr"

    NAME must be a simple token (no spaces) and avoid '=' collisions.
    """
    spec = str(spec).strip()
    if not spec:
        raise ValueError("contrast expression must not be empty")
    if "=" not in spec:
        return None, spec
    left, right = spec.split("=", 1)
    if (not left) or (not right):
        raise ValueError("invalid named contrast syntax; expected NAME=expr")
    if (" " in left) or (not _NAME_RE.fullmatch(left.strip())):
        # Treat it as a literal expression with '=' which we do not support.
        raise ValueError("invalid contrast name (expected NAME=expr with NAME as a single token)")
    return left.strip(), right.strip()


def _merge_weights(a: dict[str, float], b: dict[str, float], *, scale_b: float = 1.0) -> dict[str, float]:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0.0) + float(scale_b) * float(v)
    # Drop exact zeros for readability/stability.
    out = {k: v for k, v in out.items() if v != 0.0}
    return out


def _parse_linear_ast(node: ast.AST, *, name_map: dict[str, str]) -> tuple[dict[str, float], float]:
    """
    Parse a restricted Python AST representing a linear expression:
      expr := constant | name | (+/- expr) | (expr +/- expr) | (const * expr) | (expr / const)

    Returns: (weights, constant)
    """
    if isinstance(node, ast.Name):
        term = str(node.id)
        term = name_map.get(term, term)
        return {term: 1.0}, 0.0
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and np.isfinite(float(node.value)):
            return {}, float(node.value)
        raise ValueError("only finite numeric constants are allowed in contrast expressions")
    if isinstance(node, ast.UnaryOp):
        weights, const = _parse_linear_ast(node.operand, name_map=name_map)
        if isinstance(node.op, ast.UAdd):
            return weights, const
        if isinstance(node.op, ast.USub):
            return {k: -v for k, v in weights.items()}, -const
        raise ValueError("unsupported unary operator in contrast expression")
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, (ast.Add, ast.Sub)):
            w1, c1 = _parse_linear_ast(node.left, name_map=name_map)
            w2, c2 = _parse_linear_ast(node.right, name_map=name_map)
            if isinstance(node.op, ast.Add):
                return _merge_weights(w1, w2, scale_b=1.0), c1 + c2
            return _merge_weights(w1, w2, scale_b=-1.0), c1 - c2

        if isinstance(node.op, ast.Mult):
            w1, c1 = _parse_linear_ast(node.left, name_map=name_map)
            w2, c2 = _parse_linear_ast(node.right, name_map=name_map)

            left_is_const = (len(w1) == 0)
            right_is_const = (len(w2) == 0)
            if left_is_const and right_is_const:
                return {}, c1 * c2
            if left_is_const and (not right_is_const):
                return {k: c1 * v for k, v in w2.items()}, c1 * c2
            if right_is_const and (not left_is_const):
                return {k: c2 * v for k, v in w1.items()}, c2 * c1
            raise ValueError("invalid product: contrast expressions must be linear (no name*name terms)")

        if isinstance(node.op, ast.Div):
            w1, c1 = _parse_linear_ast(node.left, name_map=name_map)
            w2, c2 = _parse_linear_ast(node.right, name_map=name_map)
            if len(w2) != 0:
                raise ValueError("invalid division: denominator must be a numeric constant")
            if c2 == 0.0:
                raise ValueError("invalid division by zero in contrast expression")
            scale = 1.0 / float(c2)
            return {k: scale * v for k, v in w1.items()}, scale * c1

        raise ValueError("unsupported binary operator in contrast expression")

    raise ValueError(f"unsupported syntax in contrast expression: {type(node).__name__}")


def parse_contrast_spec(spec: str) -> ParsedContrast:
    """
    Parse a linear contrast specification.

    Supported:
      - "C1_high - C1_low"
      - "name=C1_high - C1_low"
      - "`weird-col` - other_col"
      - "0.5 * (A - B)"
    """
    raw_name, expr = _split_named_contrast(spec)
    transformed, name_map = _replace_backticked_names(expr)

    try:
        tree = ast.parse(transformed, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid contrast expression: {expr}") from exc

    weights, const = _parse_linear_ast(tree.body, name_map=name_map)
    if const != 0.0:
        raise ValueError("contrast expressions must not include a non-zero constant term (use 'Intercept' instead)")
    if not weights:
        raise ValueError("contrast expression must reference at least one model term")

    name = raw_name if raw_name is not None else expr
    name = str(name).strip()
    if not name:
        raise ValueError("contrast name must not be empty")
    return ParsedContrast(name=name, expression=expr, weights=weights)


def build_contrast_matrix(contrasts: list[str], design_cols: list[str]) -> pd.DataFrame:
    """
    Build a contrast matrix L with shape (n_contrasts x n_params), aligned to design_cols.
    """
    if not contrasts:
        raise ValueError("contrasts must not be empty")
    if not design_cols:
        raise ValueError("design_cols must not be empty")

    design_cols = [str(c) for c in design_cols]
    design_set = set(design_cols)

    parsed = [parse_contrast_spec(c) for c in contrasts]
    names = [p.name for p in parsed]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"duplicate contrast name(s): {dupes}")

    rows = []
    for p in parsed:
        missing = [t for t in p.weights.keys() if t not in design_set]
        if missing:
            raise ValueError(f"contrast '{p.name}' references unknown term(s): {missing}")
        row = {col: 0.0 for col in design_cols}
        for term, w in p.weights.items():
            row[str(term)] = float(w)
        rows.append(row)

    out = pd.DataFrame(rows, index=names, columns=design_cols, dtype=float)
    out.index.name = "contrast"
    return out
