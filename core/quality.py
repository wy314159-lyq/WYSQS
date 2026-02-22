from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .sqs import SQSResult
from .structure import Structure


@dataclass
class SQSQuality:
    """Transparent quality report for a single SQS result."""

    grade: str
    score: float
    wmae: float
    wrmse: float
    p95: float
    max_delta: float
    shell1_wmae: float
    hard_failures: List[str]


def _normalize_sublattices(
    sublattices: Optional[Sequence[object]],
) -> List[Tuple[List[int], Dict[int, int]]]:
    """Normalize dataclass/dict-style sublattice specs to common tuples."""

    if not sublattices:
        return []

    normalized: List[Tuple[List[int], Dict[int, int]]] = []
    for sl in sublattices:
        if isinstance(sl, dict):
            sites = list(sl.get("sites", []))
            composition = dict(sl.get("composition", {}))
        else:
            sites = list(getattr(sl, "sites"))
            composition = dict(getattr(sl, "composition"))
        normalized.append((sites, composition))
    return normalized


def _grade_from_thresholds(
    wrmse: float,
    p95: float,
    max_delta: float,
    shell1_wmae: float,
) -> str:
    """Threshold-based qualitative grade."""

    if wrmse <= 0.020 and p95 <= 0.040 and max_delta <= 0.080 and shell1_wmae <= 0.030:
        return "A+"
    if wrmse <= 0.035 and p95 <= 0.070 and max_delta <= 0.150 and shell1_wmae <= 0.050:
        return "A"
    if wrmse <= 0.060 and p95 <= 0.120 and max_delta <= 0.250 and shell1_wmae <= 0.080:
        return "B"
    if wrmse <= 0.100 and p95 <= 0.200 and max_delta <= 0.350 and shell1_wmae <= 0.120:
        return "C"
    if wrmse <= 0.150 and p95 <= 0.300 and max_delta <= 0.500:
        return "D"
    return "E"


def evaluate_sqs_quality(
    result: SQSResult,
    base_structure: Optional[Structure] = None,
    sublattices: Optional[Sequence[object]] = None,
    target: Optional[np.ndarray] = None,
    shell_weights: Optional[Dict[int, float]] = None,
) -> SQSQuality:
    """
    Evaluate one SQS result.

    Score is computed from weighted SRO deviations (off-diagonal pairs only):
      weighted_MAE, weighted_RMSE, P95(|Delta|), max(|Delta|), shell-1 weighted_MAE.
    Hard failures (composition mismatch/frozen-site mutation) force grade F.
    """

    sro = np.asarray(
        result.sro_sublattice if result.sro_sublattice is not None else result.sro,
        dtype=np.float64,
    )
    if sro.ndim != 3 or sro.shape[1] != sro.shape[2]:
        raise ValueError("result.sro must have shape (S, K, K)")
    S, K, _ = sro.shape

    if target is None:
        target_arr = np.zeros_like(sro)
    else:
        target_arr = np.asarray(target, dtype=np.float64)
        if target_arr.shape != sro.shape:
            raise ValueError(f"target shape mismatch: expected {sro.shape}, got {target_arr.shape}")

    shell_w = np.ones(S, dtype=np.float64)
    if shell_weights:
        active_shells = sorted(shell_weights.keys())
        if len(active_shells) >= S:
            shell_w = np.array([float(shell_weights[s]) for s in active_shells[:S]], dtype=np.float64)

    offdiag_delta: List[float] = []
    offdiag_w: List[float] = []
    shell1_delta: List[float] = []
    all_upper_delta: List[float] = []

    for s in range(S):
        w_s = float(shell_w[s])
        for a in range(K):
            for b in range(a, K):
                d = abs(float(sro[s, a, b] - target_arr[s, a, b]))
                all_upper_delta.append(d)
                if a != b:
                    offdiag_delta.append(d)
                    offdiag_w.append(w_s)
                    if s == 0:
                        shell1_delta.append(d)

    if offdiag_delta:
        d = np.asarray(offdiag_delta, dtype=np.float64)
        w = np.asarray(offdiag_w, dtype=np.float64)
        wsum = float(np.sum(w))
        if wsum <= 0.0:
            w = np.ones_like(d)
            wsum = float(np.sum(w))
        wmae = float(np.sum(w * d) / wsum)
        wrmse = float(np.sqrt(np.sum(w * d * d) / wsum))
        p95 = float(np.quantile(d, 0.95))
        max_delta = float(np.max(d))
    else:
        # Unary systems: no off-diagonal terms. Fall back to all upper-triangle values.
        d = np.asarray(all_upper_delta, dtype=np.float64)
        wmae = float(np.mean(d)) if d.size else 0.0
        wrmse = float(np.sqrt(np.mean(d * d))) if d.size else 0.0
        p95 = float(np.quantile(d, 0.95)) if d.size else 0.0
        max_delta = float(np.max(d)) if d.size else 0.0

    shell1_wmae = float(np.mean(shell1_delta)) if shell1_delta else wmae

    hard_failures: List[str] = []
    normalized_sublattices = _normalize_sublattices(sublattices)

    if base_structure is not None and normalized_sublattices:
        active_sites: set[int] = set()
        for idx, (sites, composition) in enumerate(normalized_sublattices):
            active_sites.update(sites)
            current = Counter(result.species[i] for i in sites)
            required = Counter({int(z): int(c) for z, c in composition.items()})
            if current != required:
                hard_failures.append(
                    f"Sublattice {idx} composition mismatch (expected {dict(required)}, got {dict(current)})."
                )

        frozen_changes = 0
        for i, z in enumerate(base_structure.species):
            if i in active_sites:
                continue
            if int(result.species[i]) != int(z):
                frozen_changes += 1
        if frozen_changes > 0:
            hard_failures.append(
                f"{frozen_changes} frozen sites changed species unexpectedly."
            )

    if hard_failures:
        return SQSQuality(
            grade="F",
            score=0.0,
            wmae=wmae,
            wrmse=wrmse,
            p95=p95,
            max_delta=max_delta,
            shell1_wmae=shell1_wmae,
            hard_failures=hard_failures,
        )

    # Continuous score: transparent linear penalty from key deviation metrics.
    penalty = 250.0 * wrmse + 60.0 * max_delta + 30.0 * p95 + 40.0 * shell1_wmae
    score = max(0.0, min(100.0, 100.0 - penalty))
    grade = _grade_from_thresholds(wrmse, p95, max_delta, shell1_wmae)

    return SQSQuality(
        grade=grade,
        score=score,
        wmae=wmae,
        wrmse=wrmse,
        p95=p95,
        max_delta=max_delta,
        shell1_wmae=shell1_wmae,
        hard_failures=[],
    )
