from __future__ import annotations

import heapq
import math
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .structure import (
    Structure,
    build_pairs,
    build_shell_matrix,
    compute_prefactors,
    detect_shells_histogram,
    detect_shells_naive,
    distance_matrix,
)


@dataclass
class Sublattice:
    """A single sublattice definition used during optimization."""

    sites: List[int]
    composition: Dict[int, int]


@dataclass
class OptimizationConfig:
    """All parameters required for one SQS optimization run."""

    structure: Structure
    sublattices: List[Sublattice]
    shell_weights: Dict[int, float]
    pair_weights: Optional[np.ndarray] = None
    target: Optional[np.ndarray] = None
    shell_radii: Optional[List[float]] = None
    iterations: int = 100_000
    keep: int = 10
    atol: float = 1e-3
    rtol: float = 1e-5
    bin_width: float = 0.05
    peak_isolation: float = 0.25
    iteration_mode: str = "random"
    seed: Optional[int] = None


@dataclass
class SQSResult:
    """One accepted SQS candidate."""

    objective: float
    species: List[int]
    sro: np.ndarray
    unique_z: List[int]
    shell_radii: List[float]
    iteration: int
    sro_sublattice: Optional[np.ndarray] = None


def _expand_composition(composition: Dict[int, int]) -> List[int]:
    """Expand {Z: count} into a deterministic species list (sorted by Z)."""

    expanded: List[int] = []
    for z in sorted(composition.keys()):
        cnt = int(composition[z])
        if cnt < 0:
            raise ValueError(f"Negative composition count for Z={z}: {cnt}")
        expanded.extend([z] * cnt)
    return expanded


def _next_permutation(arr: np.ndarray) -> bool:
    """In-place lexicographic next_permutation for a 1D int array."""

    n = int(arr.size)
    if n < 2:
        return False

    i = n - 2
    while i >= 0 and int(arr[i]) >= int(arr[i + 1]):
        i -= 1
    if i < 0:
        return False

    j = n - 1
    while int(arr[j]) <= int(arr[i]):
        j -= 1

    arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1 :] = arr[i + 1 :][::-1]
    return True


def _num_multiset_permutations(composition: Dict[int, int]) -> int:
    """Return n! / prod(c_i!) for composition counts."""

    total = int(sum(composition.values()))
    num = math.factorial(total)
    den = 1
    for cnt in composition.values():
        den *= math.factorial(int(cnt))
    return num // den


# ---------------------------------------------------------------------------
# Bond counting / objective
# ---------------------------------------------------------------------------

def count_bonds(
    pairs: np.ndarray,
    species_packed: np.ndarray,
    num_shells: int,
    num_species: int,
) -> np.ndarray:
    """Count bonds per shell and species pair."""

    bonds = np.zeros((num_shells, num_species, num_species), dtype=np.int64)
    if len(pairs) == 0:
        return bonds
    ii = pairs[:, 0]
    jj = pairs[:, 1]
    ss = pairs[:, 2]
    ai = species_packed[ii]
    aj = species_packed[jj]
    np.add.at(bonds, (ss, ai, aj), 1)
    return bonds


def compute_sro_and_objective(
    bonds: np.ndarray,
    prefactors: np.ndarray,
    pair_weights: np.ndarray,
    target: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Compute Warren-Cowley SRO and objective value."""

    _, K, _ = bonds.shape
    bonds_f = bonds.astype(np.float64)
    bonds_sym = bonds_f + bonds_f.transpose(0, 2, 1)

    diag_mask = np.eye(K, dtype=bool)[np.newaxis, :, :]
    b_eff = np.where(diag_mask, bonds_f, bonds_sym)
    sro = 1.0 - b_eff * prefactors

    triu_mask = np.triu(np.ones((K, K), dtype=bool))[np.newaxis, :, :]
    objective = float(np.sum(pair_weights * np.abs(sro - target) * triu_mask))
    return sro, objective


def recompute_sro_sublattice(
    result: SQSResult,
    structure: "Structure",
    shell_weights: Dict[int, float],
    sublattice_sites: Optional[List[List[int]]] = None,
    atol: float = 1e-3,
    rtol: float = 1e-5,
    *,
    precomputed: Optional[Dict] = None,
) -> np.ndarray:
    """
    Recompute SRO using sublattice-local Warren-Cowley definition.

    Only pairs where **both** atoms belong to the active sublattice are
    counted.  Concentrations and coordination numbers are sublattice-local.

    When *precomputed* is supplied it should contain ``sub_pairs`` and
    ``prefactors`` to avoid redundant calculations across results.
    """
    from .structure import (
        build_pairs,
        build_shell_matrix,
        compute_prefactors_sublattice,
        distance_matrix,
    )

    unique_z = result.unique_z
    K = len(unique_z)
    z_to_idx = {z: i for i, z in enumerate(unique_z)}
    active_shells = sorted(shell_weights.keys())
    S = len(active_shells)

    if precomputed is not None:
        sub_pairs = precomputed["pairs"]
        prefactors = precomputed["prefactors"]
    else:
        dist = distance_matrix(structure)
        shell_radii = result.shell_radii
        shell_mat = build_shell_matrix(dist, shell_radii, atol, rtol)

        # Filter pairs to sublattice-only bonds
        all_pairs = build_pairs(shell_mat, shell_weights)
        active_set: set[int] = set()
        if sublattice_sites:
            for sites in sublattice_sites:
                active_set.update(sites)
        if active_set and len(all_pairs) > 0:
            mask = np.array(
                [(int(r[0]) in active_set and int(r[1]) in active_set)
                 for r in all_pairs], dtype=bool,
            )
            sub_pairs = all_pairs[mask]
        else:
            sub_pairs = all_pairs

        _, prefactors = compute_prefactors_sublattice(
            shell_mat, shell_weights, result.species, sublattice_sites,
        )

    species_packed = np.array([z_to_idx[z] for z in result.species], dtype=np.int32)
    bonds = count_bonds(sub_pairs, species_packed, S, K)

    diag_mask = np.eye(K, dtype=bool)[np.newaxis, :, :]
    bonds_f = bonds.astype(np.float64)
    bonds_sym = bonds_f + bonds_f.transpose(0, 2, 1)
    b_eff = np.where(diag_mask, bonds_f, bonds_sym)
    return 1.0 - b_eff * prefactors


def fisher_yates_shuffle(
    arr: np.ndarray,
    lo: int,
    hi: int,
    rng: np.random.Generator,
) -> None:
    """In-place Fisher-Yates shuffle of arr[lo:hi]."""

    n = hi - lo
    if n < 2:
        return
    for i in range(n - 1, 0, -1):
        j = int(rng.integers(0, i + 1))
        arr[lo + i], arr[lo + j] = arr[lo + j], arr[lo + i]


def fisher_yates_shuffle_fast(
    arr: np.ndarray,
    lo: int,
    hi: int,
    rng: np.random.Generator,
) -> None:
    """Fast in-place shuffle of arr[lo:hi] using NumPy RNG."""

    n = hi - lo
    if n < 2:
        return
    rng.shuffle(arr[lo:hi])


def nth_best_objective(results: List[SQSResult], n: int) -> float:
    """Return the n-th best objective among current results."""

    if len(results) < n:
        return float("inf")
    return heapq.nsmallest(n, (r.objective for r in results))[-1]


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class SQSOptimizer:
    """Monte Carlo / systematic SQS optimizer."""

    def __init__(self, config: OptimizationConfig):
        self.config = config

    def _validate_sublattices(self, cfg: OptimizationConfig) -> List[Sublattice]:
        st = cfg.structure
        N = st.num_atoms

        if not cfg.sublattices:
            raise ValueError("No active sublattices selected.")

        seen_sites: set[int] = set()
        validated: List[Sublattice] = []
        for idx, sl in enumerate(cfg.sublattices):
            if not sl.sites:
                raise ValueError(f"Sublattice {idx} has no sites.")

            sorted_sites = sorted(sl.sites)
            if len(sorted_sites) != len(set(sorted_sites)):
                raise ValueError(f"Sublattice {idx} contains duplicate site indices.")

            for site in sorted_sites:
                if site < 0 or site >= N:
                    raise ValueError(f"Sublattice {idx} has out-of-range site index {site}.")
                if site in seen_sites:
                    raise ValueError(
                        f"Sublattice site overlap detected at index {site}."
                    )
                seen_sites.add(site)

            total = int(sum(sl.composition.values()))
            if total != len(sorted_sites):
                raise ValueError(
                    f"Sublattice {idx} composition assigns {total} atoms to "
                    f"{len(sorted_sites)} sites."
                )

            validated.append(Sublattice(sites=sorted_sites, composition=dict(sl.composition)))

        return validated

    def run(
        self,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> List[SQSResult]:
        cfg = self.config
        structure = cfg.structure
        N = structure.num_atoms

        sublattices = self._validate_sublattices(cfg)

        # 1) Distances and shells
        dist = distance_matrix(structure)
        if cfg.shell_radii is not None and len(cfg.shell_radii) > 0:
            shell_radii = list(cfg.shell_radii)
        else:
            try:
                shell_radii = detect_shells_histogram(
                    dist,
                    bin_width=cfg.bin_width,
                    peak_isolation=cfg.peak_isolation,
                )
            except ValueError:
                # Keep a robust fallback for small cells where histogram mode is
                # under-resolved.
                shell_radii = detect_shells_naive(dist, cfg.atol, cfg.rtol)

        if shell_radii[0] != 0.0:
            shell_radii.insert(0, 0.0)

        shell_mat = build_shell_matrix(dist, shell_radii, cfg.atol, cfg.rtol)

        # 2) Active pairs from selected shells
        pairs = build_pairs(shell_mat, cfg.shell_weights)
        if len(pairs) == 0:
            raise ValueError(
                "No atom pairs found for selected shell weights. "
                "Check shell indices or shell detection settings."
            )

        # 3) Build deterministic final species used for concentration/prefactor
        final_species = list(structure.species)
        for sl in sublattices:
            expanded = _expand_composition(sl.composition)
            if len(expanded) != len(sl.sites):
                raise ValueError("Internal sublattice size mismatch.")
            for local_idx, site_idx in enumerate(sl.sites):
                final_species[site_idx] = expanded[local_idx]

        unique_z, prefactors = compute_prefactors(shell_mat, cfg.shell_weights, final_species)
        K = len(unique_z)
        z_to_idx = {z: i for i, z in enumerate(unique_z)}
        active_shells = sorted(cfg.shell_weights.keys())
        S = len(active_shells)

        # 4) Pair weights: default hollow matrix (diag=0, off-diag=1), then scale per shell
        if cfg.pair_weights is not None:
            pair_weights = np.asarray(cfg.pair_weights, dtype=np.float64).copy()
            if pair_weights.shape != (S, K, K):
                raise ValueError(
                    f"pair_weights shape must be {(S, K, K)}, got {pair_weights.shape}."
                )
        else:
            pair_weights = np.ones((S, K, K), dtype=np.float64)
            diag = np.arange(K)
            pair_weights[:, diag, diag] = 0.0

        if not np.allclose(pair_weights, pair_weights.transpose(0, 2, 1)):
            raise ValueError("pair_weights must be symmetric over species indices.")

        for si, s in enumerate(active_shells):
            pair_weights[si] *= float(cfg.shell_weights[s])

        # 5) Target
        if cfg.target is not None:
            target = np.asarray(cfg.target, dtype=np.float64).copy()
            if target.shape != (S, K, K):
                raise ValueError(f"target shape must be {(S, K, K)}, got {target.shape}.")
        else:
            target = np.zeros((S, K, K), dtype=np.float64)

        if not np.allclose(target, target.transpose(0, 2, 1)):
            raise ValueError("target must be symmetric over species indices.")

        # 6) Packed species and per-sublattice pools
        species_packed = np.array([z_to_idx[z] for z in final_species], dtype=np.int32)
        sublattice_buffers: List[Tuple[np.ndarray, np.ndarray]] = []
        for sl in sublattices:
            sites_arr = np.array(sl.sites, dtype=np.int32)
            expanded = _expand_composition(sl.composition)
            sub_packed = np.array([z_to_idx[z] for z in expanded], dtype=np.int32)
            species_packed[sites_arr] = sub_packed
            sublattice_buffers.append((sites_arr, sub_packed))

        # 7) Iteration mode and loop size
        mode = cfg.iteration_mode.strip().lower()
        if mode not in {"random", "systematic"}:
            raise ValueError("iteration_mode must be either 'random' or 'systematic'.")
        if mode == "systematic" and len(sublattice_buffers) != 1:
            raise ValueError("Systematic mode currently supports exactly one active sublattice.")

        total_iterations = int(cfg.iterations)
        if mode == "systematic":
            max_perm = _num_multiset_permutations(sublattices[0].composition)
            total_iterations = min(total_iterations, max_perm)

        if total_iterations <= 0:
            return []

        rng = np.random.default_rng(cfg.seed)

        # Precomputed masks / indices
        diag_mask = np.eye(K, dtype=bool)[np.newaxis, :, :]
        triu_mask = np.triu(np.ones((K, K), dtype=bool))[np.newaxis, :, :]
        pw_triu = pair_weights * triu_mask

        p_ii = pairs[:, 0]
        p_jj = pairs[:, 1]
        p_ss = pairs[:, 2]
        SKK = S * K * K

        # Results buffer
        results: List[SQSResult] = []
        result_scores: List[Tuple[float, float]] = []
        search_threshold = float("inf")
        best_obj = float("inf")
        keep = max(1, int(cfg.keep))
        max_stored = keep * 20
        report_interval = 5000

        off_diag_mask = ~np.eye(K, dtype=bool)
        completed_iterations = 0

        for iteration in range(total_iterations):
            if stop_event is not None and stop_event.is_set():
                break

            if mode == "random":
                for sites_arr, sub_packed in sublattice_buffers:
                    rng.shuffle(sub_packed)
                    species_packed[sites_arr] = sub_packed
            else:
                sites_arr, sub_packed = sublattice_buffers[0]
                species_packed[sites_arr] = sub_packed

            ai = species_packed[p_ii]
            aj = species_packed[p_jj]
            flat_idx = p_ss * (K * K) + ai * K + aj
            bonds = np.bincount(flat_idx, minlength=SKK).reshape(S, K, K)

            bonds_f = bonds.astype(np.float64)
            bonds_sym = bonds_f + bonds_f.transpose(0, 2, 1)
            b_eff = np.where(diag_mask, bonds_f, bonds_sym)
            sro = 1.0 - b_eff * prefactors
            obj = float(np.dot(pw_triu.ravel(), np.abs(sro - target).ravel()))

            sec = float(np.sum(np.abs(sro[:, off_diag_mask]))) if K > 1 else 0.0

            if obj <= search_threshold:
                full_species = [unique_z[int(p)] for p in species_packed]
                results.append(
                    SQSResult(
                        objective=obj,
                        species=full_species,
                        sro=sro.copy(),
                        unique_z=list(unique_z),
                        shell_radii=list(shell_radii),
                        iteration=iteration,
                    )
                )
                result_scores.append((obj, sec))
                if obj < best_obj:
                    best_obj = obj

                if len(results) >= max_stored:
                    paired = sorted(
                        zip(result_scores, results), key=lambda x: (x[0][0], x[0][1])
                    )
                    paired = paired[:keep]
                    result_scores = [p[0] for p in paired]
                    results = [p[1] for p in paired]
                    search_threshold = result_scores[-1][0]

            completed_iterations = iteration + 1

            if progress_callback is not None and iteration % report_interval == 0:
                progress_callback(iteration, total_iterations, best_obj)

            if mode == "systematic":
                _, sub_packed = sublattice_buffers[0]
                if not _next_permutation(sub_packed):
                    break

        if progress_callback is not None:
            progress_callback(completed_iterations, total_iterations, best_obj)

        paired = sorted(zip(result_scores, results), key=lambda x: (x[0][0], x[0][1]))

        seen: set[Tuple[int, ...]] = set()
        unique_results: List[SQSResult] = []
        for _, result in paired:
            key = tuple(result.species)
            if key in seen:
                continue
            seen.add(key)
            unique_results.append(result)

        return unique_results
