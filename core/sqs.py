from __future__ import annotations

import heapq
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .structure import (
    Structure,
    build_pairs,
    build_shell_matrix,
    compute_prefactors,
    compute_prefactors_sublattice,
    detect_shells_histogram,
    detect_shells_naive,
    distance_matrix,
    make_supercell_hnf,
    rank_hnf_candidates,
)


@dataclass
class Sublattice:
    """A single sublattice definition used during optimization."""

    sites: List[int]
    composition: Dict[int, int]
    group_label: Optional[str] = None


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

    # ATAT-like additions
    search_mode: str = ""  # anneal, random, systematic; empty -> use iteration_mode
    anneal_start_temp: float = 0.2
    anneal_end_temp: float = 1e-3
    anneal_greedy_passes: int = 1
    triplet_weight: float = 0.2

    # Shape optimization (HNF candidates)
    enable_shape_optimization: bool = False
    primitive_structure: Optional[Structure] = None
    supercell_volume: Optional[int] = None
    supercell_dims: Optional[Tuple[int, int, int]] = None
    max_shape_candidates: int = 24
    num_threads: int = 0  # <=0 means auto (cpu count)


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

    pair_objective: float = 0.0
    triplet_objective: float = 0.0
    hnf_matrix: Optional[np.ndarray] = None
    shape_score: Optional[float] = None
    structure: Optional[Structure] = None
    sublattices: Optional[List[Sublattice]] = None


@dataclass
class _CandidateData:
    structure: Structure
    sublattices: List[Sublattice]
    shell_radii: List[float]
    unique_z: List[int]
    z_to_idx: Dict[int, int]
    pairs: np.ndarray
    prefactors: np.ndarray
    pair_weights: np.ndarray
    target: np.ndarray
    active_shells: List[int]
    species_initial: np.ndarray
    sublattice_buffers: List[Tuple[np.ndarray, np.ndarray]]
    triplets: np.ndarray
    triplet_type_weights: np.ndarray
    triplet_type_counts: np.ndarray
    triplet_probabilities: np.ndarray
    triplet_weight: float
    hnf_matrix: Optional[np.ndarray]
    shape_score: Optional[float]
    precomputed_sublattice: Optional[Dict[str, np.ndarray]]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


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


def nth_best_objective(results: List[SQSResult], n: int) -> float:
    """Return the n-th best objective among current results."""

    if len(results) < n:
        return float("inf")
    return heapq.nsmallest(n, (r.objective for r in results))[-1]


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
    """Compute Warren-Cowley SRO and pair objective value."""

    _, k, _ = bonds.shape
    bonds_f = bonds.astype(np.float64)
    bonds_sym = bonds_f + bonds_f.transpose(0, 2, 1)

    diag_mask = np.eye(k, dtype=bool)[np.newaxis, :, :]
    b_eff = np.where(diag_mask, bonds_f, bonds_sym)
    sro = 1.0 - b_eff * prefactors

    triu_mask = np.triu(np.ones((k, k), dtype=bool))[np.newaxis, :, :]
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
    precomputed: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """
    Recompute SRO using sublattice-local Warren-Cowley definition.

    Only pairs where both atoms belong to active sublattices are counted.
    """
    unique_z = result.unique_z
    k = len(unique_z)
    z_to_idx = {z: i for i, z in enumerate(unique_z)}
    active_shells = sorted(shell_weights.keys())
    s_count = len(active_shells)

    if precomputed is not None:
        sub_pairs = precomputed["pairs"]
        prefactors = precomputed["prefactors"]
    else:
        dist = distance_matrix(structure)
        shell_mat = build_shell_matrix(dist, result.shell_radii, atol, rtol)

        all_pairs = build_pairs(shell_mat, shell_weights)
        active_set: set[int] = set()
        if sublattice_sites:
            for sites in sublattice_sites:
                active_set.update(sites)
        if active_set and len(all_pairs) > 0:
            mask = np.array(
                [(int(r[0]) in active_set and int(r[1]) in active_set) for r in all_pairs],
                dtype=bool,
            )
            sub_pairs = all_pairs[mask]
        else:
            sub_pairs = all_pairs

        _, prefactors = compute_prefactors_sublattice(
            shell_mat,
            shell_weights,
            result.species,
            sublattice_sites,
        )

    species_packed = np.array([z_to_idx[z] for z in result.species], dtype=np.int32)
    bonds = count_bonds(sub_pairs, species_packed, s_count, k)

    diag_mask = np.eye(k, dtype=bool)[np.newaxis, :, :]
    bonds_f = bonds.astype(np.float64)
    bonds_sym = bonds_f + bonds_f.transpose(0, 2, 1)
    b_eff = np.where(diag_mask, bonds_f, bonds_sym)
    return 1.0 - b_eff * prefactors


# ---------------------------------------------------------------------------
# Triplet objective helpers
# ---------------------------------------------------------------------------


def _build_triplets(
    shell_mat: np.ndarray,
    shell_weights: Dict[int, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build triplet instances as i<j<k with all three edges in active shells.

    Returns:
      triplets: int32 (T,4) columns [i,j,k,type]
      type_weights: float64 (num_types,)
      type_counts: int64 (num_types,)
    """
    active_shells = sorted(shell_weights.keys())
    shell_pack = {s: idx for idx, s in enumerate(active_shells)}
    n = int(shell_mat.shape[0])

    upper_neighbors: List[List[Tuple[int, int]]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = int(shell_mat[i, j])
            if s in shell_pack:
                upper_neighbors[i].append((j, shell_pack[s]))

    type_map: Dict[Tuple[int, int, int], int] = {}
    type_signatures: List[Tuple[int, int, int]] = []
    rows: List[Tuple[int, int, int, int]] = []

    for i in range(n):
        neigh = upper_neighbors[i]
        if len(neigh) < 2:
            continue
        m = len(neigh)
        for p in range(m - 1):
            j, s_ij = neigh[p]
            for q in range(p + 1, m):
                k, s_ik = neigh[q]
                s_jk_raw = int(shell_mat[j, k])
                if s_jk_raw not in shell_pack:
                    continue
                s_jk = shell_pack[s_jk_raw]
                sig = tuple(sorted((s_ij, s_ik, s_jk)))
                t = type_map.get(sig)
                if t is None:
                    t = len(type_signatures)
                    type_map[sig] = t
                    type_signatures.append(sig)
                rows.append((i, j, k, t))

    if not rows:
        return (
            np.empty((0, 4), dtype=np.int32),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
        )

    triplets = np.array(rows, dtype=np.int32)
    num_types = len(type_signatures)
    type_counts = np.bincount(triplets[:, 3], minlength=num_types).astype(np.int64)

    type_weights = np.zeros(num_types, dtype=np.float64)
    for t, sig in enumerate(type_signatures):
        real_shells = [active_shells[s] for s in sig]
        w = [float(shell_weights[s]) for s in real_shells]
        type_weights[t] = float(np.mean(w))

    return triplets, type_weights, type_counts


def _triplet_probability_vector(concentrations: np.ndarray) -> np.ndarray:
    """Probability vector over sorted species triplets for random mixing."""
    k = int(concentrations.size)
    k3 = k * k * k
    probs = np.zeros(k3, dtype=np.float64)

    def idx(a: int, b: int, c: int) -> int:
        return (a * k + b) * k + c

    x = concentrations
    for a in range(k):
        probs[idx(a, a, a)] = x[a] ** 3

    for a in range(k):
        for b in range(a + 1, k):
            probs[idx(a, a, b)] = 3.0 * x[a] * x[a] * x[b]
            probs[idx(a, b, b)] = 3.0 * x[a] * x[b] * x[b]

    for a in range(k):
        for b in range(a + 1, k):
            for c in range(b + 1, k):
                probs[idx(a, b, c)] = 6.0 * x[a] * x[b] * x[c]

    return probs


def _count_triplets_sorted(
    triplets: np.ndarray,
    species_packed: np.ndarray,
    num_types: int,
    num_species: int,
) -> np.ndarray:
    """Count sorted species triplets per triplet type."""
    if len(triplets) == 0 or num_types == 0:
        return np.zeros((num_types, num_species ** 3), dtype=np.int64)

    i = triplets[:, 0]
    j = triplets[:, 1]
    k = triplets[:, 2]
    t = triplets[:, 3]

    a = species_packed[i]
    b = species_packed[j]
    c = species_packed[k]

    stacked = np.stack((a, b, c), axis=1)
    stacked.sort(axis=1)
    combo = (stacked[:, 0] * num_species + stacked[:, 1]) * num_species + stacked[:, 2]

    k3 = num_species ** 3
    flat = t.astype(np.int64) * k3 + combo.astype(np.int64)
    counts = np.bincount(flat, minlength=num_types * k3)
    return counts.reshape(num_types, k3)


def _triplet_objective(
    triplets: np.ndarray,
    species_packed: np.ndarray,
    type_weights: np.ndarray,
    type_counts: np.ndarray,
    probabilities: np.ndarray,
    num_species: int,
) -> float:
    if len(triplets) == 0 or type_weights.size == 0:
        return 0.0

    counts = _count_triplets_sorted(
        triplets,
        species_packed,
        int(type_weights.size),
        num_species,
    )
    obj = 0.0
    for t in range(type_weights.size):
        n_t = int(type_counts[t])
        if n_t <= 0:
            continue
        freq = counts[t].astype(np.float64) / float(n_t)
        obj += float(type_weights[t]) * float(np.sum(np.abs(freq - probabilities)))
    return obj


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


class SQSOptimizer:
    """Monte Carlo / systematic / annealing SQS optimizer."""

    def __init__(self, config: OptimizationConfig):
        self.config = config

    @staticmethod
    def _clone_sublattices(sublattices: Sequence[Sublattice]) -> List[Sublattice]:
        return [
            Sublattice(
                sites=list(sl.sites),
                composition=dict(sl.composition),
                group_label=sl.group_label,
            )
            for sl in sublattices
        ]

    def _validate_sublattices(
        self, structure: Structure, sublattices: Sequence[Sublattice]
    ) -> List[Sublattice]:
        n = structure.num_atoms
        if not sublattices:
            raise ValueError("No active sublattices selected.")

        seen_sites: set[int] = set()
        validated: List[Sublattice] = []

        for idx, sl in enumerate(sublattices):
            if not sl.sites:
                raise ValueError(f"Sublattice {idx} has no sites.")

            sorted_sites = sorted(int(x) for x in sl.sites)
            if len(sorted_sites) != len(set(sorted_sites)):
                raise ValueError(f"Sublattice {idx} contains duplicate site indices.")

            for site in sorted_sites:
                if site < 0 or site >= n:
                    raise ValueError(f"Sublattice {idx} has out-of-range site index {site}.")
                if site in seen_sites:
                    raise ValueError(f"Sublattice site overlap detected at index {site}.")
                seen_sites.add(site)

            total = int(sum(int(v) for v in sl.composition.values()))
            if total != len(sorted_sites):
                raise ValueError(
                    f"Sublattice {idx} composition assigns {total} atoms to "
                    f"{len(sorted_sites)} sites."
                )

            validated.append(
                Sublattice(
                    sites=sorted_sites,
                    composition={int(k): int(v) for k, v in sl.composition.items()},
                    group_label=sl.group_label,
                )
            )

        return validated

    def _validate_search_mode(self) -> str:
        cfg = self.config
        mode = (cfg.search_mode or "").strip().lower()
        if not mode:
            mode = (cfg.iteration_mode or "random").strip().lower()

        if mode.startswith("anneal"):
            mode = "anneal"
        elif mode.startswith("systematic"):
            mode = "systematic"
        elif mode.startswith("random"):
            mode = "random"

        if mode not in {"anneal", "random", "systematic"}:
            raise ValueError("search_mode must be one of: anneal, random, systematic.")
        return mode

    def _map_sublattices_to_structure(
        self,
        structure: Structure,
        template_sublattices: Sequence[Sublattice],
    ) -> List[Sublattice]:
        mapped: List[Sublattice] = []
        for sl in template_sublattices:
            if not sl.group_label:
                raise ValueError(
                    "Shape optimization requires group_label on each active sublattice."
                )
            sites = [i for i, g in enumerate(structure.site_groups) if g == sl.group_label]
            mapped.append(
                Sublattice(
                    sites=sites,
                    composition=dict(sl.composition),
                    group_label=sl.group_label,
                )
            )
        return self._validate_sublattices(structure, mapped)

    def _build_shape_candidates(
        self,
        base_sublattices: Sequence[Sublattice],
    ) -> List[Tuple[Structure, List[Sublattice], Optional[np.ndarray], Optional[float]]]:
        cfg = self.config

        if not cfg.enable_shape_optimization:
            return [
                (
                    cfg.structure,
                    self._validate_sublattices(cfg.structure, base_sublattices),
                    None,
                    None,
                )
            ]

        primitive = cfg.primitive_structure or cfg.structure
        if cfg.supercell_volume is not None:
            volume = int(cfg.supercell_volume)
        elif cfg.supercell_dims is not None:
            sa, sb, sc = cfg.supercell_dims
            volume = int(sa) * int(sb) * int(sc)
        else:
            if primitive.num_atoms <= 0:
                volume = 1
            else:
                volume = max(1, int(round(cfg.structure.num_atoms / primitive.num_atoms)))

        if volume < 1:
            raise ValueError("Supercell volume must be >= 1 for shape optimization.")

        ranked_hnfs = rank_hnf_candidates(
            primitive,
            volume,
            max_candidates=max(1, int(cfg.max_shape_candidates)),
            include_diagonal=cfg.supercell_dims,
        )

        candidates: List[Tuple[Structure, List[Sublattice], Optional[np.ndarray], Optional[float]]] = []
        for h in ranked_hnfs:
            st = make_supercell_hnf(primitive, h)
            sl = self._map_sublattices_to_structure(st, base_sublattices)
            shape_score = float(np.round(np.linalg.cond(st.lattice), 8))
            candidates.append((st, sl, h.copy(), shape_score))

        if not candidates:
            raise ValueError("No shape candidates generated for shape optimization.")

        return candidates

    def _prepare_candidate(
        self,
        structure: Structure,
        sublattices: Sequence[Sublattice],
        hnf_matrix: Optional[np.ndarray],
        shape_score: Optional[float],
    ) -> _CandidateData:
        cfg = self.config
        validated = self._validate_sublattices(structure, sublattices)

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
                shell_radii = detect_shells_naive(dist, cfg.atol, cfg.rtol)

        if shell_radii[0] != 0.0:
            shell_radii.insert(0, 0.0)

        shell_mat = build_shell_matrix(dist, shell_radii, cfg.atol, cfg.rtol)
        pairs = build_pairs(shell_mat, cfg.shell_weights)
        if len(pairs) == 0:
            raise ValueError(
                "No atom pairs found for selected shell weights. "
                "Check shell indices or shell detection settings."
            )

        final_species = list(structure.species)
        for sl in validated:
            expanded = _expand_composition(sl.composition)
            if len(expanded) != len(sl.sites):
                raise ValueError("Internal sublattice size mismatch.")
            for local_idx, site_idx in enumerate(sl.sites):
                final_species[site_idx] = expanded[local_idx]

        unique_z, prefactors = compute_prefactors(shell_mat, cfg.shell_weights, final_species)
        k = len(unique_z)
        s_count = len(cfg.shell_weights)
        z_to_idx = {z: i for i, z in enumerate(unique_z)}
        active_shells = sorted(cfg.shell_weights.keys())

        if cfg.pair_weights is not None:
            pair_weights = np.asarray(cfg.pair_weights, dtype=np.float64).copy()
            if pair_weights.shape != (s_count, k, k):
                raise ValueError(
                    f"pair_weights shape must be {(s_count, k, k)}, got {pair_weights.shape}."
                )
        else:
            pair_weights = np.ones((s_count, k, k), dtype=np.float64)
            diag = np.arange(k)
            pair_weights[:, diag, diag] = 0.0

        if not np.allclose(pair_weights, pair_weights.transpose(0, 2, 1)):
            raise ValueError("pair_weights must be symmetric over species indices.")

        for si, s in enumerate(active_shells):
            pair_weights[si] *= float(cfg.shell_weights[s])

        if cfg.target is not None:
            target = np.asarray(cfg.target, dtype=np.float64).copy()
            if target.shape != (s_count, k, k):
                raise ValueError(f"target shape must be {(s_count, k, k)}, got {target.shape}.")
        else:
            target = np.zeros((s_count, k, k), dtype=np.float64)

        if not np.allclose(target, target.transpose(0, 2, 1)):
            raise ValueError("target must be symmetric over species indices.")

        species_initial = np.array([z_to_idx[z] for z in final_species], dtype=np.int32)
        sublattice_buffers: List[Tuple[np.ndarray, np.ndarray]] = []
        for sl in validated:
            sites_arr = np.array(sl.sites, dtype=np.int32)
            expanded = _expand_composition(sl.composition)
            sub_packed = np.array([z_to_idx[z] for z in expanded], dtype=np.int32)
            species_initial[sites_arr] = sub_packed
            sublattice_buffers.append((sites_arr, sub_packed.copy()))

        triplet_weight = max(0.0, float(cfg.triplet_weight))
        triplets = np.empty((0, 4), dtype=np.int32)
        triplet_type_weights = np.empty((0,), dtype=np.float64)
        triplet_type_counts = np.empty((0,), dtype=np.int64)
        triplet_probabilities = np.empty((0,), dtype=np.float64)

        if triplet_weight > 0.0:
            triplets, triplet_type_weights, triplet_type_counts = _build_triplets(
                shell_mat,
                cfg.shell_weights,
            )
            conc = np.zeros(k, dtype=np.float64)
            for idx in species_initial:
                conc[int(idx)] += 1.0
            conc /= float(species_initial.size)
            triplet_probabilities = _triplet_probability_vector(conc)

        # Precompute sublattice-local SRO normalization for result display.
        precomputed_sublattice: Optional[Dict[str, np.ndarray]] = None
        if validated:
            sublattice_sites = [sl.sites for sl in validated]
            active_set: set[int] = set()
            for sites in sublattice_sites:
                active_set.update(sites)
            if active_set:
                mask = np.array(
                    [(int(r[0]) in active_set and int(r[1]) in active_set) for r in pairs],
                    dtype=bool,
                )
                sub_pairs = pairs[mask]
                _, pf_sub = compute_prefactors_sublattice(
                    shell_mat,
                    cfg.shell_weights,
                    final_species,
                    sublattice_sites,
                )
                precomputed_sublattice = {"pairs": sub_pairs, "prefactors": pf_sub}

        return _CandidateData(
            structure=structure,
            sublattices=self._clone_sublattices(validated),
            shell_radii=shell_radii,
            unique_z=list(unique_z),
            z_to_idx=z_to_idx,
            pairs=pairs,
            prefactors=prefactors,
            pair_weights=pair_weights,
            target=target,
            active_shells=active_shells,
            species_initial=species_initial,
            sublattice_buffers=sublattice_buffers,
            triplets=triplets,
            triplet_type_weights=triplet_type_weights,
            triplet_type_counts=triplet_type_counts,
            triplet_probabilities=triplet_probabilities,
            triplet_weight=triplet_weight,
            hnf_matrix=hnf_matrix,
            shape_score=shape_score,
            precomputed_sublattice=precomputed_sublattice,
        )

    def _evaluate_state(
        self,
        cand: _CandidateData,
        species_packed: np.ndarray,
    ) -> Tuple[np.ndarray, float, float, float]:
        s_count = len(cand.active_shells)
        k = len(cand.unique_z)
        bonds = count_bonds(cand.pairs, species_packed, s_count, k)
        sro, pair_obj = compute_sro_and_objective(
            bonds,
            cand.prefactors,
            cand.pair_weights,
            cand.target,
        )

        triplet_obj = 0.0
        if cand.triplet_weight > 0.0 and len(cand.triplets) > 0:
            triplet_obj = _triplet_objective(
                cand.triplets,
                species_packed,
                cand.triplet_type_weights,
                cand.triplet_type_counts,
                cand.triplet_probabilities,
                k,
            )

        total_obj = pair_obj + cand.triplet_weight * triplet_obj
        return sro, pair_obj, triplet_obj, total_obj

    def _add_result(
        self,
        results: List[SQSResult],
        score_pairs: List[Tuple[float, float]],
        keep: int,
        species_packed: np.ndarray,
        sro: np.ndarray,
        total_obj: float,
        pair_obj: float,
        triplet_obj: float,
        iteration: int,
        cand: _CandidateData,
    ) -> float:
        k = len(cand.unique_z)
        off_diag_mask = ~np.eye(k, dtype=bool)
        sec = float(np.sum(np.abs(sro[:, off_diag_mask]))) if k > 1 else 0.0

        species = [cand.unique_z[int(p)] for p in species_packed]
        results.append(
            SQSResult(
                objective=float(total_obj),
                species=species,
                sro=sro.copy(),
                unique_z=list(cand.unique_z),
                shell_radii=list(cand.shell_radii),
                iteration=int(iteration),
                pair_objective=float(pair_obj),
                triplet_objective=float(triplet_obj),
                hnf_matrix=cand.hnf_matrix.copy() if cand.hnf_matrix is not None else None,
                shape_score=cand.shape_score,
                structure=cand.structure,
                sublattices=self._clone_sublattices(cand.sublattices),
            )
        )
        score_pairs.append((float(total_obj), sec))

        max_stored = keep * 20
        threshold = float("inf")
        if len(results) >= max_stored:
            paired = sorted(zip(score_pairs, results), key=lambda x: (x[0][0], x[0][1]))
            paired = paired[:keep]
            score_pairs[:] = [p[0] for p in paired]
            results[:] = [p[1] for p in paired]
            threshold = score_pairs[-1][0]

        return threshold

    def _postprocess_candidate_results(
        self,
        results: List[SQSResult],
        score_pairs: List[Tuple[float, float]],
        keep: int,
        cand: _CandidateData,
    ) -> List[SQSResult]:
        if not results:
            return []

        paired = sorted(zip(score_pairs, results), key=lambda x: (x[0][0], x[0][1]))
        unique_results: List[SQSResult] = []
        seen: set[Tuple[int, ...]] = set()
        for _, result in paired:
            key = tuple(result.species)
            if key in seen:
                continue
            seen.add(key)
            unique_results.append(result)
            if len(unique_results) >= keep:
                break

        if cand.precomputed_sublattice is not None and cand.sublattices:
            sublattice_sites = [sl.sites for sl in cand.sublattices]
            for result in unique_results:
                result.sro_sublattice = recompute_sro_sublattice(
                    result,
                    cand.structure,
                    self.config.shell_weights,
                    sublattice_sites=sublattice_sites,
                    atol=self.config.atol,
                    rtol=self.config.rtol,
                    precomputed=cand.precomputed_sublattice,
                )

        return unique_results

    def _propose_swap(
        self,
        species_packed: np.ndarray,
        sublattice_buffers: Sequence[Tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator,
    ) -> Optional[Tuple[int, int]]:
        if not sublattice_buffers:
            return None

        for _ in range(24):
            sl_idx = int(rng.integers(0, len(sublattice_buffers)))
            sites = sublattice_buffers[sl_idx][0]
            n = int(sites.size)
            if n < 2:
                continue
            a = int(rng.integers(0, n))
            b = int(rng.integers(0, n - 1))
            if b >= a:
                b += 1
            i = int(sites[a])
            j = int(sites[b])
            if species_packed[i] != species_packed[j]:
                return i, j
        return None

    def _run_random_like(
        self,
        cand: _CandidateData,
        total_iterations: int,
        rng: np.random.Generator,
        keep: int,
        progress_base: int,
        progress_total: int,
        progress_callback: Optional[Callable[[int, int, float], None]],
        stop_event: Optional[threading.Event],
        global_best: List[float],
    ) -> List[SQSResult]:
        results: List[SQSResult] = []
        score_pairs: List[Tuple[float, float]] = []
        threshold = float("inf")

        species_packed = cand.species_initial.copy()
        report_interval = 2500

        for iteration in range(total_iterations):
            if stop_event is not None and stop_event.is_set():
                break

            for sites_arr, pool in cand.sublattice_buffers:
                rng.shuffle(pool)
                species_packed[sites_arr] = pool

            sro, pair_obj, triplet_obj, total_obj = self._evaluate_state(cand, species_packed)
            if total_obj <= threshold:
                threshold = self._add_result(
                    results,
                    score_pairs,
                    keep,
                    species_packed,
                    sro,
                    total_obj,
                    pair_obj,
                    triplet_obj,
                    progress_base + iteration,
                    cand,
                )
                if total_obj < global_best[0]:
                    global_best[0] = float(total_obj)

            if progress_callback is not None and (iteration % report_interval == 0):
                progress_callback(progress_base + iteration + 1, progress_total, global_best[0])

        if progress_callback is not None:
            progress_callback(progress_base + total_iterations, progress_total, global_best[0])

        return self._postprocess_candidate_results(results, score_pairs, keep, cand)

    def _run_systematic(
        self,
        cand: _CandidateData,
        total_iterations: int,
        keep: int,
        progress_base: int,
        progress_total: int,
        progress_callback: Optional[Callable[[int, int, float], None]],
        stop_event: Optional[threading.Event],
        global_best: List[float],
    ) -> List[SQSResult]:
        if len(cand.sublattice_buffers) != 1:
            raise ValueError("Systematic mode requires exactly one active sublattice.")

        results: List[SQSResult] = []
        score_pairs: List[Tuple[float, float]] = []
        threshold = float("inf")

        species_packed = cand.species_initial.copy()
        sites_arr, pool = cand.sublattice_buffers[0]
        pool.sort()

        max_perm = _num_multiset_permutations(cand.sublattices[0].composition)
        total = min(int(total_iterations), int(max_perm))
        if total <= 0:
            return []

        report_interval = 1000
        for iteration in range(total):
            if stop_event is not None and stop_event.is_set():
                break

            species_packed[sites_arr] = pool
            sro, pair_obj, triplet_obj, total_obj = self._evaluate_state(cand, species_packed)

            if total_obj <= threshold:
                threshold = self._add_result(
                    results,
                    score_pairs,
                    keep,
                    species_packed,
                    sro,
                    total_obj,
                    pair_obj,
                    triplet_obj,
                    progress_base + iteration,
                    cand,
                )
                if total_obj < global_best[0]:
                    global_best[0] = float(total_obj)

            if progress_callback is not None and (iteration % report_interval == 0):
                progress_callback(progress_base + iteration + 1, progress_total, global_best[0])

            if not _next_permutation(pool):
                break

        if progress_callback is not None:
            progress_callback(progress_base + total, progress_total, global_best[0])

        return self._postprocess_candidate_results(results, score_pairs, keep, cand)

    def _run_anneal(
        self,
        cand: _CandidateData,
        total_iterations: int,
        rng: np.random.Generator,
        keep: int,
        progress_base: int,
        progress_total: int,
        progress_callback: Optional[Callable[[int, int, float], None]],
        stop_event: Optional[threading.Event],
        global_best: List[float],
    ) -> List[SQSResult]:
        if total_iterations <= 0:
            return []

        results: List[SQSResult] = []
        score_pairs: List[Tuple[float, float]] = []
        threshold = float("inf")

        species_packed = cand.species_initial.copy()
        for sites_arr, pool in cand.sublattice_buffers:
            rng.shuffle(pool)
            species_packed[sites_arr] = pool

        sro, pair_obj, triplet_obj, current_obj = self._evaluate_state(cand, species_packed)
        threshold = self._add_result(
            results,
            score_pairs,
            keep,
            species_packed,
            sro,
            current_obj,
            pair_obj,
            triplet_obj,
            progress_base,
            cand,
        )
        if current_obj < global_best[0]:
            global_best[0] = float(current_obj)

        t0 = max(1e-12, float(self.config.anneal_start_temp))
        t1 = max(1e-12, float(self.config.anneal_end_temp))
        if t1 > t0:
            t0, t1 = t1, t0

        report_interval = 2000
        for iteration in range(1, total_iterations + 1):
            if stop_event is not None and stop_event.is_set():
                break

            frac = (iteration - 1) / max(total_iterations - 1, 1)
            temp = t0 * ((t1 / t0) ** frac)

            swap = self._propose_swap(species_packed, cand.sublattice_buffers, rng)
            if swap is not None:
                i, j = swap
                species_packed[i], species_packed[j] = species_packed[j], species_packed[i]
                sro_new, pair_new, triplet_new, obj_new = self._evaluate_state(cand, species_packed)

                delta = obj_new - current_obj
                if delta <= 0.0:
                    accept = True
                else:
                    x = -delta / max(temp, 1e-12)
                    x = max(-700.0, min(0.0, x))
                    accept = bool(rng.random() < math.exp(x))

                if accept:
                    sro, pair_obj, triplet_obj, current_obj = sro_new, pair_new, triplet_new, obj_new
                    if current_obj <= threshold:
                        threshold = self._add_result(
                            results,
                            score_pairs,
                            keep,
                            species_packed,
                            sro,
                            current_obj,
                            pair_obj,
                            triplet_obj,
                            progress_base + iteration,
                            cand,
                        )
                        if current_obj < global_best[0]:
                            global_best[0] = float(current_obj)
                else:
                    species_packed[i], species_packed[j] = species_packed[j], species_packed[i]

            if progress_callback is not None and (iteration % report_interval == 0):
                progress_callback(progress_base + iteration, progress_total, global_best[0])

        greedy_passes = max(0, int(self.config.anneal_greedy_passes))
        active_sites = int(sum(buf[0].size for buf in cand.sublattice_buffers))
        greedy_trials = max(1, active_sites)

        for g in range(greedy_passes):
            if stop_event is not None and stop_event.is_set():
                break
            for _ in range(greedy_trials):
                swap = self._propose_swap(species_packed, cand.sublattice_buffers, rng)
                if swap is None:
                    continue
                i, j = swap
                species_packed[i], species_packed[j] = species_packed[j], species_packed[i]
                sro_new, pair_new, triplet_new, obj_new = self._evaluate_state(cand, species_packed)
                if obj_new < current_obj:
                    sro, pair_obj, triplet_obj, current_obj = sro_new, pair_new, triplet_new, obj_new
                    if current_obj <= threshold:
                        threshold = self._add_result(
                            results,
                            score_pairs,
                            keep,
                            species_packed,
                            sro,
                            current_obj,
                            pair_obj,
                            triplet_obj,
                            progress_base + total_iterations + g,
                            cand,
                        )
                        if current_obj < global_best[0]:
                            global_best[0] = float(current_obj)
                else:
                    species_packed[i], species_packed[j] = species_packed[j], species_packed[i]

        if progress_callback is not None:
            progress_callback(progress_base + total_iterations, progress_total, global_best[0])

        return self._postprocess_candidate_results(results, score_pairs, keep, cand)

    def _make_candidate_seeds(self, num_candidates: int) -> List[int]:
        cfg = self.config
        if cfg.seed is None:
            ss = np.random.SeedSequence()
        else:
            ss = np.random.SeedSequence(int(cfg.seed))
        return [int(child.generate_state(1)[0]) for child in ss.spawn(num_candidates)]

    def _effective_num_threads(self, num_candidates: int) -> int:
        req = int(self.config.num_threads)
        if req <= 0:
            req = int(os.cpu_count() or 1)
        req = max(1, req)
        return min(req, max(1, num_candidates))

    def _run_candidate_task(
        self,
        candidate: Tuple[Structure, List[Sublattice], Optional[np.ndarray], Optional[float]],
        budget: int,
        keep: int,
        search_mode: str,
        seed: int,
        progress_base: int,
        progress_total: int,
        progress_callback: Optional[Callable[[int, int, float], None]],
        stop_event: Optional[threading.Event],
    ) -> List[SQSResult]:
        st, sublattices, hnf, shape_score = candidate
        cand = self._prepare_candidate(st, sublattices, hnf, shape_score)
        rng = np.random.default_rng(seed)
        local_best = [float("inf")]

        if search_mode == "anneal":
            return self._run_anneal(
                cand,
                budget,
                rng,
                keep,
                progress_base,
                progress_total,
                progress_callback,
                stop_event,
                local_best,
            )
        if search_mode == "systematic":
            return self._run_systematic(
                cand,
                budget,
                keep,
                progress_base,
                progress_total,
                progress_callback,
                stop_event,
                local_best,
            )
        return self._run_random_like(
            cand,
            budget,
            rng,
            keep,
            progress_base,
            progress_total,
            progress_callback,
            stop_event,
            local_best,
        )

    def run(
        self,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> List[SQSResult]:
        cfg = self.config
        search_mode = self._validate_search_mode()

        base_sublattices = self._validate_sublattices(cfg.structure, cfg.sublattices)
        candidates = self._build_shape_candidates(base_sublattices)

        if search_mode == "systematic" and len(candidates) > 1:
            # Systematic enumeration across many shapes is prohibitively expensive.
            candidates = candidates[:1]

        total_iterations = int(cfg.iterations)
        if total_iterations <= 0:
            return []

        num_candidates = len(candidates)
        per_candidate: List[int] = []
        base = total_iterations // num_candidates
        rem = total_iterations % num_candidates
        for i in range(num_candidates):
            per_candidate.append(base + (1 if i < rem else 0))
        per_candidate = [max(1, x) for x in per_candidate]

        total_budget = int(sum(per_candidate))
        progress_offsets: List[int] = []
        acc = 0
        for b in per_candidate:
            progress_offsets.append(acc)
            acc += b

        keep = max(1, int(cfg.keep))
        candidate_seeds = self._make_candidate_seeds(num_candidates)

        all_results: List[SQSResult] = []
        progress_done = 0
        best_obj = float("inf")

        num_threads = self._effective_num_threads(num_candidates)
        can_parallel = (
            num_threads > 1
            and num_candidates > 1
            and search_mode != "systematic"
        )

        if can_parallel:
            future_to_meta = {}
            with ThreadPoolExecutor(max_workers=num_threads) as ex:
                for idx, candidate in enumerate(candidates):
                    fut = ex.submit(
                        self._run_candidate_task,
                        candidate,
                        per_candidate[idx],
                        keep,
                        search_mode,
                        candidate_seeds[idx],
                        progress_offsets[idx],
                        total_budget,
                        None,  # avoid cross-thread GUI/progress callback race
                        stop_event,
                    )
                    future_to_meta[fut] = idx

                for fut in as_completed(future_to_meta):
                    idx = future_to_meta[fut]
                    candidate_results = fut.result()
                    all_results.extend(candidate_results)
                    progress_done += per_candidate[idx]
                    if candidate_results:
                        local_best = min(r.objective for r in candidate_results)
                        if local_best < best_obj:
                            best_obj = float(local_best)
                    if progress_callback is not None:
                        progress_callback(progress_done, total_budget, best_obj)
        else:
            for idx, candidate in enumerate(candidates):
                if stop_event is not None and stop_event.is_set():
                    break
                candidate_results = self._run_candidate_task(
                    candidate,
                    per_candidate[idx],
                    keep,
                    search_mode,
                    candidate_seeds[idx],
                    progress_offsets[idx],
                    total_budget,
                    progress_callback,
                    stop_event,
                )
                all_results.extend(candidate_results)
                progress_done += per_candidate[idx]
                if candidate_results:
                    local_best = min(r.objective for r in candidate_results)
                    if local_best < best_obj:
                        best_obj = float(local_best)

        if not all_results:
            if progress_callback is not None:
                progress_callback(progress_done, max(total_budget, 1), best_obj)
            return []

        # Global sorting and uniqueness: species + shape matrix.
        all_results.sort(key=lambda r: (r.objective, r.pair_objective, r.triplet_objective))
        uniq: List[SQSResult] = []
        seen: set[Tuple[Tuple[int, ...], Tuple[int, ...]]] = set()
        for r in all_results:
            if r.hnf_matrix is None:
                h_key = tuple()
            else:
                h_key = tuple(int(x) for x in np.asarray(r.hnf_matrix, dtype=np.int64).ravel())
            key = (tuple(int(z) for z in r.species), h_key)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(r)
            if len(uniq) >= keep:
                break

        if progress_callback is not None:
            progress_callback(total_budget, total_budget, uniq[0].objective if uniq else best_obj)

        return uniq
