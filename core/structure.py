# core/structure.py
# Structure dataclass, distance matrix (PBC), shell detection, shell matrix,
# pairs list, prefactors, supercell expansion.
# All algorithms match sqsgenerator C++ source exactly.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .elements import z_to_symbol, z_to_radius_angstrom


# ---------------------------------------------------------------------------
# Structure dataclass
# ---------------------------------------------------------------------------

@dataclass
class Structure:
    """
    Crystal structure representation.

    lattice     : (3,3) float64 — rows are lattice vectors a, b, c in Angstrom
    frac_coords : (N,3) float64 — fractional coordinates in [0,1)
    species     : list of N atomic numbers (int)
    site_labels : list of N site label strings, e.g. "Fe1", "Al2"
    site_groups : list of N group identifiers — atoms from the same crystallographic
                  site (Wyckoff position) share the same group string.  When read
                  from a CIF the original ``_atom_site_label`` is used; for POSCAR
                  files the element symbol is used as fallback.
    pbc         : periodic boundary conditions (always True for crystals)
    """
    lattice: np.ndarray          # (3,3)
    frac_coords: np.ndarray      # (N,3)
    species: List[int]
    site_labels: List[str]
    site_groups: Optional[List[str]] = None
    pbc: Tuple[bool, bool, bool] = (True, True, True)

    def __post_init__(self):
        self.lattice = np.asarray(self.lattice, dtype=np.float64)
        self.frac_coords = np.asarray(self.frac_coords, dtype=np.float64)
        assert self.lattice.shape == (3, 3), "lattice must be (3,3)"
        assert self.frac_coords.ndim == 2 and self.frac_coords.shape[1] == 3
        assert len(self.species) == self.frac_coords.shape[0]
        assert len(self.site_labels) == len(self.species)
        if self.site_groups is None:
            self.site_groups = [z_to_symbol(z) for z in self.species]
        assert len(self.site_groups) == len(self.species)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_atoms(self) -> int:
        return len(self.species)

    @property
    def cart_coords(self) -> np.ndarray:
        """Cartesian coordinates: frac_coords @ lattice, shape (N,3)."""
        return self.frac_coords @ self.lattice

    @property
    def unique_species(self) -> List[int]:
        """Sorted list of unique atomic numbers present."""
        return sorted(set(self.species))

    @property
    def composition(self) -> Dict[int, int]:
        """Dict {Z: count} for all species."""
        comp: Dict[int, int] = {}
        for z in self.species:
            comp[z] = comp.get(z, 0) + 1
        return comp

    @property
    def formula(self) -> str:
        """Hill-order formula string, e.g. 'Fe27W27'."""
        comp = self.composition
        parts = []
        # Carbon first, then hydrogen, then alphabetical
        for z in sorted(comp.keys()):
            sym = z_to_symbol(z)
            cnt = comp[z]
            parts.append(f"{sym}{cnt}" if cnt > 1 else sym)
        return "".join(parts)

    @property
    def lattice_params(self) -> Tuple[float, float, float, float, float, float]:
        """Return (a, b, c, alpha, beta, gamma) in Angstrom and degrees."""
        a_vec = self.lattice[0]
        b_vec = self.lattice[1]
        c_vec = self.lattice[2]
        a = float(np.linalg.norm(a_vec))
        b = float(np.linalg.norm(b_vec))
        c = float(np.linalg.norm(c_vec))
        # angles in degrees
        cos_alpha = np.dot(b_vec, c_vec) / (b * c)
        cos_beta  = np.dot(a_vec, c_vec) / (a * c)
        cos_gamma = np.dot(a_vec, b_vec) / (a * b)
        alpha = math.degrees(math.acos(max(-1.0, min(1.0, cos_alpha))))
        beta  = math.degrees(math.acos(max(-1.0, min(1.0, cos_beta))))
        gamma = math.degrees(math.acos(max(-1.0, min(1.0, cos_gamma))))
        return (a, b, c, alpha, beta, gamma)

    @property
    def group_composition(self) -> Dict[str, Tuple[int, List[int]]]:
        """Dict {group_label: (atomic_number_Z, [site_indices])}."""
        groups: Dict[str, Tuple[int, List[int]]] = {}
        for i, grp in enumerate(self.site_groups):
            if grp not in groups:
                groups[grp] = (self.species[i], [])
            groups[grp][1].append(i)
        return groups

    def copy(self) -> "Structure":
        return Structure(
            lattice=self.lattice.copy(),
            frac_coords=self.frac_coords.copy(),
            species=list(self.species),
            site_labels=list(self.site_labels),
            site_groups=list(self.site_groups),
            pbc=self.pbc,
        )


# ---------------------------------------------------------------------------
# Distance matrix with periodic boundary conditions
# ---------------------------------------------------------------------------

def distance_matrix(structure: Structure) -> np.ndarray:
    """
    Compute (N,N) symmetric matrix of minimum-image distances in Angstrom.

    Algorithm (matches core/structure.h detail::distance_matrix):
      cart = frac_coords @ lattice                           # (N,3)
      For each of 27 image translations t in {-1,0,1}^3:
          diff[i,j] = cart[i] - (t_cart + cart[j])
          d[i,j]    = ||diff[i,j]||
      result[i,j] = min over 27 images

    For large structures (N > 150) uses a chunked loop to avoid OOM.
    """
    cart = structure.frac_coords @ structure.lattice   # (N,3)
    a = structure.lattice[0]
    b = structure.lattice[1]
    c = structure.lattice[2]

    # 27 Cartesian translation vectors
    offsets = np.array([
        u * a + v * b + w * c
        for u in (-1, 0, 1)
        for v in (-1, 0, 1)
        for w in (-1, 0, 1)
    ], dtype=np.float64)   # (27,3)

    N = structure.num_atoms

    if N <= 150:
        # Fully vectorized: diff shape (27, N, N, 3)
        # cart[i] - (offset[t] + cart[j])
        # = cart[None, :, None, :] - (offsets[:, None, None, :] + cart[None, None, :, :])
        diff = (cart[np.newaxis, :, np.newaxis, :]
                - (offsets[:, np.newaxis, np.newaxis, :] + cart[np.newaxis, np.newaxis, :, :]))
        # shape (27, N, N)
        d = np.linalg.norm(diff, axis=-1)
        dist = d.min(axis=0)
    else:
        # Chunked: process in blocks of CHUNK atoms to limit memory
        CHUNK = 64
        dist = np.full((N, N), np.inf, dtype=np.float64)
        for i_start in range(0, N, CHUNK):
            i_end = min(i_start + CHUNK, N)
            ci = cart[i_start:i_end]   # (chunk, 3)
            # diff shape (27, chunk, N, 3)
            diff = (ci[np.newaxis, :, np.newaxis, :]
                    - (offsets[:, np.newaxis, np.newaxis, :] + cart[np.newaxis, np.newaxis, :, :]))
            d = np.linalg.norm(diff, axis=-1)   # (27, chunk, N)
            dist[i_start:i_end, :] = d.min(axis=0)

    # Enforce exact symmetry and zero diagonal
    dist = np.minimum(dist, dist.T)
    np.fill_diagonal(dist, 0.0)
    return dist


# ---------------------------------------------------------------------------
# Tolerance helper (matches C++ helpers::is_close)
# ---------------------------------------------------------------------------

def _is_close(a: float, b: float, atol: float, rtol: float) -> bool:
    """Absolute + relative tolerance check: |a-b| <= atol + rtol*|b|."""
    return abs(a - b) <= atol + rtol * abs(b)


# ---------------------------------------------------------------------------
# Shell detection — naive method
# ---------------------------------------------------------------------------

def detect_shells_naive(
    dist_mat: np.ndarray,
    atol: float = 1e-3,
    rtol: float = 1e-5,
) -> List[float]:
    """
    Detect coordination shell boundary distances from a distance matrix.

    Algorithm (matches core/structure.h distances_naive):
      1. Flatten and sort all distances
      2. Remove values <= atol (self-distances and numerical noise)
      3. Merge consecutive values within tolerance into midpoint
      4. Prepend 0.0

    Returns list [0.0, d1, d2, ...] where d_i are representative shell radii.
    The shell boundaries are the midpoints between consecutive d_i values.
    """
    flat = np.sort(dist_mat.ravel())
    flat = flat[flat > atol]

    if len(flat) == 0:
        return [0.0]

    reduced: List[float] = [float(flat[0])]
    for d in flat[1:]:
        last = reduced[-1]
        if _is_close(d, last, atol, rtol):
            # merge: replace with midpoint
            reduced[-1] = 0.5 * (d + last)
        else:
            reduced.append(float(d))

    # Prepend 0.0 as the self-distance shell boundary
    reduced.insert(0, 0.0)
    return reduced


# ---------------------------------------------------------------------------
# Shell detection - histogram peak method (sqsgenerator default)
# ---------------------------------------------------------------------------

def detect_shells_histogram(
    dist_mat: np.ndarray,
    bin_width: float = 0.05,
    peak_isolation: float = 0.25,
) -> List[float]:
    """
    Detect shell radii from the histogram of pair distances.

    This follows core/structure.h distances_histogram:
      1) collect non-zero distances and sort
      2) build fixed-width bins
      3) keep isolated local peaks
      4) prepend 0.0 if missing
    """
    if bin_width <= 0.0:
        raise ValueError("bin_width must be > 0")
    if not (0.0 <= peak_isolation <= 1.0):
        raise ValueError("peak_isolation must be in [0, 1]")

    flat = dist_mat.ravel()
    # Match C++ filter: dist > 0 and !is_close(dist, 0.0) with helper defaults.
    nz = flat[(flat > 0.0) & (~np.isclose(flat, 0.0, atol=1e-5, rtol=1e-3))]
    if nz.size == 0:
        return [0.0]

    distances = np.sort(nz.astype(np.float64))
    min_dist = float(distances[0])
    max_dist = float(distances[-1])
    num_edges = int((max_dist - min_dist) / bin_width) + 2
    if num_edges < 10:
        raise ValueError(
            "Not enough histogram edges; increase shell span or decrease bin_width"
        )

    edges = [min_dist + i * bin_width for i in range(num_edges)]
    freqs: Dict[int, List[float]] = {0: []}
    index = 0
    bin_idx = 0

    # Reproduce the explicit bin-walk logic from the C++ implementation.
    while index < len(distances) and bin_idx < num_edges - 1:
        lower = edges[bin_idx]
        upper = edges[bin_idx + 1]
        value = float(distances[index])
        if lower <= value < upper:
            freqs[bin_idx].append(value)
            index += 1
        else:
            bin_idx += 1
            freqs.setdefault(bin_idx, [])

    def get_freq(i: int) -> List[float]:
        return freqs.get(i, [])

    shells: List[float] = []
    for i in range(num_edges + 1):
        prev_f = get_freq(i - 1)
        cur_f = get_freq(i)
        next_f = get_freq(i + 1)
        threshold = int((1.0 - peak_isolation) * float(len(cur_f)))
        if threshold > len(prev_f) and threshold > len(next_f) and cur_f:
            shells.append(max(cur_f))

    if not shells:
        raise ValueError("No shell peaks detected from histogram")

    if not _is_close(shells[0], 0.0, atol=1e-5, rtol=1e-3):
        shells.insert(0, 0.0)
    return shells


# ---------------------------------------------------------------------------
# Shell matrix construction
# ---------------------------------------------------------------------------

def _find_shell(d: float, radii: List[float], atol: float, rtol: float) -> int:
    """
    Find which shell index a distance d belongs to.

    Shell 0 = self (d ~ 0).
    Shell s (s >= 1) satisfies: radii[s] <= d < radii[s+1]
    with tolerance matching.

    Matches core/structure.h detail::shell_matrix find_shell lambda.
    """
    if _is_close(d, 0.0, atol, rtol):
        return 0
    n = len(radii)
    for s in range(n - 1):
        lb = radii[s]
        ub = radii[s + 1]
        in_lower = _is_close(d, lb, atol, rtol) or d > lb
        in_upper = _is_close(d, ub, atol, rtol) or ub > d
        if in_lower and in_upper:
            return s + 1
    return n  # beyond last known shell


def build_shell_matrix(
    dist_mat: np.ndarray,
    shell_radii: List[float],
    atol: float = 1e-3,
    rtol: float = 1e-5,
) -> np.ndarray:
    """
    Build integer (N,N) shell matrix.

    Entry (i,j) = shell index s such that shell_radii[s] <= dist(i,j) < shell_radii[s+1].
    Shell 0 = self-pairs (diagonal).

    Matches core/structure.h detail::shell_matrix.
    """
    N = dist_mat.shape[0]
    shells = np.zeros((N, N), dtype=np.int32)

    for i in range(N):
        for j in range(i + 1, N):
            s = _find_shell(float(dist_mat[i, j]), shell_radii, atol, rtol)
            shells[i, j] = s
            shells[j, i] = s
    # diagonal stays 0
    return shells


# ---------------------------------------------------------------------------
# Pairs list
# ---------------------------------------------------------------------------

def build_pairs(
    shell_mat: np.ndarray,
    shell_weights: Dict[int, float],
) -> np.ndarray:
    """
    Build array of atom pairs for active shells.

    Returns int32 array of shape (P, 3) with columns [i, j, packed_shell_idx].
      - Only pairs i < j whose shell index is in shell_weights
      - packed_shell_idx: 0-based index into sorted(shell_weights.keys())

    Matches core/structure.h structure::pairs.
    """
    active_shells = sorted(shell_weights.keys())
    shell_pack = {s: idx for idx, s in enumerate(active_shells)}
    N = shell_mat.shape[0]

    rows = []
    for i in range(N):
        for j in range(i + 1, N):
            s = int(shell_mat[i, j])
            if s in shell_pack:
                rows.append((i, j, shell_pack[s]))

    if not rows:
        return np.empty((0, 3), dtype=np.int32)
    return np.array(rows, dtype=np.int32)


# ---------------------------------------------------------------------------
# Prefactors
# ---------------------------------------------------------------------------

def compute_prefactors(
    shell_mat: np.ndarray,
    shell_weights: Dict[int, float],
    species: List[int],
) -> Tuple[List[int], np.ndarray]:
    """
    Compute normalization prefactors for SRO calculation.

    Returns (unique_z, prefactors) where:
      unique_z   : sorted list of unique atomic numbers (packed index -> Z)
      prefactors : float64 array of shape (S, K, K)

    Formula (matches core/structure.h detail::compute_prefactors):
      M_s = count(shell_mat == s) / N        # avg neighbors per atom in shell s
      x_a = count(species == a) / N          # concentration of species a
      prefactor[s, a, b] = 1 / (M_s * x_a * x_b * N)
      prefactor[s, b, a] = prefactor[s, a, b]   (symmetric)

    Species are packed to 0-indexed integers for array indexing.
    """
    N = len(species)
    unique_z = sorted(set(species))
    K = len(unique_z)
    z_to_idx = {z: i for i, z in enumerate(unique_z)}
    active_shells = sorted(shell_weights.keys())
    S = len(active_shells)

    species_arr = np.array([z_to_idx[z] for z in species], dtype=np.int32)
    flat_shells = shell_mat.ravel()

    prefactors = np.zeros((S, K, K), dtype=np.float64)

    for si, s in enumerate(active_shells):
        # M_s: total bonds in shell s divided by N
        M_s = float(np.sum(flat_shells == s)) / N
        if M_s == 0.0:
            continue
        for a in range(K):
            x_a = float(np.sum(species_arr == a)) / N
            if x_a == 0.0:
                continue
            for b_idx in range(a, K):
                x_b = float(np.sum(species_arr == b_idx)) / N
                if x_b == 0.0:
                    continue
                pf = 1.0 / (M_s * x_a * x_b * N)
                prefactors[si, a, b_idx] = pf
                prefactors[si, b_idx, a] = pf

    return unique_z, prefactors


def compute_prefactors_sublattice(
    shell_mat: np.ndarray,
    shell_weights: Dict[int, float],
    species: List[int],
    sublattice_sites: Optional[List[List[int]]] = None,
) -> Tuple[List[int], np.ndarray]:
    """
    Compute SRO prefactors using sublattice-local quantities.

    Only pairs where **both** atoms belong to the active sublattice are
    considered.  Concentrations and the coordination number are computed
    within the sublattice:

        M_s_sub = count(shell_mat[i,j]==s for i,j both in sublattice) / N_sub
        x_a    = count(species==a within sublattice) / N_sub
        pf     = 1 / (M_s_sub * x_a * x_b * N_sub)

    Falls back to :func:`compute_prefactors` when *sublattice_sites* is
    ``None`` or empty.
    """
    if not sublattice_sites:
        return compute_prefactors(shell_mat, shell_weights, species)

    active_indices: set[int] = set()
    for sites in sublattice_sites:
        active_indices.update(sites)

    if not active_indices:
        return compute_prefactors(shell_mat, shell_weights, species)

    N = len(species)
    unique_z = sorted(set(species))
    K = len(unique_z)
    z_to_idx = {z: i for i, z in enumerate(unique_z)}
    active_shells = sorted(shell_weights.keys())
    S = len(active_shells)

    # Sublattice-local species and counts
    active_list = sorted(active_indices)
    N_sub = len(active_list)
    sub_species_arr = np.array(
        [z_to_idx[species[i]] for i in active_list], dtype=np.int32
    )

    # Build a sub-shell-matrix: only rows/cols in the sublattice
    active_arr = np.array(active_list, dtype=np.int32)
    sub_shell_mat = shell_mat[np.ix_(active_arr, active_arr)]
    flat_sub = sub_shell_mat.ravel()

    prefactors = np.zeros((S, K, K), dtype=np.float64)

    for si, s in enumerate(active_shells):
        M_s_sub = float(np.sum(flat_sub == s)) / N_sub
        if M_s_sub == 0.0:
            continue
        for a in range(K):
            x_a = float(np.sum(sub_species_arr == a)) / N_sub
            if x_a == 0.0:
                continue
            for b_idx in range(a, K):
                x_b = float(np.sum(sub_species_arr == b_idx)) / N_sub
                if x_b == 0.0:
                    continue
                pf = 1.0 / (M_s_sub * x_a * x_b * N_sub)
                prefactors[si, a, b_idx] = pf
                prefactors[si, b_idx, a] = pf

    return unique_z, prefactors


# ---------------------------------------------------------------------------
# Supercell expansion
# ---------------------------------------------------------------------------

def make_supercell(structure: Structure, sa: int, sb: int, sc: int) -> Structure:
    """
    Expand structure into a sa × sb × sc supercell.

    Algorithm (matches core/structure.h structure::supercell):
      new_lattice = lattice @ diag([sa, sb, sc])
      iscale = diag([1/sa, 1/sb, 1/sc])
      scaled_frac = frac_coords @ iscale          # coords in supercell frame
      for i in range(sa), j in range(sb), k in range(sc):
          translation = [i/sa, j/sb, k/sc]
          append scaled_frac + translation

    All fractional coordinates remain in [0, 1).
    """
    if sa < 1 or sb < 1 or sc < 1:
        raise ValueError("Supercell dimensions must be >= 1")

    scale = np.diag([float(sa), float(sb), float(sc)])
    new_lattice = structure.lattice @ scale

    iscale = np.diag([1.0 / sa, 1.0 / sb, 1.0 / sc])
    scaled_frac = structure.frac_coords @ iscale   # (N, 3)

    new_coords_list = []
    new_species: List[int] = []
    new_labels: List[str] = []
    new_groups: List[str] = []

    for i in range(sa):
        for j in range(sb):
            for k in range(sc):
                t = np.array([i / sa, j / sb, k / sc], dtype=np.float64)
                new_coords_list.append(scaled_frac + t)
                new_species.extend(structure.species)
                new_labels.extend(structure.site_labels)
                new_groups.extend(structure.site_groups)

    new_frac = np.vstack(new_coords_list)
    # Wrap to [0, 1) to avoid floating-point drift
    new_frac = new_frac % 1.0

    return Structure(
        lattice=new_lattice,
        frac_coords=new_frac,
        species=new_species,
        site_labels=new_labels,
        site_groups=new_groups,
        pbc=structure.pbc,
    )
