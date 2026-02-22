# core/io.py
# CIF and POSCAR file readers/writers.
# Primary CIF parser uses gemmi; falls back to a minimal regex-based reader.

from __future__ import annotations

import math
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from .elements import symbol_to_z, z_to_symbol
from .structure import Structure


# ---------------------------------------------------------------------------
# POSCAR reader
# ---------------------------------------------------------------------------

def read_poscar(filepath: str) -> Structure:
    """
    Parse a VASP POSCAR / CONTCAR file (VASP 5 format with element symbols).

    File layout:
      Line 0 : comment
      Line 1 : universal scale factor (float)
      Lines 2-4 : lattice vectors a, b, c (3 floats each)
      Line 5 : element symbols (VASP5; absent in VASP4)
      Line 6 : element counts
      Line 7 : "Direct" or "Cartesian" (case-insensitive)
      Lines 8+ : atomic coordinates (first 3 floats per line)

    If element symbols are missing (VASP4), they are read from the comment line.
    """
    with open(filepath, "r", encoding="utf-8") as fh:
        raw = fh.read()
    return _parse_poscar_text(raw)


def _parse_poscar_text(text: str) -> Structure:
    lines = [l.rstrip() for l in text.splitlines()]
    # Remove blank lines but keep track of original indices
    non_blank = [(i, l) for i, l in enumerate(lines) if l.strip()]

    if len(non_blank) < 8:
        raise ValueError("POSCAR file has too few lines")

    comment = non_blank[0][1].strip()
    scale = float(non_blank[1][1].split()[0])
    if scale < 0:
        # Negative scale means volume in Å³ — not common, treat as positive
        scale = abs(scale)

    a_vec = list(map(float, non_blank[2][1].split()[:3]))
    b_vec = list(map(float, non_blank[3][1].split()[:3]))
    c_vec = list(map(float, non_blank[4][1].split()[:3]))
    lattice = scale * np.array([a_vec, b_vec, c_vec], dtype=np.float64)

    # Detect VASP4 vs VASP5: line 5 is symbols if it contains letters
    line5 = non_blank[5][1].strip()
    if re.match(r'^[A-Za-z]', line5):
        # VASP5: symbols on line 5, counts on line 6
        symbols = line5.split()
        counts = list(map(int, non_blank[6][1].split()))
        coord_line_idx = 7
    else:
        # VASP4: counts on line 5, try to get symbols from comment
        counts = list(map(int, line5.split()))
        symbols = _extract_symbols_from_comment(comment, len(counts))
        coord_line_idx = 6

    if len(symbols) != len(counts):
        raise ValueError(
            f"Number of element symbols ({len(symbols)}) does not match "
            f"number of count entries ({len(counts)})"
        )

    coord_type = non_blank[coord_line_idx][1].strip().lower()
    is_cartesian = coord_type.startswith('c') or coord_type.startswith('k')

    N = sum(counts)
    coord_start = coord_line_idx + 1
    if len(non_blank) < coord_start + N:
        raise ValueError(
            f"Expected {N} coordinate lines but found only "
            f"{len(non_blank) - coord_start}"
        )

    raw_coords = np.array([
        list(map(float, non_blank[coord_start + i][1].split()[:3]))
        for i in range(N)
    ], dtype=np.float64)

    if is_cartesian:
        # Convert Cartesian (already scaled by scale factor) to fractional
        raw_coords = raw_coords * scale  # apply scale to Cartesian
        frac_coords = raw_coords @ np.linalg.inv(lattice)
    else:
        frac_coords = raw_coords

    # Wrap to [0, 1)
    frac_coords = frac_coords % 1.0

    species: List[int] = []
    site_labels: List[str] = []
    site_groups: List[str] = []
    for sym, cnt in zip(symbols, counts):
        z = symbol_to_z(sym)
        for k in range(cnt):
            species.append(z)
            site_labels.append(f"{sym}{k + 1}")
            site_groups.append(sym)

    return Structure(
        lattice=lattice,
        frac_coords=frac_coords,
        species=species,
        site_labels=site_labels,
        site_groups=site_groups,
    )


def _extract_symbols_from_comment(comment: str, n_expected: int) -> List[str]:
    """Try to extract element symbols from a VASP4 comment line."""
    tokens = comment.split()
    symbols = []
    for tok in tokens:
        tok_clean = tok.strip().capitalize()
        try:
            symbol_to_z(tok_clean)
            symbols.append(tok_clean)
        except ValueError:
            pass
    if len(symbols) == n_expected:
        return symbols
    raise ValueError(
        f"VASP4 POSCAR: could not extract {n_expected} element symbols from "
        f"comment line: '{comment}'. Please use VASP5 format with symbols on line 6."
    )


# ---------------------------------------------------------------------------
# POSCAR writer
# ---------------------------------------------------------------------------

def write_poscar(
    structure: Structure,
    filepath: str,
    comment: str = "SQS result",
) -> None:
    """
    Write structure to VASP5 POSCAR format with Direct (fractional) coordinates.
    """
    lines = [comment, "1.0"]

    # Lattice vectors
    for row in structure.lattice:
        lines.append(f"  {row[0]:20.16f}  {row[1]:20.16f}  {row[2]:20.16f}")

    # Group species
    comp = structure.composition
    unique_z_sorted = sorted(comp.keys())
    symbols = [z_to_symbol(z) for z in unique_z_sorted]
    counts = [comp[z] for z in unique_z_sorted]

    lines.append("  " + "  ".join(symbols))
    lines.append("  " + "  ".join(str(c) for c in counts))
    lines.append("Direct")

    # Write coordinates grouped by species
    species_arr = np.array(structure.species)
    for z in unique_z_sorted:
        indices = np.where(species_arr == z)[0]
        for idx in indices:
            fc = structure.frac_coords[idx]
            lines.append(f"  {fc[0]:20.16f}  {fc[1]:20.16f}  {fc[2]:20.16f}")

    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CIF reader
# ---------------------------------------------------------------------------

def read_cif(filepath: str) -> Structure:
    """
    Parse a CIF file and return a Structure.

    Tries gemmi first (pip install gemmi); falls back to a minimal regex parser.
    """
    try:
        return _read_cif_gemmi(filepath)
    except ImportError:
        pass
    except Exception as e:
        # gemmi failed for some reason, try fallback
        pass
    return _read_cif_fallback(filepath)


def _read_cif_gemmi(filepath: str) -> Structure:
    """Parse CIF using the gemmi library with full symmetry expansion."""
    import gemmi  # type: ignore

    doc = gemmi.cif.read(filepath)
    block = doc.sole_block()
    small = gemmi.make_small_structure_from_block(block)

    cell = small.cell
    a, b, c = cell.a, cell.b, cell.c
    alpha = math.radians(cell.alpha)
    beta  = math.radians(cell.beta)
    gamma = math.radians(cell.gamma)
    lattice = _cell_params_to_lattice(a, b, c, alpha, beta, gamma)

    species: List[int] = []
    frac_list: List[List[float]] = []
    site_labels: List[str] = []
    site_groups: List[str] = []
    label_counts: Dict[str, int] = {}

    # get_all_unit_cell_sites() applies space-group symmetry operations to
    # each asymmetric-unit site and returns the full set of atoms in the
    # unit cell.  Each expanded site preserves the original .label from the
    # CIF _atom_site_label column.
    all_sites = small.get_all_unit_cell_sites()

    for site in all_sites:
        sym = site.element.name if site.element else site.type_symbol
        sym = _clean_symbol(sym)
        try:
            z = symbol_to_z(sym)
        except ValueError:
            continue
        species.append(z)
        frac_list.append([site.fract.x, site.fract.y, site.fract.z])
        # site.label traces back to the parent asymmetric-unit site
        group = site.label if site.label else sym
        site_groups.append(group)
        label_counts[sym] = label_counts.get(sym, 0) + 1
        site_labels.append(f"{sym}{label_counts[sym]}")

    if not species:
        raise ValueError("No atoms found in CIF file via gemmi")

    frac_coords = np.array(frac_list, dtype=np.float64) % 1.0
    return Structure(
        lattice=lattice,
        frac_coords=frac_coords,
        species=species,
        site_labels=site_labels,
        site_groups=site_groups,
    )


# ---------------------------------------------------------------------------
# Symmetry expansion helpers (for CIF fallback reader)
# ---------------------------------------------------------------------------

def _parse_symops(text: str) -> List[str]:
    """
    Parse symmetry operations from CIF text.

    Looks for ``_symmetry_equiv_pos_as_xyz`` or
    ``_space_group_symop_operation_xyz`` inside a ``loop_`` block and
    returns the list of operation strings (e.g. ``['x,y,z', '-x,-y,z', ...]``).
    Returns ``['x,y,z']`` (identity only) when nothing is found.
    """
    loop_blocks = re.split(r'\bloop_\b', text, flags=re.IGNORECASE)

    for block in loop_blocks:
        lines = block.strip().splitlines()
        header_keys: List[str] = []
        data_start = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('_'):
                header_keys.append(stripped.split()[0].lower())
                data_start = i + 1
            elif header_keys:
                data_start = i
                break

        symop_key = None
        for k in header_keys:
            if k in ('_symmetry_equiv_pos_as_xyz',
                      '_space_group_symop_operation_xyz'):
                symop_key = k
                break
        if symop_key is None:
            continue

        col_idx = {k: i for i, k in enumerate(header_keys)}
        op_col = col_idx[symop_key]
        ops: List[str] = []
        for line in lines[data_start:]:
            stripped = line.strip()
            if not stripped or stripped.startswith('_') or stripped.startswith('#'):
                continue
            if stripped.lower().startswith('loop_'):
                break
            # Symop strings may be quoted ('x,y,z') or unquoted
            # and may share the row with an index column.
            tokens = stripped.split()
            if len(tokens) <= op_col:
                continue
            raw = tokens[op_col].strip("'\"")
            if ',' in raw:
                ops.append(raw)
        if ops:
            return ops

    return ['x,y,z']


def _apply_symop(
    op_str: str, x: float, y: float, z: float,
) -> Tuple[float, float, float]:
    """
    Apply a CIF symmetry-operation string to fractional coordinates.

    Handles expressions like ``'-x+1/2, y+1/2, -z+1/2'``.
    """
    parts = [p.strip() for p in op_str.split(',')]
    if len(parts) != 3:
        raise ValueError(f"Invalid symop string: '{op_str}'")

    def _eval_component(expr: str) -> float:
        expr = expr.replace(' ', '')
        val = 0.0
        # Tokenise into sign + term pairs
        tokens: List[str] = re.findall(r'[+\-]?[^+\-]+', expr)
        for tok in tokens:
            tok = tok.strip()
            if not tok:
                continue
            if 'x' in tok:
                coeff = tok.replace('x', '') or '+'
                val += float(coeff + '1' if coeff in ('+', '-', '') else coeff) * x
            elif 'y' in tok:
                coeff = tok.replace('y', '') or '+'
                val += float(coeff + '1' if coeff in ('+', '-', '') else coeff) * y
            elif 'z' in tok:
                coeff = tok.replace('z', '') or '+'
                val += float(coeff + '1' if coeff in ('+', '-', '') else coeff) * z
            elif '/' in tok:
                num, den = tok.split('/')
                val += float(num) / float(den)
            else:
                val += float(tok)
        return val % 1.0

    return (_eval_component(parts[0]),
            _eval_component(parts[1]),
            _eval_component(parts[2]))


def _expand_symmetry(
    asym_sites: List[Tuple[int, float, float, float, str]],
    symops: List[str],
    tol: float = 0.01,
) -> Tuple[List[int], List[List[float]], List[str], List[str]]:
    """
    Apply all symmetry operations to asymmetric-unit sites and remove
    duplicates (positions within *tol* in fractional coordinates).

    Returns ``(species, frac_list, site_labels, site_groups)``.
    """
    species: List[int] = []
    frac_list: List[List[float]] = []
    site_labels: List[str] = []
    site_groups: List[str] = []
    label_counts: Dict[str, int] = {}

    # Collect all generated positions for duplicate checking
    all_positions: List[Tuple[float, float, float]] = []

    for z, fx, fy, fz, lbl in asym_sites:
        for op in symops:
            nx, ny, nz = _apply_symop(op, fx, fy, fz)
            # Check for duplicates
            is_dup = False
            for ex, ey, ez in all_positions:
                dx = abs(nx - ex) % 1.0
                dy = abs(ny - ey) % 1.0
                dz = abs(nz - ez) % 1.0
                dx = min(dx, 1.0 - dx)
                dy = min(dy, 1.0 - dy)
                dz = min(dz, 1.0 - dz)
                if dx < tol and dy < tol and dz < tol:
                    is_dup = True
                    break
            if is_dup:
                continue
            all_positions.append((nx, ny, nz))
            species.append(z)
            frac_list.append([nx, ny, nz])
            site_groups.append(lbl)
            sym = z_to_symbol(z)
            label_counts[sym] = label_counts.get(sym, 0) + 1
            site_labels.append(f"{sym}{label_counts[sym]}")

    return species, frac_list, site_labels, site_groups


def _read_cif_fallback(filepath: str) -> Structure:
    """
    Minimal regex-based CIF reader with symmetry expansion.
    Handles standard single-block CIF files with _cell_* and _atom_site_* data.
    """
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    # ---- Cell parameters ----
    def get_float(key: str) -> float:
        m = re.search(rf'{re.escape(key)}\s+([\d\.\-\+eE]+)', text)
        if not m:
            raise ValueError(f"CIF: missing key '{key}'")
        return float(m.group(1).split('(')[0])  # strip uncertainty e.g. 3.165(2)

    a     = get_float('_cell_length_a')
    b     = get_float('_cell_length_b')
    c     = get_float('_cell_length_c')
    alpha = math.radians(get_float('_cell_angle_alpha'))
    beta  = math.radians(get_float('_cell_angle_beta'))
    gamma = math.radians(get_float('_cell_angle_gamma'))
    lattice = _cell_params_to_lattice(a, b, c, alpha, beta, gamma)

    # ---- Symmetry operations ----
    symops = _parse_symops(text)

    # ---- Atom site loop ----
    # Find the loop_ block containing _atom_site_* keys
    loop_blocks = re.split(r'\bloop_\b', text, flags=re.IGNORECASE)

    # Collect asymmetric-unit sites: (Z, fx, fy, fz, label)
    asym_sites: List[Tuple[int, float, float, float, str]] = []

    for block in loop_blocks:
        lines = block.strip().splitlines()
        # Collect header keys (lines starting with _)
        header_keys = []
        data_start = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('_'):
                header_keys.append(stripped.split()[0].lower())
                data_start = i + 1
            elif header_keys:
                data_start = i
                break

        if not any('_atom_site' in k for k in header_keys):
            continue

        # Map key -> column index
        col = {k: i for i, k in enumerate(header_keys)}

        # Determine which columns hold what we need
        sym_col = (col.get('_atom_site_type_symbol')
                   or col.get('_atom_site_label')
                   or col.get('_atom_site_element_symbol'))
        label_col = col.get('_atom_site_label')
        x_col = col.get('_atom_site_fract_x')
        y_col = col.get('_atom_site_fract_y')
        z_col = col.get('_atom_site_fract_z')

        if sym_col is None or x_col is None or y_col is None or z_col is None:
            continue

        # Parse data rows
        for line in lines[data_start:]:
            stripped = line.strip()
            if not stripped or stripped.startswith('_') or stripped.startswith('#'):
                continue
            if stripped.lower().startswith('loop_'):
                break
            tokens = stripped.split()
            if len(tokens) <= max(sym_col, x_col, y_col, z_col):
                continue
            sym = _clean_symbol(tokens[sym_col])
            try:
                z = symbol_to_z(sym)
            except ValueError:
                continue
            try:
                fx = float(tokens[x_col].split('(')[0])
                fy = float(tokens[y_col].split('(')[0])
                fz = float(tokens[z_col].split('(')[0])
            except ValueError:
                continue
            lbl = tokens[label_col] if label_col is not None and label_col < len(tokens) else sym
            asym_sites.append((z, fx, fy, fz, lbl))

        if asym_sites:
            break  # found the atom site loop

    if not asym_sites:
        raise ValueError(
            "Could not parse atom sites from CIF file. "
            "Ensure the file contains _atom_site_fract_x/y/z and "
            "_atom_site_type_symbol columns."
        )

    # ---- Apply symmetry expansion ----
    species, frac_list, site_labels, site_groups = _expand_symmetry(asym_sites, symops)

    frac_coords = np.array(frac_list, dtype=np.float64) % 1.0
    return Structure(
        lattice=lattice,
        frac_coords=frac_coords,
        species=species,
        site_labels=site_labels,
        site_groups=site_groups,
    )


# ---------------------------------------------------------------------------
# CIF writer
# ---------------------------------------------------------------------------

def write_cif(structure: Structure, filepath: str) -> None:
    """
    Write structure to a minimal CIF file.
    Uses gemmi if available; otherwise writes a hand-crafted CIF.
    """
    try:
        _write_cif_gemmi(structure, filepath)
        return
    except ImportError:
        pass
    _write_cif_manual(structure, filepath)


def _write_cif_gemmi(structure: Structure, filepath: str) -> None:
    import gemmi  # type: ignore

    a, b, c, alpha, beta, gamma = structure.lattice_params
    doc = gemmi.cif.Document()
    block = doc.add_new_block("SQS")

    block.set_pair('_cell_length_a',    f'{a:.6f}')
    block.set_pair('_cell_length_b',    f'{b:.6f}')
    block.set_pair('_cell_length_c',    f'{c:.6f}')
    block.set_pair('_cell_angle_alpha', f'{alpha:.4f}')
    block.set_pair('_cell_angle_beta',  f'{beta:.4f}')
    block.set_pair('_cell_angle_gamma', f'{gamma:.4f}')
    block.set_pair('_symmetry_space_group_name_H-M', 'P 1')
    block.set_pair('_symmetry_Int_Tables_number', '1')

    loop = block.init_loop('_atom_site_', [
        'label', 'type_symbol',
        'fract_x', 'fract_y', 'fract_z'
    ])
    for i, (z, fc) in enumerate(zip(structure.species, structure.frac_coords)):
        sym = z_to_symbol(z)
        label = structure.site_labels[i] if i < len(structure.site_labels) else f"{sym}{i+1}"
        loop.add_row([label, sym,
                      f'{fc[0]:.6f}', f'{fc[1]:.6f}', f'{fc[2]:.6f}'])

    doc.write_file(filepath)


def _write_cif_manual(structure: Structure, filepath: str) -> None:
    a, b, c, alpha, beta, gamma = structure.lattice_params
    lines = [
        "data_SQS",
        f"_cell_length_a                  {a:.6f}",
        f"_cell_length_b                  {b:.6f}",
        f"_cell_length_c                  {c:.6f}",
        f"_cell_angle_alpha               {alpha:.4f}",
        f"_cell_angle_beta                {beta:.4f}",
        f"_cell_angle_gamma               {gamma:.4f}",
        "_symmetry_space_group_name_H-M  'P 1'",
        "_symmetry_Int_Tables_number     1",
        "",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
    ]
    for i, (z, fc) in enumerate(zip(structure.species, structure.frac_coords)):
        sym = z_to_symbol(z)
        label = structure.site_labels[i] if i < len(structure.site_labels) else f"{sym}{i+1}"
        lines.append(f"  {label:<8s}  {sym:<4s}  {fc[0]:.6f}  {fc[1]:.6f}  {fc[2]:.6f}")

    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cell_params_to_lattice(
    a: float, b: float, c: float,
    alpha: float, beta: float, gamma: float,
) -> np.ndarray:
    """
    Convert unit cell parameters to a 3×3 lattice matrix (rows = vectors).

    Convention (standard crystallographic):
      a_vec = [a, 0, 0]
      b_vec = [b*cos(gamma), b*sin(gamma), 0]
      c_vec = [cx, cy, cz]
        cx = c * cos(beta)
        cy = c * (cos(alpha) - cos(beta)*cos(gamma)) / sin(gamma)
        cz = sqrt(c^2 - cx^2 - cy^2)
    """
    cos_a = math.cos(alpha)
    cos_b = math.cos(beta)
    cos_g = math.cos(gamma)
    sin_g = math.sin(gamma)

    cx = c * cos_b
    cy = c * (cos_a - cos_b * cos_g) / sin_g
    cz_sq = c * c - cx * cx - cy * cy
    if cz_sq < 0:
        cz_sq = 0.0
    cz = math.sqrt(cz_sq)

    return np.array([
        [a,              0.0,  0.0],
        [b * cos_g, b * sin_g, 0.0],
        [cx,             cy,   cz ],
    ], dtype=np.float64)


def _clean_symbol(sym: str) -> str:
    """
    Clean an atom site symbol/label to a pure element symbol.
    E.g. 'Fe1', 'Fe2+', 'Fe3+', 'FE' -> 'Fe'
    """
    # Remove trailing digits, charges, and whitespace
    sym = sym.strip()
    # Remove oxidation state suffixes like 2+, 3-, etc.
    sym = re.sub(r'[\d\+\-]+$', '', sym)
    # Capitalize properly: first letter upper, rest lower
    if len(sym) >= 1:
        sym = sym[0].upper() + sym[1:].lower()
    return sym


def read_structure(filepath: str) -> Structure:
    """
    Auto-detect file format and read structure.
    Supports .cif, POSCAR, CONTCAR, .vasp files.
    """
    basename = os.path.basename(filepath).lower()
    if basename.endswith('.cif'):
        return read_cif(filepath)
    elif basename in ('poscar', 'contcar') or basename.endswith('.vasp'):
        return read_poscar(filepath)
    else:
        # Try POSCAR first, then CIF
        try:
            return read_poscar(filepath)
        except Exception:
            return read_cif(filepath)
