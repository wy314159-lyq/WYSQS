# gui/params_panel.py
# Optimization parameters: iterations, shell weights, pair weights, tolerances.

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.structure import (
    Structure,
    detect_shells_histogram,
    detect_shells_naive,
    distance_matrix,
)
from core.elements import z_to_symbol


class ParamsPanel(QWidget):
    """
    Right panel: optimization parameters and shell/pair weight settings.

    Signals:
      params_changed() — any parameter changed
    """

    params_changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._structure: Optional[Structure] = None
        self._shell_radii: List[float] = []
        self._unique_z: List[int] = []
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        layout.addWidget(self._build_opt_group())
        layout.addWidget(self._build_shell_group())
        layout.addWidget(self._build_pair_group())
        layout.addStretch()

    def _build_opt_group(self) -> QGroupBox:
        gb = QGroupBox("Optimization")
        form = QFormLayout(gb)
        form.setSpacing(6)

        self._iter_spin = QSpinBox()
        self._iter_spin.setRange(1_000, 10_000_000)
        self._iter_spin.setValue(100_000)
        self._iter_spin.setSingleStep(10_000)
        self._iter_spin.setGroupSeparatorShown(True)
        form.addRow("Iterations:", self._iter_spin)

        self._keep_spin = QSpinBox()
        self._keep_spin.setRange(1, 200)
        self._keep_spin.setValue(10)
        form.addRow("Keep best N:", self._keep_spin)

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(
            ["Anneal (ATAT-like)", "Random (Shuffle)", "Systematic"]
        )
        form.addRow("Search:", self._mode_combo)

        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(-1, 2_147_483_647)
        self._seed_spin.setValue(-1)
        self._seed_spin.setSpecialValueText("Random")
        form.addRow("RNG seed:", self._seed_spin)

        self._threads_spin = QSpinBox()
        self._threads_spin.setRange(0, 256)
        self._threads_spin.setValue(0)
        self._threads_spin.setSpecialValueText("Auto")
        form.addRow("CPU threads:", self._threads_spin)

        self._anneal_t0_spin = QDoubleSpinBox()
        self._anneal_t0_spin.setRange(1e-6, 1e3)
        self._anneal_t0_spin.setDecimals(6)
        self._anneal_t0_spin.setValue(0.2)
        self._anneal_t0_spin.setSingleStep(0.01)
        form.addRow("Anneal T0:", self._anneal_t0_spin)

        self._anneal_t1_spin = QDoubleSpinBox()
        self._anneal_t1_spin.setRange(1e-8, 1e2)
        self._anneal_t1_spin.setDecimals(8)
        self._anneal_t1_spin.setValue(0.001)
        self._anneal_t1_spin.setSingleStep(0.0005)
        form.addRow("Anneal Tend:", self._anneal_t1_spin)

        self._triplet_weight_spin = QDoubleSpinBox()
        self._triplet_weight_spin.setRange(0.0, 10.0)
        self._triplet_weight_spin.setDecimals(4)
        self._triplet_weight_spin.setValue(0.2)
        self._triplet_weight_spin.setSingleStep(0.05)
        form.addRow("Triplet weight:", self._triplet_weight_spin)

        self._shape_opt_check = QCheckBox("Enable shape optimization")
        self._shape_opt_check.setChecked(True)
        form.addRow("", self._shape_opt_check)

        self._shape_candidates_spin = QSpinBox()
        self._shape_candidates_spin.setRange(1, 256)
        self._shape_candidates_spin.setValue(24)
        form.addRow("Shape candidates:", self._shape_candidates_spin)

        return gb

    def _build_shell_group(self) -> QGroupBox:
        gb = QGroupBox("Shell Settings")
        vbox = QVBoxLayout(gb)
        vbox.setSpacing(6)

        # Tolerances
        tol_row = QHBoxLayout()
        tol_row.addWidget(QLabel("atol:"))
        self._atol_spin = QDoubleSpinBox()
        self._atol_spin.setRange(1e-6, 1.0)
        self._atol_spin.setDecimals(5)
        self._atol_spin.setValue(0.001)
        self._atol_spin.setSingleStep(0.001)
        self._atol_spin.setFixedWidth(90)
        tol_row.addWidget(self._atol_spin)

        tol_row.addWidget(QLabel("rtol:"))
        self._rtol_spin = QDoubleSpinBox()
        self._rtol_spin.setRange(1e-9, 0.1)
        self._rtol_spin.setDecimals(7)
        self._rtol_spin.setValue(0.00001)
        self._rtol_spin.setSingleStep(0.00001)
        self._rtol_spin.setFixedWidth(100)
        tol_row.addWidget(self._rtol_spin)

        detect_btn = QPushButton("Re-detect")
        detect_btn.setFixedWidth(80)
        detect_btn.clicked.connect(self._redetect_shells)
        tol_row.addWidget(detect_btn)
        tol_row.addStretch()
        vbox.addLayout(tol_row)

        # Shell weights table
        self._shell_table = QTableWidget(0, 3)
        self._shell_table.setHorizontalHeaderLabels(["Shell", "Radius (Å)", "Weight"])
        self._shell_table.horizontalHeader().setStretchLastSection(True)
        self._shell_table.setColumnWidth(0, 50)
        self._shell_table.setColumnWidth(1, 90)
        self._shell_table.setMaximumHeight(180)
        vbox.addWidget(self._shell_table)

        note = QLabel("Default shell weights follow sqsgenerator: w_s = 1/s")
        note.setWordWrap(True)
        note.setStyleSheet("font-size: 10px; color: #888888;")
        vbox.addWidget(note)

        return gb

    def _build_pair_group(self) -> QGroupBox:
        gb = QGroupBox("Pair Weights (Advanced)")
        vbox = QVBoxLayout(gb)

        info = QLabel(
            "Override per-species-pair weights. Default: off-diagonal=1.0, diagonal=0.0.\n"
            "Populated automatically after structure is loaded."
        )
        info.setWordWrap(True)
        info.setStyleSheet("font-size: 10px; color: #888888;")
        vbox.addWidget(info)

        self._pair_table = QTableWidget(0, 4)
        self._pair_table.setHorizontalHeaderLabels(
            ["Shell", "Species A", "Species B", "Weight"]
        )
        self._pair_table.horizontalHeader().setStretchLastSection(True)
        self._pair_table.setMaximumHeight(160)
        vbox.addWidget(self._pair_table)

        return gb

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_structure(self, structure: Structure) -> None:
        """Called when a new supercell is ready. Detects shells and populates tables."""
        self._structure = structure
        self._redetect_shells()

    def set_species(self, unique_z: List[int]) -> None:
        """Update the pair weights table for the given species list."""
        self._unique_z = unique_z
        self._populate_pair_table()

    def _redetect_shells(self) -> None:
        if self._structure is None:
            return
        dist = distance_matrix(self._structure)
        try:
            # sqsgenerator default shell detection mode is histogram peak detection.
            self._shell_radii = detect_shells_histogram(dist)
        except Exception:
            try:
                self._shell_radii = detect_shells_naive(
                    dist,
                    self._atol_spin.value(),
                    self._rtol_spin.value(),
                )
            except Exception:
                return
        self._populate_shell_table()

    def _populate_shell_table(self) -> None:
        radii = self._shell_radii
        # radii[0] = 0.0 (self), radii[1..] = shell representative distances
        n_shells = len(radii) - 1  # number of real shells
        self._shell_table.setRowCount(n_shells)

        for s in range(1, n_shells + 1):
            row = s - 1
            # Shell index
            item_s = QTableWidgetItem(str(s))
            item_s.setFlags(Qt.ItemIsEnabled)
            item_s.setTextAlignment(Qt.AlignCenter)
            self._shell_table.setItem(row, 0, item_s)

            # Representative radius (midpoint between shell boundaries)
            r = radii[s]
            item_r = QTableWidgetItem(f"{r:.4f}")
            item_r.setFlags(Qt.ItemIsEnabled)
            item_r.setTextAlignment(Qt.AlignCenter)
            self._shell_table.setItem(row, 1, item_r)

            # sqsgenerator default shell weights: 1/s
            w_item = QTableWidgetItem(f"{1.0 / float(s):.6g}")
            w_item.setTextAlignment(Qt.AlignCenter)
            self._shell_table.setItem(row, 2, w_item)

    def _populate_pair_table(self) -> None:
        unique_z = self._unique_z
        n_shells = max(1, len(self._shell_radii) - 1)
        K = len(unique_z)

        rows = []
        for s in range(1, n_shells + 1):
            for a in range(K):
                for b in range(a, K):
                    rows.append((s, unique_z[a], unique_z[b]))

        self._pair_table.setRowCount(len(rows))
        for row_idx, (s, za, zb) in enumerate(rows):
            item_s = QTableWidgetItem(str(s))
            item_s.setFlags(Qt.ItemIsEnabled)
            item_s.setTextAlignment(Qt.AlignCenter)
            self._pair_table.setItem(row_idx, 0, item_s)

            sym_a = z_to_symbol(za)
            sym_b = z_to_symbol(zb)
            for col, txt in [(1, sym_a), (2, sym_b)]:
                it = QTableWidgetItem(txt)
                it.setFlags(Qt.ItemIsEnabled)
                it.setTextAlignment(Qt.AlignCenter)
                self._pair_table.setItem(row_idx, col, it)

            default_w = "0.0" if za == zb else "1.0"
            w_item = QTableWidgetItem(default_w)
            w_item.setTextAlignment(Qt.AlignCenter)
            self._pair_table.setItem(row_idx, 3, w_item)

    def get_config(self) -> dict:
        """
        Return all parameter values as a dict:
          iterations, keep, mode, seed, shell_weights, pair_weights_override,
          atol, rtol
        """
        iterations = self._iter_spin.value()
        keep = self._keep_spin.value()
        mode = self._mode_combo.currentText()
        seed_val = self._seed_spin.value()
        seed = None if seed_val == -1 else seed_val
        atol = self._atol_spin.value()
        rtol = self._rtol_spin.value()

        # Shell weights: {shell_index: weight}
        shell_weights: Dict[int, float] = {}
        for row in range(self._shell_table.rowCount()):
            s_item = self._shell_table.item(row, 0)
            w_item = self._shell_table.item(row, 2)
            if s_item and w_item:
                try:
                    s = int(s_item.text())
                    w = float(w_item.text())
                    if w > 0.0:
                        shell_weights[s] = w
                except ValueError:
                    pass

        # Pair weights override: list of (shell, za, zb, weight)
        pair_overrides: List[Tuple[int, int, int, float]] = []
        for row in range(self._pair_table.rowCount()):
            items = [self._pair_table.item(row, c) for c in range(4)]
            if all(items):
                try:
                    s = int(items[0].text())
                    za = self._unique_z[
                        [z_to_symbol(z) for z in self._unique_z].index(items[1].text())
                    ]
                    zb = self._unique_z[
                        [z_to_symbol(z) for z in self._unique_z].index(items[2].text())
                    ]
                    w = float(items[3].text())
                    pair_overrides.append((s, za, zb, w))
                except (ValueError, IndexError):
                    pass

        return {
            "iterations": iterations,
            "keep": keep,
            "mode": mode,
            "seed": seed,
            "shell_weights": shell_weights,
            "pair_overrides": pair_overrides,
            "atol": atol,
            "rtol": rtol,
            "shell_radii": list(self._shell_radii),
            "anneal_t0": self._anneal_t0_spin.value(),
            "anneal_t1": self._anneal_t1_spin.value(),
            "triplet_weight": self._triplet_weight_spin.value(),
            "enable_shape_opt": bool(self._shape_opt_check.isChecked()),
            "shape_candidates": self._shape_candidates_spin.value(),
            "num_threads": self._threads_spin.value(),
        }

    def get_shell_radii(self) -> List[float]:
        return list(self._shell_radii)
