# gui/results_panel.py
# Results display: SRO parameter table (color-coded) + 3D viewer + export.

from __future__ import annotations

import csv
import os
from typing import List, Optional

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.elements import z_to_symbol
from core.io import write_cif, write_poscar
from core.quality import SQSQuality, evaluate_sqs_quality
from core.sqs import OptimizationConfig, SQSResult
from core.structure import Structure
from gui.viz3d import StructureViewer3D


class ResultsPanel(QWidget):
    """
    Results tab: list of results, SRO table, 3D viewer, export buttons.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._results: List[SQSResult] = []
        self._quality: List[SQSQuality] = []
        self._base_structure: Optional[Structure] = None
        self._config: Optional[OptimizationConfig] = None
        self._current_result: Optional[SQSResult] = None
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Horizontal)

        # Left: result list + SRO table + export
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        # Result list
        list_gb = QGroupBox("Results")
        list_vbox = QVBoxLayout(list_gb)
        self._result_list = QListWidget()
        self._result_list.currentRowChanged.connect(self._on_result_selected)
        list_vbox.addWidget(self._result_list)
        left_layout.addWidget(list_gb)

        # SRO table
        sro_gb = QGroupBox("Warren-Cowley SRO Parameters")
        sro_vbox = QVBoxLayout(sro_gb)

        legend = QLabel(
            "Color: "
            '<span style="background:#1a6b1a; color:white; padding:1px 4px;">|α| &lt; 0.05</span>  '
            '<span style="background:#7a6a00; color:white; padding:1px 4px;">|α| &lt; 0.15</span>  '
            '<span style="background:#7a1a1a; color:white; padding:1px 4px;">|α| ≥ 0.15</span>'
        )
        legend.setTextFormat(Qt.RichText)
        legend.setStyleSheet("font-size: 10px;")
        sro_vbox.addWidget(legend)

        self._quality_label = QLabel("Quality: -")
        self._quality_label.setWordWrap(True)
        self._quality_label.setStyleSheet("font-size: 10px; color: #c8c8c8;")
        sro_vbox.addWidget(self._quality_label)

        self._sro_table = QTableWidget(0, 6)
        self._sro_table.setHorizontalHeaderLabels(
            ["Shell", "Species i", "Species j", "α_ij", "Target", "|Δ|"]
        )
        self._sro_table.horizontalHeader().setStretchLastSection(True)
        self._sro_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._sro_table.setAlternatingRowColors(False)
        self._sro_table.setMinimumHeight(200)
        sro_vbox.addWidget(self._sro_table)
        left_layout.addWidget(sro_gb)

        # Export buttons
        export_gb = QGroupBox("Export")
        export_hbox = QHBoxLayout(export_gb)
        for label, slot in [
            ("POSCAR", self._export_poscar),
            ("CIF", self._export_cif),
            ("SRO CSV", self._export_sro_csv),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            export_hbox.addWidget(btn)
        export_hbox.addStretch()
        left_layout.addWidget(export_gb)

        splitter.addWidget(left)

        # Right: 3D viewer
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        viewer_gb = QGroupBox("3D Structure")
        viewer_vbox = QVBoxLayout(viewer_gb)
        self._viewer = StructureViewer3D()
        viewer_vbox.addWidget(self._viewer)
        right_layout.addWidget(viewer_gb)
        splitter.addWidget(right)

        splitter.setSizes([500, 500])
        layout.addWidget(splitter)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def display_results(
        self,
        results: List[SQSResult],
        base_structure: Structure,
        config: Optional[OptimizationConfig] = None,
    ) -> None:
        """Populate the panel with optimization results."""
        self._results = results
        self._quality = []
        self._base_structure = base_structure
        self._config = config

        shell_weights = config.shell_weights if config is not None else None
        target = config.target if config is not None else None
        sublattices = config.sublattices if config is not None else None

        self._result_list.clear()
        for i, r in enumerate(results):
            q = evaluate_sqs_quality(
                r,
                base_structure=base_structure,
                sublattices=sublattices,
                target=target,
                shell_weights=shell_weights,
            )
            self._quality.append(q)
            item = QListWidgetItem(
                f"#{i+1}  {q.grade} {q.score:5.1f}  obj={r.objective:.6f}  (iter {r.iteration:,})"
            )
            self._result_list.addItem(item)

        if results:
            self._result_list.setCurrentRow(0)

    def clear(self) -> None:
        self._results = []
        self._quality = []
        self._config = None
        self._result_list.clear()
        self._quality_label.setText("Quality: -")
        self._sro_table.setRowCount(0)
        self._viewer.clear()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_result_selected(self, row: int) -> None:
        if row < 0 or row >= len(self._results):
            return
        self._current_result = self._results[row]
        self._show_result(self._current_result, row)

    def _show_result(self, result: SQSResult, row: int) -> None:
        self._update_quality_label(row)
        self._populate_sro_table(result)
        self._update_3d_view(result)

    def _update_quality_label(self, row: int) -> None:
        if row < 0 or row >= len(self._quality):
            self._quality_label.setText("Quality: -")
            return
        q = self._quality[row]
        if q.hard_failures:
            fail_text = "; ".join(q.hard_failures)
            self._quality_label.setText(
                f"Quality: <b>F / 0.0</b>  Hard-fail: {fail_text}"
            )
            return
        self._quality_label.setText(
            "Quality: "
            f"<b>{q.grade} / {q.score:.1f}</b>  "
            f"wRMSE={q.wrmse:.4f}, wMAE={q.wmae:.4f}, "
            f"P95={q.p95:.4f}, max={q.max_delta:.4f}, "
            f"shell1-wMAE={q.shell1_wmae:.4f}"
        )

    def _populate_sro_table(self, result: SQSResult) -> None:
        # Prefer sublattice-aware SRO for display if available
        sro = result.sro_sublattice if result.sro_sublattice is not None else result.sro
        unique_z = result.unique_z
        K = len(unique_z)
        S = sro.shape[0]

        # Collect upper-triangle entries
        rows = []
        for s in range(S):
            for a in range(K):
                for b in range(a, K):
                    alpha = float(sro[s, a, b])
                    target = 0.0
                    delta = abs(alpha - target)
                    rows.append((s + 1, unique_z[a], unique_z[b], alpha, target, delta))

        self._sro_table.setRowCount(len(rows))
        for row_idx, (shell, za, zb, alpha, target, delta) in enumerate(rows):
            values = [
                str(shell),
                z_to_symbol(za),
                z_to_symbol(zb),
                f"{alpha:+.6f}",
                f"{target:.4f}",
                f"{delta:.6f}",
            ]
            for col, val in enumerate(values):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                # Color coding based on |alpha|
                if delta < 0.05:
                    bg = QColor("#1a4a1a")
                elif delta < 0.15:
                    bg = QColor("#4a3a00")
                else:
                    bg = QColor("#4a1a1a")
                item.setBackground(bg)
                item.setForeground(QColor("#e0e0e0"))
                self._sro_table.setItem(row_idx, col, item)

        self._sro_table.resizeColumnsToContents()

    def _update_3d_view(self, result: SQSResult) -> None:
        if self._base_structure is None:
            return
        # Build a structure with the result's species
        from core.structure import Structure
        st = self._base_structure
        result_structure = Structure(
            lattice=st.lattice.copy(),
            frac_coords=st.frac_coords.copy(),
            species=list(result.species),
            site_labels=list(st.site_labels),
            site_groups=list(st.site_groups),
        )
        self._viewer.plot_structure(result_structure)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _get_result_structure(self) -> Optional[Structure]:
        if self._current_result is None or self._base_structure is None:
            QMessageBox.warning(self, "No Result", "No result selected.")
            return None
        st = self._base_structure
        return Structure(
            lattice=st.lattice.copy(),
            frac_coords=st.frac_coords.copy(),
            species=list(self._current_result.species),
            site_labels=list(st.site_labels),
            site_groups=list(st.site_groups),
        )

    def _export_poscar(self) -> None:
        st = self._get_result_structure()
        if st is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export POSCAR", "POSCAR_SQS", "POSCAR files (POSCAR*);;All files (*)"
        )
        if not path:
            return
        try:
            write_poscar(st, path, comment="SQS result")
            QMessageBox.information(self, "Exported", f"POSCAR written to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _export_cif(self) -> None:
        st = self._get_result_structure()
        if st is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CIF", "sqs_result.cif", "CIF files (*.cif);;All files (*)"
        )
        if not path:
            return
        try:
            write_cif(st, path)
            QMessageBox.information(self, "Exported", f"CIF written to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _export_sro_csv(self) -> None:
        if self._current_result is None:
            QMessageBox.warning(self, "No Result", "No result selected.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export SRO CSV", "sro_parameters.csv", "CSV files (*.csv)"
        )
        if not path:
            return
        try:
            result = self._current_result
            sro = result.sro_sublattice if result.sro_sublattice is not None else result.sro
            unique_z = result.unique_z
            K = len(unique_z)
            S = sro.shape[0]
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["Shell", "Species_i", "Species_j",
                                  "alpha_ij", "Target", "Delta"])
                for s in range(S):
                    for a in range(K):
                        for b in range(a, K):
                            alpha = float(sro[s, a, b])
                            target = 0.0
                            writer.writerow([
                                s + 1,
                                z_to_symbol(unique_z[a]),
                                z_to_symbol(unique_z[b]),
                                f"{alpha:.8f}",
                                f"{target:.4f}",
                                f"{abs(alpha - target):.8f}",
                            ])
            QMessageBox.information(self, "Exported", f"SRO CSV written to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))
