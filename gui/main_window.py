# gui/main_window.py
# Main application window: tabs, worker thread, progress bar.

from __future__ import annotations

import threading
from typing import Dict, List, Optional

import numpy as np
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.elements import z_to_symbol
from core.sqs import OptimizationConfig, SQSOptimizer, SQSResult, Sublattice
from core.structure import Structure
from gui.params_panel import ParamsPanel
from gui.results_panel import ResultsPanel
from gui.structure_panel import StructurePanel
from gui.viz3d import StructureViewer3D


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------

class SQSWorker(QThread):
    """
    Runs SQSOptimizer.run() in a background thread.
    Emits progress and finished/error signals back to the main thread.
    """

    progress = pyqtSignal(int, int, float)   # iteration, total, best_objective
    finished = pyqtSignal(list)              # list[SQSResult]
    error = pyqtSignal(str)

    def __init__(self, config: OptimizationConfig, parent=None):
        super().__init__(parent)
        self._config = config
        self._stop_event = threading.Event()

    def run(self) -> None:
        try:
            optimizer = SQSOptimizer(self._config)
            results = optimizer.run(
                progress_callback=self._on_progress,
                stop_event=self._stop_event,
            )
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))

    def stop(self) -> None:
        self._stop_event.set()

    def _on_progress(self, iteration: int, total: int, best_obj: float) -> None:
        self.progress.emit(iteration, total, best_obj)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """
    Top-level application window.

    Layout:
      Tab 0 "Setup"   — StructurePanel (left) + ParamsPanel (right) + Run/Stop bar
      Tab 1 "Results" — ResultsPanel (SRO table + 3D viewer)
    """

    def __init__(self):
        super().__init__()
        self._worker: Optional[SQSWorker] = None
        self._supercell: Optional[Structure] = None
        self._last_config: Optional[OptimizationConfig] = None
        self._setup_ui()
        self.setWindowTitle("SQS Generator")
        self.resize(1280, 800)

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # Tab widget
        self._tabs = QTabWidget()
        root.addWidget(self._tabs)

        # ---- Setup tab ----
        setup_widget = QWidget()
        setup_layout = QVBoxLayout(setup_widget)
        setup_layout.setContentsMargins(0, 0, 0, 0)
        setup_layout.setSpacing(6)

        panels_row = QHBoxLayout()
        panels_row.setSpacing(8)

        self._struct_panel = StructurePanel()
        self._struct_panel.setMinimumWidth(340)
        self._struct_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        panels_row.addWidget(self._struct_panel)

        # Preview viewer in setup tab
        self._preview_viewer = StructureViewer3D()
        self._preview_viewer.setMinimumWidth(300)
        self._preview_viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        panels_row.addWidget(self._preview_viewer)

        self._params_panel = ParamsPanel()
        self._params_panel.setMinimumWidth(320)
        self._params_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        panels_row.addWidget(self._params_panel)

        setup_layout.addLayout(panels_row)

        # Run bar
        run_bar = QHBoxLayout()
        run_bar.setSpacing(8)

        self._run_btn = QPushButton("▶  Run SQS")
        self._run_btn.setFixedHeight(36)
        font = QFont()
        font.setBold(True)
        font.setPointSize(11)
        self._run_btn.setFont(font)
        self._run_btn.setStyleSheet(
            "QPushButton { background: #2a6a2a; color: white; border-radius: 4px; }"
            "QPushButton:hover { background: #3a8a3a; }"
            "QPushButton:disabled { background: #444444; color: #888888; }"
        )
        self._run_btn.clicked.connect(self._on_run_clicked)
        run_bar.addWidget(self._run_btn)

        self._stop_btn = QPushButton("■  Stop")
        self._stop_btn.setFixedHeight(36)
        self._stop_btn.setFixedWidth(90)
        self._stop_btn.setEnabled(False)
        self._stop_btn.setStyleSheet(
            "QPushButton { background: #6a2a2a; color: white; border-radius: 4px; }"
            "QPushButton:hover { background: #8a3a3a; }"
            "QPushButton:disabled { background: #444444; color: #888888; }"
        )
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        run_bar.addWidget(self._stop_btn)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setFixedHeight(36)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("Ready")
        run_bar.addWidget(self._progress_bar)

        setup_layout.addLayout(run_bar)
        self._tabs.addTab(setup_widget, "Setup")

        # ---- Results tab ----
        self._results_panel = ResultsPanel()
        self._tabs.addTab(self._results_panel, "Results")

        # ---- Status bar ----
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Load a CIF or POSCAR file to begin.")

        # ---- Connect signals ----
        self._struct_panel.structure_loaded.connect(self._on_structure_loaded)
        self._struct_panel.supercell_ready.connect(self._on_supercell_ready)
        self._struct_panel.composition_changed.connect(self._on_composition_changed)

    # ------------------------------------------------------------------
    # Structure signals
    # ------------------------------------------------------------------

    def _on_structure_loaded(self, structure: Structure) -> None:
        self._status_bar.showMessage(
            f"Loaded: {structure.formula}  ({structure.num_atoms} atoms)"
        )

    def _on_supercell_ready(self, supercell: Structure) -> None:
        self._supercell = supercell
        self._params_panel.set_structure(supercell)
        self._preview_viewer.plot_structure(supercell)
        self._status_bar.showMessage(
            f"Supercell ready: {supercell.formula}  ({supercell.num_atoms} atoms)"
        )

    def _on_composition_changed(self) -> None:
        # Update pair weights table with current species
        if self._supercell is None:
            return
        _, comp = self._struct_panel.get_sublattice_config()
        unique_z = sorted(set(self._supercell.species) | set(comp.keys()))
        self._params_panel.set_species(unique_z)

    # ------------------------------------------------------------------
    # Run / Stop
    # ------------------------------------------------------------------

    def _on_run_clicked(self) -> None:
        if self._supercell is None:
            QMessageBox.warning(self, "No Structure", "Please load a structure file first.")
            return

        if not self._struct_panel.is_composition_valid():
            QMessageBox.warning(
                self,
                "Invalid Composition",
                "Composition counts do not sum to site counts.\n"
                "Please fix the substitution site assignments.",
            )
            return

        sublattices_raw, comp = self._struct_panel.get_sublattice_config()
        if not sublattices_raw:
            QMessageBox.warning(
                self,
                "No Active Sublattice",
                "No active substitution sites detected.\n"
                "Rows that keep the original species are frozen by design.",
            )
            return
        if not comp:
            QMessageBox.warning(self, "No Composition", "Composition is empty.")
            return

        params = self._params_panel.get_config()
        shell_weights = params["shell_weights"]
        if not shell_weights:
            QMessageBox.warning(
                self,
                "No Shell Weights",
                "All shell weights are 0. Set at least one shell weight > 0.",
            )
            return

        # Build pair_weights array if overrides exist
        pair_weights = None
        if params["pair_overrides"]:
            unique_z = sorted(set(self._supercell.species) | set(comp.keys()))
            K = len(unique_z)
            S = len(shell_weights)
            z_to_idx = {z: i for i, z in enumerate(unique_z)}
            active_shells = sorted(shell_weights.keys())
            s_to_idx = {s: i for i, s in enumerate(active_shells)}
            pair_weights = np.ones((S, K, K), dtype=np.float64)
            diag = np.arange(K)
            pair_weights[:, diag, diag] = 0.0
            for (s, za, zb, w) in params["pair_overrides"]:
                if s in s_to_idx and za in z_to_idx and zb in z_to_idx:
                    si = s_to_idx[s]
                    ai = z_to_idx[za]
                    bi = z_to_idx[zb]
                    pair_weights[si, ai, bi] = w
                    pair_weights[si, bi, ai] = w

        mode_label = str(params["mode"]).lower()
        if mode_label.startswith("anneal"):
            search_mode = "anneal"
        elif mode_label.startswith("systematic"):
            search_mode = "systematic"
        else:
            search_mode = "random"
        sublattices = [
            Sublattice(
                sites=list(sl["sites"]),
                composition=dict(sl["composition"]),
                group_label=str(sl.get("group_label")) if sl.get("group_label") else None,
            )
            for sl in sublattices_raw
        ]

        primitive = self._struct_panel.get_primitive()
        sa, sb, sc = self._struct_panel.get_supercell_dims()
        volume = int(sa) * int(sb) * int(sc)

        config = OptimizationConfig(
            structure=self._supercell,
            sublattices=sublattices,
            shell_weights=shell_weights,
            pair_weights=pair_weights,
            target=None,
            shell_radii=params["shell_radii"],
            iterations=params["iterations"],
            keep=params["keep"],
            atol=params["atol"],
            rtol=params["rtol"],
            iteration_mode=search_mode,
            seed=params["seed"],
            search_mode=search_mode,
            anneal_start_temp=params["anneal_t0"],
            anneal_end_temp=params["anneal_t1"],
            triplet_weight=params["triplet_weight"],
            enable_shape_optimization=bool(params["enable_shape_opt"]),
            primitive_structure=primitive,
            supercell_volume=volume,
            supercell_dims=(sa, sb, sc),
            max_shape_candidates=params["shape_candidates"],
            num_threads=params["num_threads"],
        )
        self._last_config = config

        self._worker = SQSWorker(config)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("Running…  0%")
        self._status_bar.showMessage("Optimization running…")
        self._results_panel.clear()

        self._worker.start()

    def _on_stop_clicked(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self._stop_btn.setEnabled(False)
            self._status_bar.showMessage("Stop requested — finishing current chunk…")

    def _on_progress(self, iteration: int, total: int, best_obj: float) -> None:
        pct = int(100 * iteration / max(total, 1))
        self._progress_bar.setValue(pct)
        obj_str = f"{best_obj:.6f}" if best_obj < 1e10 else "—"
        self._progress_bar.setFormat(
            f"{pct}%  iter {iteration:,}/{total:,}  best={obj_str}"
        )
        QApplication.processEvents()

    def _on_finished(self, results: List[SQSResult]) -> None:
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._progress_bar.setValue(100)

        if not results:
            self._progress_bar.setFormat("Done — no results found")
            self._status_bar.showMessage(
                "Optimization finished. No results found — try more iterations or "
                "check shell weights."
            )
            QMessageBox.information(
                self,
                "No Results",
                "The optimizer finished but found no results.\n\n"
                "Suggestions:\n"
                "• Increase the number of iterations\n"
                "• Check that shell weights are set correctly\n"
                "• Verify the composition sums match site counts",
            )
            return

        best = results[0].objective
        self._progress_bar.setFormat(
            f"Done — {len(results)} result(s), best obj = {best:.6f}"
        )
        self._status_bar.showMessage(
            f"Optimization complete: {len(results)} unique result(s), "
            f"best objective = {best:.6f}"
        )

        self._results_panel.display_results(results, self._supercell, self._last_config)
        self._tabs.setCurrentIndex(1)

    def _on_error(self, message: str) -> None:
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("Error")
        self._status_bar.showMessage(f"Error: {message}")
        QMessageBox.critical(self, "Optimization Error", message)
