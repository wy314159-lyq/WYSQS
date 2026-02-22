# gui/structure_panel.py
# File loading, supercell settings, and substitution site editor.

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.elements import all_symbols, symbol_to_z, z_to_symbol
from core.io import read_structure
from core.structure import Structure, make_supercell


class _CompositionRowWidget(QWidget):
    """
    Inline widget for one species-count pair inside the substitution table.
    Shows a QComboBox (element) + QSpinBox (count).
    """
    changed = pyqtSignal()

    def __init__(self, symbols: List[str], parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.setSpacing(4)

        self.combo = QComboBox()
        self.combo.addItems(symbols)
        self.combo.setFixedWidth(70)

        self.spin = QSpinBox()
        self.spin.setRange(0, 9999)
        self.spin.setValue(0)
        self.spin.setFixedWidth(60)

        layout.addWidget(self.combo)
        layout.addWidget(QLabel("×"))
        layout.addWidget(self.spin)
        layout.addStretch()

        self.combo.currentIndexChanged.connect(self.changed)
        self.spin.valueChanged.connect(self.changed)

    def get_species(self) -> str:
        return self.combo.currentText()

    def get_count(self) -> int:
        return self.spin.value()

    def set_species(self, sym: str) -> None:
        idx = self.combo.findText(sym)
        if idx >= 0:
            self.combo.setCurrentIndex(idx)

    def set_count(self, n: int) -> None:
        self.spin.setValue(n)


class _SiteGroupRow(QWidget):
    """
    One row in the substitution editor representing a group of sites
    that share the same crystallographic site label (Wyckoff position).
    Contains multiple _CompositionRowWidget entries.
    """
    changed = pyqtSignal()

    def __init__(
        self,
        group_label: str,
        original_z: int,
        site_count: int,
        site_indices: List[int],
        all_syms: List[str],
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.group_label = group_label
        self.original_z = original_z
        self.site_count = site_count
        self.site_indices = site_indices
        self._rows: List[_CompositionRowWidget] = []

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 2, 4, 2)
        main_layout.setSpacing(2)

        # Header
        hdr = QHBoxLayout()
        orig_sym = z_to_symbol(original_z)
        self._header_label = QLabel(
            f"<b>{group_label}</b> ({orig_sym}, {site_count} sites)"
        )
        self._header_label.setFixedWidth(180)
        hdr.addWidget(self._header_label)

        self._status_label = QLabel()
        self._status_label.setFixedWidth(160)
        hdr.addWidget(self._status_label)

        add_btn = QPushButton("+")
        add_btn.setFixedWidth(28)
        add_btn.setToolTip("Add another species")
        add_btn.clicked.connect(self._add_row)
        hdr.addWidget(add_btn)

        rm_btn = QPushButton("−")
        rm_btn.setFixedWidth(28)
        rm_btn.setToolTip("Remove last species")
        rm_btn.clicked.connect(self._remove_row)
        hdr.addWidget(rm_btn)
        hdr.addStretch()
        main_layout.addLayout(hdr)

        # Composition rows container
        self._rows_widget = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_widget)
        self._rows_layout.setContentsMargins(20, 0, 0, 0)
        self._rows_layout.setSpacing(2)
        main_layout.addWidget(self._rows_widget)

        self._all_syms = all_syms
        # Default: keep original species
        self._add_row(default_sym=orig_sym, default_count=site_count)
        self._update_status()

    def _add_row(
        self,
        default_sym: Optional[str] = None,
        default_count: int = 0,
    ) -> None:
        row = _CompositionRowWidget(self._all_syms)
        if default_sym:
            row.set_species(default_sym)
        row.set_count(default_count)
        row.changed.connect(self._on_changed)
        self._rows.append(row)
        self._rows_layout.addWidget(row)
        self._on_changed()

    def _remove_row(self) -> None:
        if len(self._rows) <= 1:
            return
        row = self._rows.pop()
        self._rows_layout.removeWidget(row)
        row.deleteLater()
        self._on_changed()

    def _on_changed(self) -> None:
        self._update_status()
        self.changed.emit()

    def _update_status(self) -> None:
        total = sum(r.get_count() for r in self._rows)
        if total == self.site_count:
            self._status_label.setText(
                f'<span style="color:#44cc44;">✓ {total}/{self.site_count}</span>'
            )
        else:
            self._status_label.setText(
                f'<span style="color:#cc4444;">✗ {total}/{self.site_count}</span>'
            )

    def is_valid(self) -> bool:
        return sum(r.get_count() for r in self._rows) == self.site_count

    def get_composition(self) -> Dict[int, int]:
        """Return {Z: count} for this site group."""
        comp: Dict[int, int] = {}
        for row in self._rows:
            sym = row.get_species()
            cnt = row.get_count()
            if cnt > 0:
                try:
                    z = symbol_to_z(sym)
                    comp[z] = comp.get(z, 0) + cnt
                except ValueError:
                    pass
        return comp


class StructurePanel(QWidget):
    """
    Left panel: file loading, supercell settings, substitution site editor.

    Signals:
      structure_loaded(Structure)   — primitive cell loaded
      supercell_ready(Structure)    — supercell after Apply
      composition_changed()         — user edited composition
    """

    structure_loaded = pyqtSignal(object)
    supercell_ready = pyqtSignal(object)
    composition_changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._primitive: Optional[Structure] = None
        self._supercell: Optional[Structure] = None
        self._site_rows: List[_SiteGroupRow] = []
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        layout.addWidget(self._build_file_group())
        layout.addWidget(self._build_supercell_group())
        layout.addWidget(self._build_substitution_group())
        layout.addStretch()

    def _build_file_group(self) -> QGroupBox:
        gb = QGroupBox("Structure File")
        vbox = QVBoxLayout(gb)

        btn = QPushButton("Open CIF / POSCAR…")
        btn.clicked.connect(self._open_file)
        vbox.addWidget(btn)

        self._file_label = QLabel("No file loaded")
        self._file_label.setWordWrap(True)
        self._file_label.setStyleSheet("color: #888888; font-size: 11px;")
        vbox.addWidget(self._file_label)

        self._info_label = QLabel("")
        self._info_label.setWordWrap(True)
        self._info_label.setStyleSheet("font-size: 11px;")
        vbox.addWidget(self._info_label)

        return gb

    def _build_supercell_group(self) -> QGroupBox:
        gb = QGroupBox("Supercell")
        grid = QHBoxLayout(gb)

        for label, attr in [("a×", "_sa"), ("b×", "_sb"), ("c×", "_sc")]:
            grid.addWidget(QLabel(label))
            spin = QSpinBox()
            spin.setRange(1, 10)
            spin.setValue(1)
            spin.setFixedWidth(50)
            spin.valueChanged.connect(self._update_atom_count)
            setattr(self, attr, spin)
            grid.addWidget(spin)

        self._atom_count_label = QLabel("Atoms: —")
        self._atom_count_label.setStyleSheet("font-size: 11px; color: #aaaaaa;")
        grid.addWidget(self._atom_count_label)

        apply_btn = QPushButton("Apply")
        apply_btn.setFixedWidth(60)
        apply_btn.clicked.connect(self._apply_supercell)
        grid.addWidget(apply_btn)

        return gb

    def _build_substitution_group(self) -> QGroupBox:
        gb = QGroupBox("Substitution Sites")
        vbox = QVBoxLayout(gb)

        info = QLabel(
            "Define which species to place on each site group.\n"
            "Counts must sum to the number of sites in each group."
        )
        info.setWordWrap(True)
        info.setStyleSheet("font-size: 11px; color: #aaaaaa;")
        vbox.addWidget(info)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(200)

        self._sub_container = QWidget()
        self._sub_layout = QVBoxLayout(self._sub_container)
        self._sub_layout.setContentsMargins(0, 0, 0, 0)
        self._sub_layout.setSpacing(4)
        self._sub_layout.addStretch()

        scroll.setWidget(self._sub_container)
        vbox.addWidget(scroll)

        self._validation_label = QLabel("")
        self._validation_label.setWordWrap(True)
        self._validation_label.setStyleSheet("font-size: 11px;")
        vbox.addWidget(self._validation_label)

        return gb

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Structure File",
            "",
            "Structure files (*.cif POSCAR CONTCAR *.vasp);;All files (*)",
        )
        if not path:
            return
        try:
            structure = read_structure(path)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
            return

        self._primitive = structure
        import os
        self._file_label.setText(os.path.basename(path))
        self._update_info_label(structure)
        self._update_atom_count()
        self._apply_supercell()
        self.structure_loaded.emit(structure)

    def _update_info_label(self, st: Structure) -> None:
        a, b, c, alpha, beta, gamma = st.lattice_params
        self._info_label.setText(
            f"Formula: {st.formula}  |  N={st.num_atoms}\n"
            f"a={a:.3f} b={b:.3f} c={c:.3f} Å\n"
            f"α={alpha:.2f}° β={beta:.2f}° γ={gamma:.2f}°"
        )

    def _update_atom_count(self) -> None:
        if self._primitive is None:
            self._atom_count_label.setText("Atoms: —")
            return
        sa = self._sa.value()
        sb = self._sb.value()
        sc = self._sc.value()
        n = self._primitive.num_atoms * sa * sb * sc
        self._atom_count_label.setText(f"Atoms: {n}")

    def _apply_supercell(self) -> None:
        if self._primitive is None:
            return
        sa = self._sa.value()
        sb = self._sb.value()
        sc = self._sc.value()
        try:
            self._supercell = make_supercell(self._primitive, sa, sb, sc)
        except Exception as e:
            QMessageBox.critical(self, "Supercell Error", str(e))
            return
        self._populate_substitution_editor(self._supercell)
        self.supercell_ready.emit(self._supercell)

    def _populate_substitution_editor(self, st: Structure) -> None:
        """Rebuild the substitution site editor for the given structure."""
        # Clear existing rows
        for row in self._site_rows:
            self._sub_layout.removeWidget(row)
            row.deleteLater()
        self._site_rows.clear()

        # Remove stretch
        item = self._sub_layout.takeAt(self._sub_layout.count() - 1)
        if item:
            del item

        all_syms = all_symbols()
        group_comp = st.group_composition  # {group_label: (Z, [indices])}
        # Sort by (Z, label) for deterministic ordering
        sorted_groups = sorted(group_comp.items(), key=lambda x: (x[1][0], x[0]))

        for group_label, (z, indices) in sorted_groups:
            row = _SiteGroupRow(
                group_label=group_label,
                original_z=z,
                site_count=len(indices),
                site_indices=indices,
                all_syms=all_syms,
            )
            row.changed.connect(self._on_composition_changed)
            self._site_rows.append(row)
            self._sub_layout.addWidget(row)

        self._sub_layout.addStretch()
        self._on_composition_changed()

    def _on_composition_changed(self) -> None:
        valid = all(row.is_valid() for row in self._site_rows)
        if valid:
            self._validation_label.setText(
                '<span style="color:#44cc44;">✓ All compositions valid</span>'
            )
        else:
            self._validation_label.setText(
                '<span style="color:#cc4444;">✗ Some compositions do not sum to site count</span>'
            )
        self.composition_changed.emit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_supercell(self) -> Optional[Structure]:
        return self._supercell

    def is_composition_valid(self) -> bool:
        return bool(self._site_rows) and all(r.is_valid() for r in self._site_rows)

    def get_sublattice_config(self) -> Tuple[List[Dict[str, object]], Dict[int, int]]:
        """
        Return (active_sublattices, merged_composition) for the optimizer.

        active_sublattices: [{"sites": [...], "composition": {Z: count}}, ...]
          - rows that remain exactly unchanged (all original species) are skipped
        merged_composition: {Z: count} merged across active sublattices
        """
        if self._supercell is None:
            return [], {}

        active_sublattices: List[Dict[str, object]] = []
        merged_comp: Dict[int, int] = {}

        for row in self._site_rows:
            sites = row.site_indices
            comp = row.get_composition()
            # If this row stays exactly as the original species, it is frozen.
            if comp == {row.original_z: len(sites)}:
                continue

            active_sublattices.append(
                {
                    "sites": sorted(sites),
                    "composition": comp,
                    "group_label": row.group_label,
                }
            )
            for z, cnt in comp.items():
                merged_comp[z] = merged_comp.get(z, 0) + cnt

        return active_sublattices, merged_comp

    def get_primitive(self) -> Optional[Structure]:
        return self._primitive

    def get_supercell_dims(self) -> Tuple[int, int, int]:
        return (self._sa.value(), self._sb.value(), self._sc.value())
