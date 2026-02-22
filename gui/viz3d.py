# gui/viz3d.py
# 3D crystal structure viewer embedded in PyQt5 using matplotlib.

from __future__ import annotations

from typing import List, Optional

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PyQt5.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from core.elements import z_to_color, z_to_radius_angstrom, z_to_symbol
from core.structure import Structure


class StructureViewer3D(QWidget):
    """
    Embeds a matplotlib 3D axes inside a QWidget.

    Features:
      - Atoms drawn as scatter spheres, size ∝ covalent radius, CPK colors
      - Unit cell box drawn as 12 edges from lattice vectors
      - Bonds drawn for pairs within 1.2 × (r_i + r_j)
      - Highlighted sites drawn with a yellow ring
      - Interactive rotate/zoom via matplotlib navigation toolbar
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._structure: Optional[Structure] = None
        self._highlight: List[int] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        self._fig = Figure(figsize=(5, 5), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plot_structure(
        self,
        structure: Structure,
        highlight_sites: Optional[List[int]] = None,
    ) -> None:
        """Render the structure. highlight_sites: atom indices to mark in yellow."""
        self._structure = structure
        self._highlight = highlight_sites or []
        self._draw()

    def clear(self) -> None:
        self._fig.clear()
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw(self) -> None:
        if self._structure is None:
            return

        self._fig.clear()
        ax: Axes3D = self._fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#1a1a2e')
        self._fig.patch.set_facecolor('#1a1a2e')

        st = self._structure
        cart = st.cart_coords   # (N, 3)
        N = st.num_atoms

        # ---- Unit cell box ----
        self._draw_cell(ax, st.lattice)

        # ---- Bonds ----
        self._draw_bonds(ax, cart, st.species)

        # ---- Atoms ----
        highlight_set = set(self._highlight)
        for i in range(N):
            z = st.species[i]
            color = z_to_color(z)
            r = z_to_radius_angstrom(z)
            size = max(20, min(300, (r * 120) ** 2))
            x, y, zc = cart[i]
            ax.scatter([x], [y], [zc], c=[color], s=size,
                       depthshade=True, edgecolors='white', linewidths=0.3, zorder=5)
            if i in highlight_set:
                ax.scatter([x], [y], [zc], c='yellow', s=size * 2.5,
                           depthshade=False, edgecolors='yellow',
                           linewidths=1.5, alpha=0.4, zorder=4)

        # ---- Atom labels (only for small structures) ----
        if N <= 30:
            for i in range(N):
                z = st.species[i]
                sym = z_to_symbol(z)
                x, y, zc = cart[i]
                ax.text(x, y, zc, sym, fontsize=6, color='white',
                        ha='center', va='center', zorder=6)

        # ---- Axis styling ----
        ax.set_xlabel('x (Å)', color='#aaaaaa', fontsize=8)
        ax.set_ylabel('y (Å)', color='#aaaaaa', fontsize=8)
        ax.set_zlabel('z (Å)', color='#aaaaaa', fontsize=8)
        ax.tick_params(colors='#aaaaaa', labelsize=7)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor('#333355')
        ax.grid(False)

        self._canvas.draw_idle()

    def _draw_cell(self, ax: Axes3D, lattice: np.ndarray) -> None:
        """Draw the 12 edges of the unit cell parallelepiped."""
        a, b, c = lattice[0], lattice[1], lattice[2]
        o = np.zeros(3)
        # 8 corners
        corners = [
            o, a, b, c,
            a + b, a + c, b + c,
            a + b + c,
        ]
        # 12 edges: pairs of corner indices
        edges = [
            (0, 1), (0, 2), (0, 3),
            (1, 4), (1, 5),
            (2, 4), (2, 6),
            (3, 5), (3, 6),
            (4, 7), (5, 7), (6, 7),
        ]
        segs = [[corners[i], corners[j]] for i, j in edges]
        lc = Line3DCollection(segs, colors='#5588cc', linewidths=0.8, alpha=0.7)
        ax.add_collection3d(lc)

    def _draw_bonds(self, ax: Axes3D, cart: np.ndarray, species: List[int]) -> None:
        """Draw bonds for pairs within 1.2 × (r_i + r_j)."""
        N = len(species)
        if N > 200:
            return  # skip bonds for large structures (too slow)

        segs = []
        for i in range(N):
            ri = z_to_radius_angstrom(species[i])
            for j in range(i + 1, N):
                rj = z_to_radius_angstrom(species[j])
                cutoff = 1.2 * (ri + rj)
                diff = cart[i] - cart[j]
                dist = float(np.linalg.norm(diff))
                if 0.1 < dist <= cutoff:
                    segs.append([cart[i], cart[j]])

        if segs:
            lc = Line3DCollection(segs, colors='#888888', linewidths=0.6, alpha=0.5)
            ax.add_collection3d(lc)
