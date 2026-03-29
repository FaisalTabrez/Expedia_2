import asyncio
from typing import List, Sequence

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtWidgets import QVBoxLayout, QWidget, QPushButton, QHBoxLayout

from src.bridge import BridgeClient


class ExpediaMap(QWidget):
    """3D visualization widget for EXPEDIA manifolds using PyVista and Qt."""

    def __init__(self, bridge_client: BridgeClient, loop: asyncio.AbstractEventLoop = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.bridge_client = bridge_client
        self.loop = loop
        self._accessions: np.ndarray | None = None
        
        # State for LOD
        self._full_cloud: pv.PolyData | None = None
        self._preview_cloud: pv.PolyData | None = None
        self._is_lod_active = True
        self.lod_threshold = 1_000_000

        self.layout = QVBoxLayout(self)
        
        # UI controls
        self.controls_layout = QHBoxLayout()
        self.lod_btn = QPushButton("Toggle LOD (High/Low)")
        self.lod_btn.clicked.connect(self.toggle_lod)
        self.controls_layout.addWidget(self.lod_btn)
        self.layout.addLayout(self.controls_layout)

        self.plotter = QtInteractor(self)
        self.layout.addWidget(self.plotter.interactor)
        self.setLayout(self.layout)

    def render_points(
        self,
        accessions: Sequence[str],
        xyz: np.ndarray,
        cluster_ids: np.ndarray,
    ) -> None:
        """Render manifold points as PolyData; supports multi-million point datasets."""
        self._accessions = np.asarray(accessions)
        points = np.asarray(xyz, dtype=np.float32)
        clusters = np.asarray(cluster_ids, dtype=np.int32)

        self._full_cloud = pv.PolyData(points)
        self._full_cloud["cluster_id"] = clusters

        num_points = len(accessions)
        if num_points > self.lod_threshold:
            # Random sampling for outline
            indices = np.random.choice(num_points, self.lod_threshold, replace=False)
            self._preview_cloud = pv.PolyData(points[indices])
            self._preview_cloud["cluster_id"] = clusters[indices]
            # Start with preview
            self._is_lod_active = True
            self._render_cloud(self._preview_cloud)
        else:
            self._preview_cloud = None
            self._is_lod_active = False
            self._render_cloud(self._full_cloud)

    def _render_cloud(self, cloud: pv.PolyData) -> None:
        self.plotter.clear()
        self.plotter.add_points(
            cloud,
            scalars="cluster_id",
            cmap="viridis",
            point_size=2,
            render_points_as_spheres=True,
        )
        self.plotter.add_axes()
        self.plotter.show_grid()

    def toggle_lod(self):
        if self._preview_cloud is None:
            return # not needed
        
        self._is_lod_active = not self._is_lod_active
        if self._is_lod_active:
            self._render_cloud(self._preview_cloud)
        else:
            self._render_cloud(self._full_cloud)

    def enable_lasso_selection(self) -> None:
        """Enable selection mode and send selected accession IDs back through the bridge."""

        def _on_select(selection: pv.PolyData) -> None:
            if selection is None or selection.n_points == 0 or self._accessions is None:
                return
            if "vtkOriginalPointIds" not in selection.point_data:
                return
            idx = np.asarray(selection.point_data["vtkOriginalPointIds"], dtype=np.int64)
            selected = self._accessions[idx].tolist()
            coro = self.bridge_client.notify(
                "consensus_lineage_request",
                {"accession_ids": selected},
            )
            if self.loop:
                asyncio.run_coroutine_threadsafe(coro, self.loop)
            else:
                asyncio.create_task(coro)

        self.plotter.enable_cell_picking(callback=_on_select, through=False, show_message=True)

    def highlight_query(self, neighbor_accessions: List[str]) -> None:
        """Highlight the query's location on the 3D Manifold with a large white marker."""
        if self._full_cloud is None or self._accessions is None:
            return
            
        # Find indices of neighbors
        mask = np.isin(self._accessions, neighbor_accessions)
        if not np.any(mask):
            return
            
        # Extract coordinates of neighbors
        neighbor_points = self._full_cloud.points[mask]
        
        # Calculate centroid to represent 'Evolutionary Neighborhood'
        centroid = np.mean(neighbor_points, axis=0)
        
        # Add a large white sphere at the centroid
        sphere = pv.Sphere(radius=0.1, center=centroid)
        self.plotter.add_mesh(sphere, color="white", opacity=0.8, point_size=15, render_points_as_spheres=True, name="query_marker")
        
        # Optional: draw lines from centroid to nearest neighbors
        # lines = pv.lines_from_points(np.vstack([np.tile(centroid, (len(neighbor_points), 1)), neighbor_points]))
        # self.plotter.add_mesh(lines, color="white", opacity=0.3, name="query_lines")
        
        self.plotter.render()

