from pathlib import Path

content = """import asyncio
import numpy as np
import pyvista as pv
from typing import List, Sequence
from pyvistaqt import QtInteractor
from PySide6.QtWidgets import QVBoxLayout, QWidget, QPushButton, QHBoxLayout
from PySide6.QtCore import QTimer

from src.bridge import BridgeClient

KINGDOM_COLORS = {
    "Bacteria": [0, 0, 255],
    "Eukaryota": [0, 255, 0],
    "Archaea": [255, 215, 0],
    "Viruses": [238, 130, 238]
}
DEFAULT_COLOR = [128, 128, 128]

class ExpediaMap(QWidget):
    \"\"\"3D visualization widget for EXPEDIA manifolds using PyVista and Qt.\"\"\"

    def __init__(self, bridge_client: BridgeClient, loop: asyncio.AbstractEventLoop = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.bridge_client = bridge_client
        self.loop = loop
        self._accessions: np.ndarray | None = None

        self._full_cloud: pv.PolyData | None = None
        self._preview_cloud: pv.PolyData | None = None
        self._is_lod_active = True
        self.lod_threshold = 1_000_000

        self.layout = QVBoxLayout(self)

        self.controls_layout = QHBoxLayout()
        self.lod_btn = QPushButton("Toggle LOD (High/Low)")
        self.lod_btn.clicked.connect(self.toggle_lod)
        self.controls_layout.addWidget(self.lod_btn)
        self.layout.addLayout(self.controls_layout)

        self.plotter = QtInteractor(self)
        self.layout.addWidget(self.plotter.interactor)
        self.setLayout(self.layout)
        
        self.flash_timer = QTimer(self)
        self.flash_timer.timeout.connect(self._toggle_hazard_color)
        self.hazard_state = False
        self.hazard_centroid = None

    def render_points(
        self,
        accessions: Sequence[str],
        xyz: np.ndarray,
        cluster_ids: np.ndarray,
        kingdoms: Sequence[str] = None,
        outlier_scores: np.ndarray = None,
    ) -> None:
        self._accessions = np.asarray(accessions)
        points = np.asarray(xyz, dtype=np.float32)
        
        num_points = points.shape[0]
        
        # Color mapping array
        rgba = np.zeros((num_points, 4), dtype=np.uint8)
        
        if kingdoms is None:
            kingdoms = ["Unknown"] * num_points
        if outlier_scores is None:
            outlier_scores = np.zeros(num_points, dtype=np.float32)
            
        for i, k in enumerate(kingdoms):
            color = KINGDOM_COLORS.get(k, DEFAULT_COLOR)
            rgba[i, :3] = color
            
        # Outlier score controls opacity/glow (0.0 to 1.0)
        # Assuming outlier_score higher means more outlier
        # Base opacity 50 (dim), high outlier up to 255 (bright)
        normalized_scores = np.clip(outlier_scores, 0, 1)
        rgba[:, 3] = (100 + 155 * normalized_scores).astype(np.uint8)

        self._full_cloud = pv.PolyData(points)
        self._full_cloud.point_data["colors"] = rgba # rgba array mapped directly

        if num_points > self.lod_threshold:
            indices = np.random.choice(num_points, self.lod_threshold, replace=False)
            self._preview_cloud = pv.PolyData(points[indices])
            self._preview_cloud.point_data["colors"] = rgba[indices]
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
            scalars="colors",
            rgba=True,
            point_size=2,
            render_points_as_spheres=True,
        )
        self.plotter.add_axes()
        self.plotter.show_grid()

    def toggle_lod(self):
        if self._preview_cloud is None:
            return

        self._is_lod_active = not self._is_lod_active
        if self._is_lod_active:
            self._render_cloud(self._preview_cloud)
        else:
            self._render_cloud(self._full_cloud)

    def enable_lasso_selection(self) -> None:
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

    def highlight_query(self, neighbor_accessions: List[str], anomaly: bool = False) -> None:
        if self._full_cloud is None or self._accessions is None:
            return

        mask = np.isin(self._accessions, neighbor_accessions)
        if not np.any(mask):
            return

        neighbor_points = self._full_cloud.points[mask]
        self.hazard_centroid = np.mean(neighbor_points, axis=0)

        # Remove previous marker if exists
        if hasattr(self, 'query_sphere_actor'):
            try:
                self.plotter.remove_actor(self.query_sphere_actor)
            except:
                pass

        if anomaly:
            # Start flashing red marker
            self.hazard_state = True
            self.flash_timer.start(500) # 500ms
            self._draw_marker(color="red")
        else:
            # Normal pulsing/static white marker
            self.flash_timer.stop()
            self._draw_marker(color="white")
            
        self.plotter.render()
        
    def _draw_marker(self, color="white"):
        sphere = pv.Sphere(radius=0.1, center=self.hazard_centroid)
        self.query_sphere_actor = self.plotter.add_mesh(
            sphere, 
            color=color, 
            opacity=0.8, 
            point_size=15, 
            render_points_as_spheres=True, 
            name="query_marker"
        )
        
    def _toggle_hazard_color(self):
        if not self.hazard_centroid is None:
            color = "red" if self.hazard_state else "yellow"
            self.hazard_state = not self.hazard_state
            self._draw_marker(color)
            self.plotter.render()
"""

Path("src/expedia_map.py").write_text(content)
