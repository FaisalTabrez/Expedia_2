import asyncio
from typing import List, Sequence

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtWidgets import QVBoxLayout, QWidget

from src.bridge import BridgeClient


class ExpediaMap(QWidget):
    """3D visualization widget for EXPEDIA manifolds using PyVista and Qt."""

    def __init__(self, bridge_client: BridgeClient, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.bridge_client = bridge_client
        self._accessions: np.ndarray | None = None

        self.layout = QVBoxLayout(self)
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

        cloud = pv.PolyData(points)
        cloud["cluster_id"] = clusters

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

    def enable_lasso_selection(self) -> None:
        """Enable selection mode and send selected accession IDs back through the bridge.

        PyVista's rectangle selection is used here as a practical lasso-like interaction for dense clouds.
        """

        def _on_select(selection: pv.PolyData) -> None:
            if selection is None or selection.n_points == 0 or self._accessions is None:
                return
            if "vtkOriginalPointIds" not in selection.point_data:
                return
            idx = np.asarray(selection.point_data["vtkOriginalPointIds"], dtype=np.int64)
            selected = self._accessions[idx].tolist()
            asyncio.create_task(
                self.bridge_client.notify(
                    "consensus_lineage_request",
                    {"accession_ids": selected},
                )
            )

        self.plotter.enable_cell_picking(callback=_on_select, through=False, show_message=True)
