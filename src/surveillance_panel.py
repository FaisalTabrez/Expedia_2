import asyncio
from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, 
    QTextEdit, QFileDialog, QMessageBox
)

from src.bridge import BridgeClient

class SurveillancePanel(QWidget):
    """Panel for importing and surveilling custom DNA sequences into the Manifold."""
    
    def __init__(self, bridge_client: BridgeClient, map_widget, loop: asyncio.AbstractEventLoop, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.bridge_client = bridge_client
        self.map_widget = map_widget
        self.loop = loop
        
        layout = QVBoxLayout(self)
        
        self.title_label = QLabel("<b>Surveillance Input</b><br>Upload FASTA or fast-paste raw DNA.")
        layout.addWidget(self.title_label)
        
        self.sequence_text = QTextEdit()
        self.sequence_text.setPlaceholderText("ATGC...")
        layout.addWidget(self.sequence_text)
        
        self.load_fasta_btn = QPushButton("Load from FASTA")
        self.load_fasta_btn.clicked.connect(self._load_fasta)
        layout.addWidget(self.load_fasta_btn)
        
        self.analyze_btn = QPushButton("Analyze Sequence")
        self.analyze_btn.clicked.connect(self._analyze)
        self.analyze_btn.setStyleSheet("background-color: darkred; color: white; font-weight: bold;")
        layout.addWidget(self.analyze_btn)
        
        self.results_label = QLabel("Result: N/A")
        self.results_label.setWordWrap(True)
        layout.addWidget(self.results_label)
        
    def _load_fasta(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open FASTA", "", "FASTA Files (*.fasta *.fa)")
        if not file_path:
            return
        try:
            # Quick read of first sequence
            from Bio import SeqIO
            with open(file_path, "r") as f:
                record = next(SeqIO.parse(f, "fasta"))
                self.sequence_text.setPlainText(str(record.seq))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not read FASTA: {e}")

    def _analyze(self):
        seq = self.sequence_text.toPlainText().strip()
        if not seq:
            return
            
        self.analyze_btn.setText("Analyzing...")
        self.analyze_btn.setEnabled(False)
        
        async def submit_request():
            try:
                result = await self.bridge_client.request(
                    "analyze_surveillance_sequence",
                    {"sequence": seq}
                )
                if result.get("status") == "ok":
                    consensus = result.get("consensus", {})
                    status = consensus.get("status", "UNKNOWN")
                    
                    # Update label in UI thread (PySide6 generally tolerates simple setText from thread, but QMetaObject.invokeMethod is safer)
                    self.results_label.setText(f"Status: {status}\nConsensus: {consensus}")

                    # Highlight neighbors
                    neighbors = result.get("neighbors", [])
                    if neighbors:
                        self.map_widget.highlight_query(neighbors)
                else:
                    self.results_label.setText(f"Error: {result.get('message')}")
            except Exception as e:
                self.results_label.setText(f"Error calling RPC: {e}")
            finally:
                self.analyze_btn.setText("Analyze Sequence")
                self.analyze_btn.setEnabled(True)

        asyncio.run_coroutine_threadsafe(submit_request(), self.loop)
