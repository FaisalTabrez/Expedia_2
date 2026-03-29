import re
with open("src/science_kernel.py", "r") as f:
    code = f.read()

# Modify AvalancheAnalyzer.run signature
code = code.replace(
    'def run(self, vectors: np.ndarray, accessions: List[str]) -> pl.DataFrame:',
    'def run(self, vectors: np.ndarray, accessions: List[str], kingdoms: List[str], outlier_scores_raw: List[float] = None) -> pl.DataFrame:'
)

# Update DataFrame creation
old_df = '''        manifold_df = pl.DataFrame(
            {
                "accession": accessions,
                "x": embedding_3d[:, 0],
                "y": embedding_3d[:, 1],
                "z": embedding_3d[:, 2],
                "cluster_id": cluster_labels,
                "outlier_score": outlier_scores,
            }
        )'''
new_df = '''        manifold_df = pl.DataFrame(
            {
                "accession": accessions,
                "kingdom": kingdoms,
                "x": embedding_3d[:, 0],
                "y": embedding_3d[:, 1],
                "z": embedding_3d[:, 2],
                "cluster_id": cluster_labels,
                "outlier_score": outlier_scores,
            }
        )'''
code = code.replace(old_df, new_df)

with open("src/science_kernel.py", "w") as f:
    f.write(code)
