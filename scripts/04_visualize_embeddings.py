import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import pandas as pd

# Load model and entity embeddings
result = pipeline_result_from_directory("results/TransE_100e_3000_e128")
model = result.model
factory = result.training

# Get entity embeddings
entity_tensor = model.entity_representations[0](indices=None).detach().cpu().numpy()
entity_list = list(factory.entity_to_id.keys())

# TSNE projection
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(entity_tensor)

# Plot
plt.figure(figsize=(10, 10))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=3, alpha=0.6)
plt.title("TSNE Projection of Entity Embeddings")
plt.xlabel("TSNE-1")
plt.ylabel("TSNE-2")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/entity_embeddings_tsne.png")
plt.show()
print("[✓] TSNE plot saved to results/entity_embeddings_tsne.png")
