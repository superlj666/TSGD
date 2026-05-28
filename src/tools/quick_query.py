import sys
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.tools.utils import get_embeddings_model, setup_logging

def main():
    query_text = r"A regular hexagon with center at the origin in the complex plane has opposite pairs of sides one unit apart. One pair of sides is parallel to the imaginary axis. Let $R$ be the region outside the hexagon, and let $S = \left\lbrace\frac{1}{z} \ | \ z \in R\right\rbrace$.  Find the area of $S.$"
    index_path = "./data/math_precalculus_5_val_text-embedding-3-large_idx.npz"
    threshold = 0.8

    # print(f"Loading index from {index_path}...")
    data = np.load(index_path)
    embeddings = data['embeddings']
    ids = data['ids'] if 'ids' in data else np.array([f"item_{i}" for i in range(len(embeddings))])

    # print("Generating query embedding...")
    model = get_embeddings_model("text-embedding-3-large")
    query_vec = model.embed_query(query_text)
    query_vec = np.array(query_vec).reshape(1, -1)

    # print("Calculating similarity...")
    sims = cosine_similarity(query_vec, embeddings)[0]

    # print(f"\nResults (Similarity > {threshold}):")
    found = False
    for i, sim in enumerate(sims):
        if sim > threshold:
            print(f"ID: {ids[i]}, Similarity: {sim:.4f}")
            found = True
    
    if not found:
        print("No items found with similarity > 0.8")

if __name__ == "__main__":
    main()
