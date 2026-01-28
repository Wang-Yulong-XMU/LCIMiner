import numpy as np
from sentence_transformers import SentenceTransformer

def embed_vocab_full_precision(
    vocab,
    model_name="all-MiniLM-L6-v2",
    batch_size=32
):
    model = SentenceTransformer(model_name)

    embeddings = model.encode(
        vocab,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True 
    )

    embeddings = embeddings.astype(np.float64)

    return embeddings
def cosine_similarity_matrix_full_precision(embeddings):
    """
    embeddings: (N, D), already normalized, float64
    return: (N, N) cosine similarity matrix, float64
    """
    similarity_matrix = np.matmul(embeddings, embeddings.T)
    return similarity_matrix
from sklearn.cluster import AffinityPropagation

def affinity_propagation_cluster(similarity_matrix):
    """
    similarity_matrix: (N, N), float64
    """

    preference = np.median(similarity_matrix)

    ap = AffinityPropagation(
        affinity="precomputed",
        damping=0.9,
        preference=preference,
        max_iter=1000,
        convergence_iter=50,
        random_state=42,
        verbose=True
    )

    ap.fit(similarity_matrix)

    return ap
from collections import defaultdict

def build_clusters(vocab, ap):
    """
    return:
    {
        cluster_id: {
            "exemplar": str,
            "members": List[str]
        }
    }
    """

    clusters = defaultdict(list)

    for idx, label in enumerate(ap.labels_):
        clusters[label].append(vocab[idx])

    result = {}
    for label, members in clusters.items():
        exemplar_idx = ap.cluster_centers_indices_[label]
        result[label] = {
            "exemplar": vocab[exemplar_idx],
            "members": members
        }

    return result
from collections import defaultdict

def build_large_clusters(
    vocab,
    ap,
    min_cluster_size=20
):

    clusters = defaultdict(list)

    for idx, label in enumerate(ap.labels_):
        clusters[label].append(idx)

    results = []

    for label, member_indices in clusters.items():
        cluster_size = len(member_indices)
        if cluster_size < min_cluster_size:
            continue

        exemplar_idx = ap.cluster_centers_indices_[label]

        results.append({
            "cluster_id": int(label),
            "exemplar": vocab[exemplar_idx],
            "cluster_size": cluster_size,
            "members": [vocab[i] for i in member_indices]
        })

    results.sort(key=lambda x: x["cluster_size"], reverse=True)

    return results
import json

def save_clusters_to_json(clusters, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)




embeddings = embed_vocab_full_precision(vocab)

similarity_matrix = cosine_similarity_matrix_full_precision(embeddings)

ap = affinity_propagation_cluster(similarity_matrix)

large_clusters = build_large_clusters(
    vocab,
    ap,
    min_cluster_size=20
)

print("Number of retained clusters:", len(large_clusters))
save_clusters_to_json(
    large_clusters,
    "ap_large_clusters_min20.json"
)
