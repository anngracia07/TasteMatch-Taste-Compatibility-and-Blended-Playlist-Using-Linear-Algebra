import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from data_loader import FEATURES
from linear_algebra import cosine_similarity, rref, gram_schmidt, principal_angles


def analyze_taste(df, selected_you, selected_them):
    """Run the core PCA/subspace compatibility analysis."""
    # 1. Selected song matrices
    A_raw = df.loc[selected_you, FEATURES].to_numpy(dtype=float)
    B_raw = df.loc[selected_them, FEATURES].to_numpy(dtype=float)

    # 2. Common standardization
    combined_selected = np.vstack([A_raw, B_raw])
    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined_selected)

    A_scaled = combined_scaled[:len(A_raw)]
    B_scaled = combined_scaled[len(A_raw):]

    # 3. Rank analysis
    A_matrix = A_scaled.T
    B_matrix = B_scaled.T

    rank_A = int(np.linalg.matrix_rank(A_matrix))
    rank_B = int(np.linalg.matrix_rank(B_matrix))

    rref_A = rref(A_matrix)
    rref_B = rref(B_matrix)

    # 4. PCA for each user's music collection
    n_components = min(3, A_scaled.shape[0], A_scaled.shape[1])
    n_components_B = min(3, B_scaled.shape[0], B_scaled.shape[1])

    if n_components < 3 or n_components_B < 3:
        raise ValueError("At least 3 songs are required for a 3-component PCA basis.")

    pca_A = PCA(n_components=3)
    pca_B = PCA(n_components=3)

    scores_A = pca_A.fit_transform(A_scaled)
    scores_B = pca_B.fit_transform(B_scaled)

    QA = pca_A.components_.T
    QB = pca_B.components_.T

    # 5. Gram-Schmidt supporting calculation
    GS_A = gram_schmidt(A_matrix)
    GS_B = gram_schmidt(B_matrix)

    # 6. Principal angles
    angles, singular_values, M, U, Vt = principal_angles(QA, QB)
    subspace_compatibility = float(np.mean(singular_values) * 100.0)

    # 7. Average-vector cosine baseline
    baseline_scaler = StandardScaler()
    all_baseline_scaled = baseline_scaler.fit_transform(
        df[FEATURES].to_numpy(dtype=float)
    )

    A_baseline = all_baseline_scaled[selected_you]
    B_baseline = all_baseline_scaled[selected_them]

    mean_A = np.mean(A_baseline, axis=0)
    mean_B = np.mean(B_baseline, axis=0)

    cosine = cosine_similarity(mean_A, mean_B)
    cosine_percent = ((cosine + 1.0) / 2.0) * 100.0

    # 8. Shared / consensus taste subspace
    combined_basis = np.hstack([QA, QB])
    shared_U, shared_S, shared_Vt = np.linalg.svd(
        combined_basis,
        full_matrices=False
    )
    Q_shared = shared_U[:, :3]

    P_A = QA @ QA.T
    P_B = QB @ QB.T

    return {
        "A_raw": A_raw,
        "B_raw": B_raw,
        "A_scaled": A_scaled,
        "B_scaled": B_scaled,
        "A_matrix": A_matrix,
        "B_matrix": B_matrix,
        "rank_A": rank_A,
        "rank_B": rank_B,
        "rref_A": rref_A,
        "rref_B": rref_B,
        "pca_A": pca_A,
        "pca_B": pca_B,
        "scores_A": scores_A,
        "scores_B": scores_B,
        "QA": QA,
        "QB": QB,
        "GS_A": GS_A,
        "GS_B": GS_B,
        "angles": angles,
        "singular_values": singular_values,
        "M": M,
        "subspace_compatibility": subspace_compatibility,
        "mean_A": mean_A,
        "mean_B": mean_B,
        "cosine": cosine,
        "cosine_percent": cosine_percent,
        "Q_shared": Q_shared,
        "shared_singular_values": shared_S,
        "P_A": P_A,
        "P_B": P_B,
        "scaler": scaler,
    }
