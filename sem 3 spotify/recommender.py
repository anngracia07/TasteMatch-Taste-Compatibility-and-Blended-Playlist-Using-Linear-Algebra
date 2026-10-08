import numpy as np
from sklearn.decomposition import PCA

from data_loader import FEATURES
from linear_algebra import minmax_score


def generate_recommendations(df, selected_you, selected_them, analysis):
    """Generate projection-based recommendations and visualization data."""
    scaler = analysis["scaler"]
    QA = analysis["QA"]
    QB = analysis["QB"]
    Q_shared = analysis["Q_shared"]
    subspace_compatibility = analysis["subspace_compatibility"]

    # 9. Candidate songs
    selected_set = set(selected_you + selected_them)
    candidate_indices = [idx for idx in df.index if idx not in selected_set]

    candidate_raw = df.loc[candidate_indices, FEATURES].to_numpy(dtype=float)
    candidate_scaled = scaler.transform(candidate_raw)

    if len(candidate_scaled) == 0:
        raise ValueError("There are no candidate songs left for recommendations.")

    # 10. Projection onto shared taste space
    projected_shared = candidate_scaled @ Q_shared @ Q_shared.T
    shared_error = np.linalg.norm(candidate_scaled - projected_shared, axis=1)

    projected_A = candidate_scaled @ QA @ QA.T
    projected_B = candidate_scaled @ QB @ QB.T

    error_A = np.linalg.norm(candidate_scaled - projected_A, axis=1)
    error_B = np.linalg.norm(candidate_scaled - projected_B, axis=1)
    average_personal_error = (error_A + error_B) / 2.0

    # 11. Recommendation score
    shared_closeness = 1.0 - minmax_score(shared_error)
    personal_closeness = 1.0 - minmax_score(average_personal_error)

    recommendation_score = 100.0 * (
        0.70 * shared_closeness + 0.30 * personal_closeness
    )

    # 12. Recommended playlist
    recommendation_df = df.loc[
        candidate_indices,
        ["track_name", "artists", "track_genre"],
    ].copy()

    recommendation_df["recommendation_score"] = recommendation_score
    recommendation_df["projection_error"] = shared_error
    recommendation_df["fit_to_you"] = (1.0 - minmax_score(error_A)) * 100.0
    recommendation_df["fit_to_them"] = (1.0 - minmax_score(error_B)) * 100.0

    recommendation_df = (
        recommendation_df
        .sort_values(by="recommendation_score", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    # 13. PCA for visualization
    all_raw = df[FEATURES].to_numpy(dtype=float)
    all_scaled = scaler.transform(all_raw)

    pca_visual = PCA(n_components=3)
    all_pca = pca_visual.fit_transform(all_scaled)

    A_pca = all_pca[selected_you]
    B_pca = all_pca[selected_them]

    recommended_original_indices = []

    for _, rec in recommendation_df.iterrows():
        matches = df.index[
            (df["track_name"].astype(str) == str(rec["track_name"]))
            & (df["artists"].astype(str) == str(rec["artists"]))
        ].tolist()

        match = next(
            (idx for idx in matches if idx not in selected_set),
            None,
        )

        if match is not None:
            recommended_original_indices.append(match)

    if recommended_original_indices:
        recommended_pca = all_pca[recommended_original_indices]
    else:
        recommended_pca = np.empty((0, 3))

    # 14. Feature means for visualization
    feature_mean_A = df.loc[selected_you, FEATURES].mean().to_numpy()
    feature_mean_B = df.loc[selected_them, FEATURES].mean().to_numpy()

    # 15. Verdict
    if subspace_compatibility >= 85:
        verdict = "Very high geometric compatibility"
    elif subspace_compatibility >= 70:
        verdict = "High geometric compatibility"
    elif subspace_compatibility >= 50:
        verdict = "Moderate geometric compatibility"
    else:
        verdict = "Low geometric compatibility"

    return {
        "candidate_indices": candidate_indices,
        "shared_error": shared_error,
        "error_A": error_A,
        "error_B": error_B,
        "recommendation_df": recommendation_df,
        "all_pca": all_pca,
        "A_pca": A_pca,
        "B_pca": B_pca,
        "recommended_pca": recommended_pca,
        "pca_visual": pca_visual,
        "feature_mean_A": feature_mean_A,
        "feature_mean_B": feature_mean_B,
        "verdict": verdict,
    }
