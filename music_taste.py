"""Linear algebra pipeline for comparing two musical taste subspaces."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "music_taste_project_dataset.csv"
FEATURES = ["danceability", "energy", "valence", "acousticness",
            "speechiness", "instrumentalness", "tempo"]
REQUIRED_COLUMNS = ["track_name", "artists", "track_genre", *FEATURES]
TOP_SONGS = 15
TASTE_DIM = 3
MIN_SELECTION = 5
RREF_TOL = 1e-9
COMPATIBILITY_THRESHOLDS = {"very_high": 80.0, "high": 65.0, "moderate": 45.0}

# Editable synthetic-profile weights. Scores use globally standardized values.
USER_A_WEIGHTS = {"energy": 1.0, "danceability": 1.0, "valence": 1.0}
USER_B_WEIGHTS = {"acousticness": 1.0, "energy": -1.0, "danceability": -1.0}


class DatasetError(ValueError):
    """Raised when the supplied runtime dataset is not usable."""


def load_dataset(path=DATA_PATH):
    """Load and validate the CSV; impute missing feature values with medians."""
    if not Path(path).is_file():
        raise DatasetError(f"Dataset not found: {Path(path).name}")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise DatasetError(f"Could not read the dataset: {exc}") from exc
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise DatasetError("Dataset is missing required columns: " + ", ".join(missing))
    if frame.empty:
        raise DatasetError("The dataset contains no songs.")
    frame = frame.copy()
    malformed_numeric_values = 0
    imputed_numeric_values = 0
    for feature in FEATURES:
        original = frame[feature]
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
        frame[feature] = frame[feature].replace([np.inf, -np.inf], np.nan)
        if frame[feature].isna().all():
            raise DatasetError(f"Feature '{feature}' has no usable numeric values.")
        malformed = frame[feature].isna() & original.notna() & original.astype(str).str.strip().ne("")
        malformed_numeric_values += int(malformed.sum())
        imputed_numeric_values += int(frame[feature].isna().sum())
        frame[feature] = frame[feature].fillna(frame[feature].median())
    for label in ("track_name", "artists", "track_genre"):
        frame[label] = frame[label].fillna("Unknown").astype(str)
    frame.insert(0, "track_id", np.arange(len(frame), dtype=int))
    duplicates = int(frame.duplicated(subset=REQUIRED_COLUMNS).sum())
    return frame, duplicates, {
        "imputed_numeric_values": imputed_numeric_values,
        "malformed_numeric_values": malformed_numeric_values,
    }


def rref(matrix, tolerance=RREF_TOL):
    """Gauss-Jordan elimination with partial pivoting. Returns RREF, pivots, rank."""
    result = np.asarray(matrix, dtype=float).copy()
    if result.ndim != 2:
        raise ValueError("RREF input must be a two-dimensional matrix.")
    rows, columns = result.shape
    pivot_columns = []
    pivot_row = 0
    for column in range(columns):
        if pivot_row >= rows:
            break
        candidate = pivot_row + int(np.argmax(np.abs(result[pivot_row:, column])))
        if abs(result[candidate, column]) <= tolerance:
            result[np.abs(result[:, column]) <= tolerance, column] = 0.0
            continue
        result[[pivot_row, candidate]] = result[[candidate, pivot_row]]
        result[pivot_row] /= result[pivot_row, column]
        for row in range(rows):
            if row != pivot_row:
                result[row] -= result[row, column] * result[pivot_row]
        result[np.abs(result) <= tolerance] = 0.0
        pivot_columns.append(column)
        pivot_row += 1
    return result, pivot_columns, len(pivot_columns)


def gram_schmidt(vectors, tolerance=RREF_TOL):
    """Modified Gram-Schmidt on vectors supplied as columns; skips dependent inputs."""
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2:
        raise ValueError("Gram-Schmidt input must be a two-dimensional matrix.")
    orthogonal = []
    for column in range(vectors.shape[1]):
        vector = vectors[:, column].copy()
        for unit in orthogonal:
            vector -= np.dot(unit, vector) * unit
        length = np.linalg.norm(vector)
        if length > tolerance:
            orthogonal.append(vector / length)
    return np.column_stack(orthogonal) if orthogonal else np.empty((vectors.shape[0], 0))


def _json_matrix(array, decimals=4):
    return np.round(np.asarray(array, dtype=float), decimals).tolist()


def _song_records(frame, ids):
    columns = ["track_name", "artists", "track_genre"]
    return frame.set_index("track_id").loc[list(ids), columns].reset_index().to_dict("records")


def analyze(frame, scaled, ids_a, ids_b, mode="MANUAL"):
    """Run all visible LA stages using the exact selected song IDs."""
    n = len(frame)
    for ids, name in ((ids_a, "User A"), (ids_b, "User B")):
        if len(ids) < MIN_SELECTION:
            raise ValueError(f"{name} needs at least {MIN_SELECTION} selected songs.")
        if len(ids) > TOP_SONGS:
            raise ValueError(f"{name} has {len(ids)} songs; the maximum is {TOP_SONGS}.")
        if len(set(ids)) != len(ids):
            raise ValueError(f"{name} contains duplicate song selections.")
        if any(not isinstance(i, (int, np.integer)) or i < 0 or i >= n for i in ids):
            raise ValueError(f"{name} selection contains an unknown song ID.")
    # Dataset is n x 7 (one song per row); user matrices are 7 x n_user (songs as columns).
    matrix_a = scaled[np.asarray(ids_a)].T
    matrix_b = scaled[np.asarray(ids_b)].T
    reduced_a, pivots_a, rank_a = rref(matrix_a)
    reduced_b, pivots_b, rank_b = rref(matrix_b)
    # Pivot columns identify an independent basis for each complete selected-song span.
    basis_a = matrix_a[:, pivots_a]
    basis_b = matrix_b[:, pivots_b]
    # Center around each user's mean before covariance/PCA. PCA variation directions are
    # kept distinct from the explicit Gram-Schmidt bases built from pivot song vectors.
    samples_a, samples_b = matrix_a.T, matrix_b.T
    centered_a = samples_a - samples_a.mean(axis=0)
    centered_b = samples_b - samples_b.mean(axis=0)
    cov_a = np.atleast_2d(np.cov(samples_a, rowvar=False, ddof=1))
    cov_b = np.atleast_2d(np.cov(samples_b, rowvar=False, ddof=1))
    eig_a, vec_a = np.linalg.eigh(cov_a)
    eig_b, vec_b = np.linalg.eigh(cov_b)
    order_a, order_b = np.argsort(eig_a)[::-1], np.argsort(eig_b)[::-1]
    eig_a, vec_a = eig_a[order_a], vec_a[:, order_a]
    eig_b, vec_b = eig_b[order_b], vec_b[:, order_b]
    # Supporting full orthonormal span and the first three basis-vector taste model.
    full_gs_a = gram_schmidt(basis_a)
    full_gs_b = gram_schmidt(basis_b)
    taste_a = gram_schmidt(basis_a[:, :TASTE_DIM])
    taste_b = gram_schmidt(basis_b[:, :TASTE_DIM])
    if taste_a.shape[1] != TASTE_DIM or taste_b.shape[1] != TASTE_DIM:
        raise ValueError("Selected songs do not provide three independent taste directions. Select a different set.")
    # Projection of each user mean onto their own taste space; LS coefficients solve min ||Q c - x||.
    mean_a, mean_b = samples_a.mean(axis=0), samples_b.mean(axis=0)
    projection_a = taste_a @ (taste_a.T @ mean_a)
    projection_b = taste_b @ (taste_b.T @ mean_b)
    coeff_a, *_ = np.linalg.lstsq(taste_a, mean_a, rcond=None)
    coeff_b, *_ = np.linalg.lstsq(taste_b, mean_b, rcond=None)

    cross = taste_a.T @ taste_b
    left, singular, right_t = np.linalg.svd(cross, full_matrices=False)
    singular = np.clip(singular, 0.0, 1.0)
    angles = np.degrees(np.arccos(singular))
    principal_a = taste_a @ left
    principal_b = taste_b @ right_t.T
    # Keep each corresponding principal-vector pair in the same orientation.
    for column in range(principal_a.shape[1]):
        if np.dot(principal_a[:, column], principal_b[:, column]) < 0:
            principal_b[:, column] *= -1
    shared = gram_schmidt(principal_a + principal_b)
    if shared.shape[1] == 0:
        shared = (principal_a[:, :1] + principal_b[:, :1])
        shared /= max(np.linalg.norm(shared), RREF_TOL)
    shared_details = []
    for i in range(len(singular)):
        direction = principal_a[:, i] + principal_b[:, i]
        direction_norm = np.linalg.norm(direction)
        if direction_norm > RREF_TOL:
            direction = direction / direction_norm
        shared_details.append({
            "direction": i + 1,
            "angle_degrees": round(float(angles[i]), 2),
            "overlap_strength": round(float(singular[i]), 4),
            "aligned_shared_vector": _json_matrix(direction),
        })

    # Projection-energy fits in [0,1] are combined across both users and shared span.
    candidates = np.asarray(scaled, dtype=float)
    squared_norms = np.maximum(np.sum(candidates ** 2, axis=1), RREF_TOL)
    fit_a = np.sum((taste_a.T @ candidates.T) ** 2, axis=0) / squared_norms
    fit_b = np.sum((taste_b.T @ candidates.T) ** 2, axis=0) / squared_norms
    shared_fit = np.sum((shared.T @ candidates.T) ** 2, axis=0) / squared_norms
    blend = 0.4 * fit_a + 0.4 * fit_b + 0.2 * shared_fit
    projection_error = np.linalg.norm(candidates - (candidates @ shared) @ shared.T, axis=1)
    used_ids = set(ids_a) | set(ids_b)
    ranked = sorted((i for i in range(n) if i not in used_ids), key=lambda i: (-blend[i], i))[:10]
    playlist = []
    for position, song_id in enumerate(ranked, 1):
        song = _song_records(frame, [song_id])[0]
        song.update(
            rank=position,
            blend_score=round(float(blend[song_id]), 4),
            fit_to_A=round(float(fit_a[song_id]), 4),
            fit_to_B=round(float(fit_b[song_id]), 4),
            shared_fit=round(float(shared_fit[song_id]), 4),
            projection_error=round(float(projection_error[song_id]), 4),
        )
        playlist.append(song)

    # A single global PCA coordinate system, fitted from every standardized dataset song.
    visualization_center = candidates.mean(axis=0)
    centered_dataset = candidates - visualization_center
    _, _, visualization_axes = np.linalg.svd(centered_dataset, full_matrices=False)
    visualization_axes = visualization_axes[:3].T
    all_coordinates = centered_dataset @ visualization_axes

    def coordinate_records(ids):
        records = _song_records(frame, ids)
        for record in records:
            coordinates = all_coordinates[int(record["track_id"])]
            record.update(PC1=round(float(coordinates[0]), 4), PC2=round(float(coordinates[1]), 4), PC3=round(float(coordinates[2]), 4))
        return records

    selected_union = set(ids_a) | set(ids_b)
    pca_a = coordinate_records(ids_a)
    pca_b = coordinate_records(ids_b)
    pca_playlist = coordinate_records(ranked)
    raw_feature_mean_a = frame.iloc[list(ids_a)][FEATURES].mean().to_numpy(dtype=float)
    raw_feature_mean_b = frame.iloc[list(ids_b)][FEATURES].mean().to_numpy(dtype=float)
    display_feature_mean_a = samples_a.mean(axis=0)
    display_feature_mean_b = samples_b.mean(axis=0)
    mean_norm_a, mean_norm_b = np.linalg.norm(mean_a), np.linalg.norm(mean_b)
    cosine_baseline = float(np.dot(mean_a, mean_b) / (mean_norm_a * mean_norm_b)) if mean_norm_a > RREF_TOL and mean_norm_b > RREF_TOL else 0.0
    cosine_baseline = float(np.clip(cosine_baseline, -1.0, 1.0))

    def user_result(ids, raw_matrix, reduced, pivots, rank, basis, full_q, q, cov, values, eigenvectors, mean, projection, coefficients):
        residual = mean - projection
        return {
            "songs": _song_records(frame, ids), "count": len(ids),
            "matrix_shape": list(raw_matrix.shape), "song_matrix": _json_matrix(raw_matrix), "rank": int(rank),
            "nullity": int(raw_matrix.shape[1] - rank), "pivot_columns": [int(i + 1) for i in pivots],
            "independent_song_basis": _json_matrix(basis), "rref": _json_matrix(reduced),
            "full_orthonormal_span": _json_matrix(full_q),
            "taste_basis": _json_matrix(q), "orthogonality_check": _json_matrix(q.T @ q),
            "covariance": _json_matrix(cov), "eigenvalues": _json_matrix(values),
            "eigenvectors": _json_matrix(eigenvectors),
            "diagonalization_reconstruction_error": float(np.linalg.norm(cov - eigenvectors @ np.diag(values) @ eigenvectors.T)),
            "pca_explained_variance_ratio": _json_matrix(values[:TASTE_DIM] / max(float(values.sum()), RREF_TOL)),
            "mean_vector": _json_matrix(mean), "projection": _json_matrix(projection),
            "least_squares_coefficients": _json_matrix(coefficients),
            "projection_residual": _json_matrix(residual),
            "projection_residual_norm": float(np.linalg.norm(residual)),
        }

    compatibility = float(np.mean(singular) * 100.0)
    mean_angle = float(np.mean(angles))
    if compatibility >= COMPATIBILITY_THRESHOLDS["very_high"]:
        interpretation = "Very high geometric compatibility"
    elif compatibility >= COMPATIBILITY_THRESHOLDS["high"]:
        interpretation = "High geometric compatibility"
    elif compatibility >= COMPATIBILITY_THRESHOLDS["moderate"]:
        interpretation = "Moderate geometric compatibility"
    else:
        interpretation = "Low geometric compatibility"
    return {
        "mode": mode, "feature_order": FEATURES, "taste_dimension": TASTE_DIM,
        "user_a": user_result(ids_a, matrix_a, reduced_a, pivots_a, rank_a, basis_a, full_gs_a, taste_a, cov_a, eig_a, vec_a, mean_a, projection_a, coeff_a),
        "user_b": user_result(ids_b, matrix_b, reduced_b, pivots_b, rank_b, basis_b, full_gs_b, taste_b, cov_b, eig_b, vec_b, mean_b, projection_b, coeff_b),
        "principal_angles": _json_matrix(angles), "singular_values": _json_matrix(singular),
        "cross_basis_matrix": _json_matrix(cross), "shared_directions": _json_matrix(shared),
        "shared_direction_details": shared_details,
        "compatibility": round(compatibility, 2), "playlist": playlist,
        "compatibility_interpretation": interpretation,
        "mean_principal_angle": round(mean_angle, 2),
        "baseline_cosine": round(cosine_baseline, 4),
        "baseline_percent": round((cosine_baseline + 1.0) * 50.0, 2),
        "baseline_label": "Average-vector cosine baseline",
        "principal_angle_table": [
            {"direction": i + 1, "singular_value": round(float(singular[i]), 4), "angle_degrees": round(float(angles[i]), 2), "overlap_strength": round(float(singular[i]), 4)}
            for i in range(len(singular))
        ],
        "pca_explained_variance": {
            "user_a": _json_matrix(eig_a[:TASTE_DIM] / max(float(eig_a.sum()), RREF_TOL)),
            "user_b": _json_matrix(eig_b[:TASTE_DIM] / max(float(eig_b.sum()), RREF_TOL)),
            "eigenvalues_a": _json_matrix(eig_a[:TASTE_DIM]),
            "eigenvalues_b": _json_matrix(eig_b[:TASTE_DIM]),
        },
        "pca_visualization": {"center": _json_matrix(visualization_center), "axes": _json_matrix(visualization_axes), "user_a": pca_a, "user_b": pca_b, "playlist": pca_playlist},
        "feature_comparison": {"labels": FEATURES, "raw_mean_a": _json_matrix(raw_feature_mean_a), "raw_mean_b": _json_matrix(raw_feature_mean_b), "display_mean_a": _json_matrix(display_feature_mean_a), "display_mean_b": _json_matrix(display_feature_mean_b), "display_label": "Dataset-standardized feature means (display only)"},
        "used_track_ids": sorted(selected_union), "playlist_count": len(playlist),
    }


def startup_data():
    """Load data once and fit one global scaler for users and all candidates."""
    frame, duplicate_count, quality = load_dataset()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(frame[FEATURES].to_numpy(dtype=float))
    frame["track_id"] = frame["track_id"].astype(int)
    scores_a = sum(USER_A_WEIGHTS[f] * scaled[:, FEATURES.index(f)] for f in USER_A_WEIGHTS)
    scores_b = sum(USER_B_WEIGHTS[f] * scaled[:, FEATURES.index(f)] for f in USER_B_WEIGHTS)
    # Stable tie-breaking by original dataset row ID.
    auto_a = sorted(range(len(frame)), key=lambda i: (-scores_a[i], i))[:TOP_SONGS]
    auto_b = sorted(range(len(frame)), key=lambda i: (-scores_b[i], i))[:TOP_SONGS]
    return frame, scaled, duplicate_count, quality, auto_a, auto_b


if __name__ == "__main__":
    data, standardized, duplicates, quality, automatic_a, automatic_b = startup_data()
    print(f"Loaded {len(data)} songs; {data['track_genre'].nunique()} genres; {duplicates} duplicate rows.")
    result = analyze(data, standardized, automatic_a, automatic_b, mode="AUTOMATIC")
    print(f"Automatic profile compatibility: {result['compatibility']}%")
    print("Principal angles:", result["principal_angles"])
