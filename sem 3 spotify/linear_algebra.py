import numpy as np


def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na < 1e-12 or nb < 1e-12:
        return 0.0

    return float(np.dot(a, b) / (na * nb))

def minmax_score(values):
    """
    Convert an array into a 0-1 score where larger means better.
    Constant arrays become 1 for all entries.
    """
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return values

    lo = np.min(values)
    hi = np.max(values)

    if hi - lo < 1e-12:
        return np.ones_like(values)

    return (values - lo) / (hi - lo)

def rref(matrix, tol=1e-10):
    """
    Reduced Row Echelon Form using elementary row operations.

    Included as supporting linear-algebra analysis.
    For the actual numerical PCA/subspace calculation,
    NumPy/SVD/PCA are more numerically appropriate.
    """
    A = np.array(matrix, dtype=float, copy=True)

    if A.size == 0:
        return A

    rows, cols = A.shape
    pivot_row = 0

    for col in range(cols):
        if pivot_row >= rows:
            break

        # Find the row with the largest pivot candidate.
        pivot = pivot_row + np.argmax(np.abs(A[pivot_row:, col]))

        if abs(A[pivot, col]) < tol:
            continue

        # Swap into pivot position.
        A[[pivot_row, pivot]] = A[[pivot, pivot_row]]

        # Normalize pivot row.
        A[pivot_row] = A[pivot_row] / A[pivot_row, col]

        # Eliminate this column from every other row.
        for r in range(rows):
            if r != pivot_row and abs(A[r, col]) > tol:
                A[r] = A[r] - A[r, col] * A[pivot_row]

        pivot_row += 1

    A[np.abs(A) < tol] = 0.0
    return A

def gram_schmidt(vectors, tol=1e-10):
    """
    Classical Gram-Schmidt orthonormalization.

    Input:
        vectors = matrix whose columns are vectors.

    Output:
        Q = orthonormal independent columns.
    """
    V = np.asarray(vectors, dtype=float)

    if V.ndim == 1:
        V = V.reshape(-1, 1)

    Q = []

    for j in range(V.shape[1]):
        v = V[:, j].copy()

        for q in Q:
            v = v - np.dot(q, v) * q

        norm = np.linalg.norm(v)

        if norm > tol:
            Q.append(v / norm)

    if not Q:
        return np.empty((V.shape[0], 0))

    return np.column_stack(Q)

def principal_angles(QA, QB):
    """
    Calculate principal angles between two orthonormal bases.

    M = QA^T QB
    M = U Sigma V^T
    sigma_i = cos(theta_i)
    """
    M = QA.T @ QB

    U, singular_values, Vt = np.linalg.svd(
        M,
        full_matrices=False
    )

    singular_values = np.clip(singular_values, -1.0, 1.0)

    angles = np.degrees(np.arccos(singular_values))

    return angles, singular_values, M, U, Vt
