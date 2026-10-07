# Taste Compatibility and Blended Playlist Using Linear Algebra

A local educational Flask app that models two people's musical preferences as three-dimensional subspaces in a seven-feature song space, compares the subspaces with principal angles, and ranks a blended playlist from the supplied CSV.

## Project structure

- `music_taste.py` — validated data loading, global standardization, synthetic profile selection, numerical RREF, basis selection, Gram–Schmidt, projections, least squares, covariance/eigenanalysis/PCA, SVD, principal angles, compatibility, and playlist ranking.
- `app.py` — local Flask page and validated JSON analysis endpoint.
- `templates/index.html`, `static/style.css`, `static/app.js` — interface, selection controls, and staged math results.
- `music_taste_project_dataset.csv` — runtime data (the app uses its path relative to the Python files).
- `spotify.docx` — supplied standardization reference; not needed at runtime.

## Setup and run

From this project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000>. The server binds to localhost and debug mode is off. No Spotify credentials, database, network connection, or global package installation is needed.

## Using the app

### Manual selection

Choose **Manual selection**, search by track name or artist, optionally filter by genre, and check songs for each person. Selected songs appear as removable chips. Each person must have 5–15 songs; the UI blocks additions beyond 15, prevents a song from being assigned to both people, and the backend validates every ID and count. Select the exact songs first, then choose **Analyze taste**. Those selected song rows, standardized with the one dataset-wide scaler, become the columns of A and B.

### Automatic profiles

Choose **Automatic profiles** to see the deterministic top 15 selected for each synthetic profile. User A scores standardized energy + danceability + valence. User B scores standardized acousticness − energy − danceability. The weights are declared near the top of `music_taste.py`; ties break by dataset row order. These exact selected rows flow through the same `analyze` function as manual selections.

## Mathematical workflow

1. **Data matrix and scaling:** X is n × 7 (songs in rows; features in columns). One `StandardScaler` fits all dataset songs, and the same transformed matrix supplies user inputs and playlist candidates. The results include a feature-by-feature comparison of both users' raw and standardized means.
2. **User matrices:** A and B are 7 × m and 7 × k; songs are columns in the declared feature order.
3. **RREF, rank, nullity:** a visible Gauss–Jordan function uses partial pivoting and tolerance. Pivot columns give independent song columns; rank + nullity equals the number of selected songs.
4. **Full basis and taste approximation:** pivot songs form a basis for the full selected-song span. The first three independent pivot-song vectors are processed using modified Gram–Schmidt to form Q_A and Q_B. The three-dimensional subspace is an explicit modeling approximation, not a claim that the complete song span has rank three.
5. **Projection and least squares:** project each user's mean feature vector using Q Qᵀx̄. `numpy.linalg.lstsq` also calculates the coefficient vector minimizing ||Qc − x̄||₂.
6. **Covariance, eigenvectors, diagonalization, PCA:** covariance uses each user's songs as sample rows and is centered by the sample mean. A symmetric eigensolver gives C = VΛVᵀ; leading eigenvectors and variance ratios report PCA patterns, separately from the Gram–Schmidt taste basis.
7. **SVD and principal angles:** SVD of Q_AᵀQ_B produces singular values σᵢ = cos θᵢ; arccos gives principal angles. Compatibility is the mean singular value as a percentage, with a plain-language range label. The results show each angle, overlap strength, and an aligned shared direction. Shared directions are the corresponding ambient principal vectors averaged and orthonormalized.
8. **Cosine baseline:** cosine similarity between the two globally standardized mean feature vectors provides an intuitive reference beside the subspace compatibility score. Its display is remapped to 0–100 for easier reading; it is labeled as a baseline, not a probability.
9. **Common PCA plots:** one visualization PCA is fit to the full globally standardized dataset. The UI plots both selected sets and the recommended songs in the same PC1–PC2 coordinate system, and compares each user's PC1–PC3 explained-variance ratios.
10. **Playlist:** every dataset candidate is scored using its squared projection fraction into each user's space and the shared space: 0.4 A + 0.4 B + 0.2 shared. The 10 best candidates, with stable tie-breaking, show their fit to each user, shared-space fit, and residual distance from the shared-space projection. Already selected profile songs are omitted.

## Viva notes: concept → purpose → outcome

| Concept | Purpose | Outcome |
| --- | --- | --- |
| Feature matrix | Convert songs into points in one common space. | Each row is one song and each of seven columns is a feature. |
| Standardization | Remove unit-scale dominance, especially BPM. | Each feature contributes according to its dataset variation. |
| RREF / Gaussian elimination | Expose pivot columns and dependencies. | Reduced matrix, rank, nullity, and independent selected songs. |
| Basis / linear independence | Describe the complete span without redundant vectors. | Independent song columns span the full user space. |
| Gram–Schmidt | Turn three selected independent directions into orthonormal axes. | Q_A and Q_B define the modeled taste subspaces. |
| Projection / least squares | Approximate a mean preference vector within its taste space. | A projection plus coefficients and residual size. |
| Covariance / eigenanalysis | Find directions of greatest feature variation. | Eigenvalues, eigenvectors, diagonalization, and PCA ratios. |
| SVD / principal angles | Compare two spaces direction by direction. | A detailed angle table, overlap strengths, and mean-cosine compatibility score. |
| Cosine baseline | Compare the users' mean feature vectors directly. | An intuitive reference alongside the subspace score. |
| PCA visualization | Put both users and playlist tracks into one common view. | Shared PC1–PC2 plots and a PC1–PC3 variance comparison. |
| Shared directions / playlist | Find candidates represented in both taste spaces. | An explainable ranked list of ten existing songs. |

## Assumptions and limitations

- Three dimensions are chosen to make a useful low-dimensional comparison. The complete user-song span may have rank up to seven.
- The taste basis is formed from the first three independent pivot song columns, so its exact orientation depends on the selected song order. It is a transparent classroom model, not a learned psychological model.
- Automatic profiles are synthetic equations, not real listening histories.
- PCA describes within-user variation and is reported separately; it is not substituted for the Gram–Schmidt comparison basis.
- Compatibility is a geometric mean-cosine score in percent, not a probability or a learned recommendation confidence.
- The dataset is small and the candidate score is a simple projection-energy blend. It does not use popularity, lyrics, streaming APIs, or real user feedback.

## Backend endpoint

`POST /api/analyze` accepts JSON. Manual example: `{"mode":"MANUAL","user_a":[0,1,2,3,4],"user_b":[5,6,7,8,9]}`. Automatic mode uses `{"mode":"AUTOMATIC"}`. Invalid modes, malformed IDs, out-of-range IDs, and invalid selection counts return a readable 400 error; dataset startup problems are shown on the page.
