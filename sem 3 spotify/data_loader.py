import os
import pandas as pd

DATA_FILE = "music_taste_project_dataset.csv"

FEATURES = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "speechiness",
    "instrumentalness",
    "tempo",
]

DISPLAY_FEATURES = [
    "Danceability",
    "Energy",
    "Valence",
    "Acousticness",
    "Speechiness",
    "Instrumentalness",
    "Tempo",
]

def load_dataset(data_file=DATA_FILE):
    """Load and validate the TasteMatch dataset."""
    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"Could not find: {data_file}. Put the CSV in the same folder as the Python files."
        )

    df = pd.read_csv(data_file)

    required = ["track_name", "artists", "track_genre"] + FEATURES
    missing_required = [col for col in required if col not in df.columns]

    if missing_required:
        raise ValueError(
            "Missing required columns:\n" + "\n".join(missing_required)
        )

    df = df.reset_index(drop=True)

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    return df
