"""Local Flask interface for the music taste linear algebra project."""
from flask import Flask, jsonify, render_template, request

from music_taste import DatasetError, TOP_SONGS, analyze, startup_data

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024

try:
    FRAME, SCALED, DUPLICATE_COUNT, DATA_QUALITY, AUTO_A, AUTO_B = startup_data()
    DATA_ERROR = None
except DatasetError as exc:
    FRAME, SCALED, DUPLICATE_COUNT, DATA_QUALITY, AUTO_A, AUTO_B = None, None, 0, {}, [], []
    DATA_ERROR = str(exc)


@app.get("/")
def index():
    if DATA_ERROR:
        return render_template("index.html", startup_error=DATA_ERROR, stats=None,
                               songs=[], genres=[], auto_a=[], auto_b=[], max_songs=TOP_SONGS)
    records = FRAME[["track_id", "track_name", "artists", "track_genre"]].to_dict("records")
    return render_template(
        "index.html", startup_error=None,
        stats={"songs": len(FRAME), "genres": int(FRAME["track_genre"].nunique()), "duplicates": DUPLICATE_COUNT, **DATA_QUALITY},
        songs=records, genres=sorted(FRAME["track_genre"].unique().tolist()),
        auto_a=AUTO_A, auto_b=AUTO_B, max_songs=TOP_SONGS,
    )


@app.post("/api/analyze")
def analyze_api():
    if DATA_ERROR:
        return jsonify(error=DATA_ERROR), 503
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Send a JSON object with mode and selected song IDs."), 400
    mode = payload.get("mode")
    if mode == "AUTOMATIC":
        ids_a, ids_b = AUTO_A, AUTO_B
    elif mode == "MANUAL":
        ids_a, ids_b = payload.get("user_a"), payload.get("user_b")
        if not isinstance(ids_a, list) or not isinstance(ids_b, list):
            return jsonify(error="Manual mode needs song ID lists for both users."), 400
        if any(type(i) is not int for i in ids_a + ids_b):
            return jsonify(error="Song IDs must be integers."), 400
        if set(ids_a) & set(ids_b):
            return jsonify(error="A song cannot be selected for both users in manual mode."), 400
    else:
        return jsonify(error="Mode must be MANUAL or AUTOMATIC."), 400
    try:
        return jsonify(analyze(FRAME, SCALED, ids_a, ids_b, mode=mode))
    except (ValueError, IndexError) as exc:
        return jsonify(error=str(exc)), 400


if __name__ == "__main__":
    # Debug stays off in the classroom demonstration configuration.
    app.run(host="127.0.0.1", port=5000, debug=False)
