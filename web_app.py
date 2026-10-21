"""
Bird Sound ID - local website
=============================

Starts a small web server on your laptop that connects the web page
(web/index.html) to bird_sound_analyzer.py.

    python3 web_app.py

Then open http://127.0.0.1:5050 (it opens automatically).
Record with the browser's mic or upload a file, and the page shows the answer.
Stop the server with Ctrl+C.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_VENV = os.path.join(HERE, ".venv")
_py = os.path.join(_VENV, "Scripts", "python.exe") if os.name == "nt" else os.path.join(_VENV, "bin", "python")
if os.path.exists(_py) and os.path.realpath(sys.prefix) != os.path.realpath(_VENV):
    # Restart with the project's own Python (has numpy, BirdNET, Flask)
    import subprocess
    try:
        sys.exit(subprocess.call([_py, os.path.abspath(__file__)] + sys.argv[1:]))
    except KeyboardInterrupt:
        sys.exit(130)

import datetime as dt
import threading
import time
import uuid
import webbrowser

import matplotlib
matplotlib.use("Agg")  # draw graphs to files, never open windows from the server

import numpy as np
from flask import Flask, abort, jsonify, request, send_from_directory

import bird_sound_analyzer as bsa

# Locally: 127.0.0.1:5050 (5000 is used by AirPlay on macOS).
# Hosted (e.g. Hugging Face Spaces): set BIRD_HOSTED=1 and PORT.
HOSTED = os.environ.get("BIRD_HOSTED") == "1"
PORT = int(os.environ.get("PORT", 5050))
KEEP_FILES_HOURS = 1  # when hosted, visitors' recordings are deleted after this long
MAX_HOSTED_SECONDS = 30  # free servers have little memory: longer uploads are trimmed
WEB_DIR = os.path.join(HERE, "web")
app = Flask(__name__, static_folder=None)
analysis_lock = threading.Lock()  # one analysis at a time


def num(x, digits=3):
    return None if x is None else round(float(x), digits)


def build_response(res):
    hits = res["birdnet_hits"]
    birdnet_ok = hits is not None

    species = {}
    non_bird_hits = {}
    for d in hits or []:
        bucket = non_bird_hits if bsa.is_birdnet_nonbird(d) else species
        cur = bucket.setdefault(d["common_name"], dict(
            name=d["common_name"], scientific=d["scientific_name"], confidence=0.0, times=[]))
        cur["confidence"] = max(cur["confidence"], float(d["confidence"]))
        cur["times"].append([num(d["start_time"], 1), num(d["end_time"], 1)])
    species = sorted(species.values(), key=lambda x: -x["confidence"])
    non_bird_hits = sorted(non_bird_hits.values(), key=lambda x: -x["confidence"])

    events = res["events"]
    groups = []
    for s in res["sources"]:
        f = s["features"]
        groups.append(dict(
            name=s["name"], notes=len(s["notes"]),
            start=num(s["notes"][0]["start"], 2), end=num(s["notes"][-1]["end"], 2),
            peak_freq=num(f["peak_freq"], 0), duration=num(f["duration"], 3),
            bandwidth=num(f["bandwidth"], 0), shape=f["shape"], rate=num(f["rate"], 2),
            flatness=num(f["flatness"], 2),
            guess=s["best"][0], guess_score=num(s["best"][1], 2),
        ))

    contours = []
    for e in events:
        step = max(1, len(e["track_t"]) // 60)  # keep the payload small
        contours.append(dict(
            id=e["id"], source=e["source"], label=e["bird_label"],
            start=num(e["start"], 3), end=num(e["end"], 3), peak=num(e["peak_freq"], 0),
            shape=e["shape"],
            points=[[num(t, 3), num(fr, 0)] for t, fr in zip(e["track_t"][::step], e["track_f"][::step])],
        ))

    not_bird = sorted({e["bird_label"].replace("not a bird", "").strip(" ()") or "unclassified noise"
                       for e in events if e["bird_label"].startswith("not")})
    n_bird = sum(1 for e in events if e["bird_label"] == "bird")
    n_maybe = sum(1 for e in events if e["bird_label"].startswith("possible"))

    name = res["name"]
    return dict(
        answer=res["answer"], confidence=num(res["confidence"]), answer_source=res["answer_source"],
        birdnet_available=birdnet_ok, species=species, non_bird_hits=non_bird_hits,
        groups=groups, contours=contours, not_bird_sounds=not_bird,
        counts=dict(events=len(events), bird=n_bird, maybe=n_maybe,
                    not_bird=len(events) - n_bird - n_maybe),
        duration=num(res["duration"], 2), sample_rate=res["sample_rate"], name=name,
        files=dict(
            audio=f"/files/recordings/{name}.wav",
            plot=f"/files/results/{os.path.basename(res['plot'])}",
            csv_frames=f"/files/results/{os.path.basename(res['csv_frames'])}",
            csv_events=f"/files/results/{os.path.basename(res['csv_events'])}",
        ),
    )


@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/files/<kind>/<path:filename>")
def files(kind, filename):
    folder = {"recordings": bsa.REC_DIR, "results": bsa.OUT_DIR}.get(kind)
    if folder is None:
        abort(404)
    return send_from_directory(folder, filename)


@app.get("/api/status")
def status():
    return jsonify(birdnet_species=len(bsa.birdnet_species()),
                   birdnet_ready=bsa._BIRDNET is not None, hosted=HOSTED)


@app.get("/api/species")
def species_search():
    q = request.args.get("q", "").strip().lower()
    labels = bsa.birdnet_species()
    hits = [dict(name=c, scientific=s) for s, c in labels if q and q in (s + " " + c).lower()]
    return jsonify(total=len(labels), matches=len(hits), results=hits[:50])


@app.post("/api/analyze")
def analyze():
    upload = request.files.get("audio")
    if upload is None:
        return jsonify(error="No audio received."), 400
    os.makedirs(bsa.REC_DIR, exist_ok=True)
    if HOSTED:
        delete_old_files()
    # random part so visitors can't guess each other's file names
    name = f"web_{dt.datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:12]}"
    path = os.path.join(bsa.REC_DIR, name + ".wav")
    upload.save(path)

    try:
        audio, sr = bsa.load_audio(path)
    except (Exception, SystemExit) as err:
        return jsonify(error=f"Could not read the audio file ({err})."), 400
    if len(audio) < sr * 0.5:
        return jsonify(error="The recording is shorter than half a second - record a bit longer."), 400
    if float(np.max(np.abs(audio))) < 1e-4:
        return jsonify(error="The recording is silent. Check that the browser is allowed to use the "
                             "microphone."), 400

    def coord(key):
        try:
            return float(request.form[key])
        except (KeyError, ValueError):
            return None

    note = None
    if HOSTED and len(audio) > MAX_HOSTED_SECONDS * sr:
        audio = audio[:MAX_HOSTED_SECONDS * sr]
        bsa.save_wav(path, audio, sr)
        note = f"Only the first {MAX_HOSTED_SECONDS} seconds were analysed."

    lat, lon = coord("lat"), coord("lon")
    with analysis_lock:
        res = bsa.analyze(audio, sr, path, show=False, lat=lat, lon=lon)
    out = build_response(res)
    out["note"] = note
    return jsonify(out)


def delete_old_files():
    cutoff = time.time() - KEEP_FILES_HOURS * 3600
    for folder in (bsa.REC_DIR, bsa.OUT_DIR):
        if not os.path.isdir(folder):
            continue
        for fn in os.listdir(folder):
            fp = os.path.join(folder, fn)
            if fn.startswith("web_") and os.path.getmtime(fp) < cutoff:
                try:
                    os.remove(fp)
                except OSError:
                    pass


def warm_up_birdnet():
    """Load BirdNET's model in the background so the first analysis is quick."""
    try:
        with analysis_lock:
            bsa.load_birdnet()
    except Exception:
        pass


if HOSTED:  # under gunicorn, load BirdNET as soon as the server starts
    threading.Thread(target=warm_up_birdnet, daemon=True).start()

if __name__ == "__main__":
    url = f"http://127.0.0.1:{PORT}"
    print(f"\nBird Sound ID is running at {url}\nPress Ctrl+C to stop.\n")
    if not HOSTED:
        threading.Thread(target=warm_up_birdnet, daemon=True).start()
    if "--no-browser" not in sys.argv and not HOSTED:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host="0.0.0.0" if HOSTED else "127.0.0.1", port=PORT, debug=False, threaded=True)
