"""
Bird Sound Analyzer
===================

Record bird sounds from the laptop mic (or load a .wav file), then:
  1. Find every separate sound "event" in time AND frequency (so two birds
     calling at the same moment in different pitch bands are kept apart).
  2. Decide whether each event is a bird or not (voice, hum, claps, noise...).
  3. Group events into "sources" (Bird A, Bird B, ...) by how similar they sound.
  4. Identify each source by narrowing step by step:
        peak frequency -> duration -> bandwidth -> pitch shape -> call rate -> tonality
     Each step uses one more parameter, until one candidate is left.
  5. Plot waveform, spectrogram, frequency-vs-time and the spectrum.
  6. Save the time -> frequency data to CSV.

Optional: if `birdnetlib` is installed, BirdNET (a trained neural network for
~6000 species) is also run for a second opinion.

Usage
-----
    python bird_sound_analyzer.py                       # interactive menu
    python bird_sound_analyzer.py record --seconds 10   # record + analyze
    python bird_sound_analyzer.py analyze my_clip.wav   # analyze a file
    python bird_sound_analyzer.py demo                  # synthetic test, no mic
    python bird_sound_analyzer.py learn "House Sparrow" clip.wav
                                                        # add your own species profile
    python bird_sound_analyzer.py list                  # show species database

Install:  pip install numpy scipy matplotlib sounddevice
Optional: pip install birdnetlib   (needs a Python version TensorFlow supports)
"""

import argparse
import contextlib
import csv
import datetime as dt
import io
import json
import os
import sys
import warnings

_VENV = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv")
_py = os.path.join(_VENV, "Scripts", "python.exe") if os.name == "nt" else os.path.join(_VENV, "bin", "python")
if os.path.exists(_py) and os.path.realpath(sys.prefix) != os.path.realpath(_VENV):
    # Restart with the project's own Python so numpy/BirdNET are always available
    import subprocess
    try:
        sys.exit(subprocess.call([_py, os.path.abspath(__file__)] + sys.argv[1:]))
    except KeyboardInterrupt:
        sys.exit(130)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # hide TensorFlow chatter
warnings.filterwarnings("ignore")

import numpy as np
from scipy import ndimage, signal
from scipy.io import wavfile

HERE = os.path.dirname(os.path.abspath(__file__))
REC_DIR = os.path.join(HERE, "recordings")
OUT_DIR = os.path.join(HERE, "results")
CUSTOM_DB = os.path.join(HERE, "species_db_custom.json")

# ----------------------------------------------------------------------------
# Settings (tune these while testing)
# ----------------------------------------------------------------------------
SAMPLE_RATE = 44100
NFFT = 1024              # ~23 ms window at 44.1 kHz
HOP = 256                # ~5.8 ms between frames
FMIN, FMAX = 150, 12000  # analysis band (Hz)
THRESH_DB = 12           # how far above the background noise a sound must be
MIN_EVENT_SEC = 0.03     # ignore blips shorter than this
MIN_EVENT_BW_HZ = 80     # ignore components narrower than this
MAX_BELOW_LOUDEST_DB = 30  # ignore sounds this much quieter than the loudest one
MIN_NOTES_PER_BIRD = 2   # a "bird" needs at least this many notes to be counted
BIRDNET_MIN_CONF = 0.05  # keep even weak BirdNET guesses so there is always a best answer
NEW_BIRD_UNLIKENESS = 0.75 # a note this different from a bird's usual notes starts a new bird

# Order in which parameters are added while narrowing down candidates
NARROWING_ORDER = ["peak_freq", "duration", "bandwidth", "shape", "rate", "flatness"]

PARAM_LABEL = {
    "peak_freq": "peak frequency (Hz)",
    "duration": "call duration (s)",
    "bandwidth": "bandwidth (Hz)",
    "shape": "pitch shape",
    "rate": "calls per second",
    "flatness": "tonality (0 = pure whistle, 1 = noise)",
}

# ----------------------------------------------------------------------------
# Species reference database
# ----------------------------------------------------------------------------
# APPROXIMATE values for single calls/notes, taken from typical field-guide
# descriptions. They are a starting point, not ground truth: replace or extend
# them with `learn` using recordings of birds in YOUR area.
# Ranges are [min, max]. shape is any of: flat, rising, falling, mixed.
SPECIES_DB = {
    "House Sparrow":          dict(peak_freq=[2800, 5000], duration=[0.08, 0.30], bandwidth=[1500, 5000], shape=["mixed", "falling"], rate=[1.0, 4.0], flatness=[0.15, 0.45]),
    "American Robin":         dict(peak_freq=[1800, 3500], duration=[0.15, 0.50], bandwidth=[600, 2000],  shape=["mixed", "rising", "falling"], rate=[1.0, 3.0], flatness=[0.0, 0.25]),
    "Northern Cardinal":      dict(peak_freq=[2000, 4500], duration=[0.15, 0.60], bandwidth=[1500, 4000], shape=["falling", "rising"], rate=[1.0, 4.0], flatness=[0.0, 0.20]),
    "Black-capped Chickadee": dict(peak_freq=[3000, 4300], duration=[0.25, 0.60], bandwidth=[100, 700],   shape=["falling", "flat"], rate=[0.5, 1.5], flatness=[0.0, 0.15]),
    "Mourning Dove":          dict(peak_freq=[400, 750],   duration=[0.30, 1.50], bandwidth=[80, 400],    shape=["flat", "rising", "mixed"], rate=[0.3, 1.5], flatness=[0.0, 0.25]),
    "Rock Pigeon":            dict(peak_freq=[300, 650],   duration=[0.30, 1.20], bandwidth=[80, 350],    shape=["flat", "mixed"], rate=[0.3, 1.5], flatness=[0.05, 0.35]),
    "Great Horned Owl":       dict(peak_freq=[200, 450],   duration=[0.20, 0.90], bandwidth=[80, 250],    shape=["flat"], rate=[0.4, 1.5], flatness=[0.0, 0.25]),
    "Common Myna":            dict(peak_freq=[1000, 4000], duration=[0.10, 0.50], bandwidth=[1000, 4000], shape=["mixed", "rising", "falling"], rate=[1.0, 4.0], flatness=[0.15, 0.50]),
    "Asian Koel":             dict(peak_freq=[1000, 2000], duration=[0.25, 0.70], bandwidth=[300, 1200],  shape=["rising"], rate=[0.4, 1.5], flatness=[0.0, 0.20]),
    "Crow (House/American)":  dict(peak_freq=[800, 2200],  duration=[0.20, 0.55], bandwidth=[1000, 3500], shape=["flat", "mixed"], rate=[0.5, 2.0], flatness=[0.35, 0.80]),
    "Rose-ringed Parakeet":   dict(peak_freq=[2000, 5000], duration=[0.08, 0.30], bandwidth=[2000, 5000], shape=["mixed", "falling"], rate=[2.0, 6.0], flatness=[0.30, 0.70]),
    "Blue Jay":               dict(peak_freq=[1500, 3500], duration=[0.20, 0.50], bandwidth=[1500, 3500], shape=["mixed", "falling"], rate=[0.8, 2.5], flatness=[0.30, 0.70]),
    "European Robin":         dict(peak_freq=[3000, 7000], duration=[0.03, 0.20], bandwidth=[1500, 5000], shape=["mixed", "rising", "falling"], rate=[4.0, 10.0], flatness=[0.0, 0.25]),
    "Song Sparrow":           dict(peak_freq=[2000, 6000], duration=[0.04, 0.30], bandwidth=[1000, 4000], shape=["mixed", "falling"], rate=[3.0, 10.0], flatness=[0.05, 0.35]),
    "Common Tailorbird":      dict(peak_freq=[2500, 4500], duration=[0.04, 0.15], bandwidth=[500, 2000],  shape=["rising", "mixed"], rate=[3.0, 6.0], flatness=[0.0, 0.25]),
    "Coppersmith Barbet":     dict(peak_freq=[600, 950],   duration=[0.04, 0.12], bandwidth=[80, 300],    shape=["flat"], rate=[1.5, 3.0], flatness=[0.0, 0.25]),
    "Red-vented Bulbul":      dict(peak_freq=[1500, 3500], duration=[0.10, 0.40], bandwidth=[800, 2500],  shape=["mixed", "rising"], rate=[1.0, 4.0], flatness=[0.05, 0.30]),
    "Barn Swallow":           dict(peak_freq=[3000, 8000], duration=[0.03, 0.15], bandwidth=[1500, 5000], shape=["mixed"], rate=[4.0, 12.0], flatness=[0.10, 0.40]),
}


def load_species_db():
    db = dict(SPECIES_DB)
    if os.path.exists(CUSTOM_DB):
        with open(CUSTOM_DB) as f:
            db.update(json.load(f))  # your learned profiles override built-ins
    return db


# ----------------------------------------------------------------------------
# Audio in / out
# ----------------------------------------------------------------------------
def record_audio(seconds, sr=SAMPLE_RATE):
    try:
        import sounddevice as sd
    except ImportError:
        sys.exit("sounddevice is not installed: pip install sounddevice")
    print(f"\nRecording {seconds:.0f} s from the default microphone... (make the sound now)")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    print("Recording finished.")
    audio = audio[:, 0]
    if np.max(np.abs(audio)) < 1e-4:
        print("WARNING: the recording is silent. On macOS, allow microphone access for your "
              "terminal app in System Settings > Privacy & Security > Microphone.")
    os.makedirs(REC_DIR, exist_ok=True)
    path = os.path.join(REC_DIR, f"rec_{dt.datetime.now():%Y%m%d_%H%M%S}.wav")
    save_wav(path, audio, sr)
    print(f"Saved recording: {path}")
    return audio, sr, path


def save_wav(path, audio, sr):
    pcm = np.clip(audio, -1, 1)
    wavfile.write(path, sr, (pcm * 32767).astype(np.int16))


def load_audio(path):
    try:
        sr, data = wavfile.read(path)
    except ValueError:
        try:
            import soundfile as sf  # handles flac/ogg and odd wav formats
        except ImportError:
            sys.exit("Could not read file. Use a .wav file, or: pip install soundfile")
        data, sr = sf.read(path, dtype="float32")
    if data.dtype.kind == "i":
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    elif data.dtype.kind == "u":
        data = (data.astype(np.float32) - 128) / 128
    data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


# ----------------------------------------------------------------------------
# Spectrogram and event detection
# ----------------------------------------------------------------------------
def compute_spectrogram(audio, sr):
    # High-pass to drop mains hum / handling rumble before analysis
    sos = signal.butter(4, FMIN, btype="highpass", fs=sr, output="sos")
    audio = signal.sosfiltfilt(sos, audio)
    f, t, Z = signal.stft(audio, fs=sr, nperseg=NFFT, noverlap=NFFT - HOP, boundary=None)
    S = np.abs(Z) ** 2
    band = (f >= FMIN) & (f <= min(FMAX, sr / 2))
    return f[band], t, S[band]


def detect_events(f, t, S):
    """Find blobs in the spectrogram that rise above the background noise.
    Working on the 2-D time x frequency image (not just loudness over time)
    is what lets overlapping birds in different pitch bands be separated."""
    S_db = 10 * np.log10(S + 1e-12)
    noise_floor = np.median(S_db, axis=1, keepdims=True)  # per-frequency background
    smooth = ndimage.gaussian_filter(S_db, sigma=(1.5, 2))  # blur speckle so one note stays one blob
    mask = smooth > noise_floor + THRESH_DB
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
    mask = ndimage.binary_closing(mask, structure=np.ones((5, 9)))  # bridge gaps inside a note
    labels, n = ndimage.label(mask)

    dt_frame = t[1] - t[0] if len(t) > 1 else HOP / SAMPLE_RATE
    df = f[1] - f[0]
    events = []
    for idx, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        fs_, ts_ = sl
        comp = labels[sl] == idx
        duration = (ts_.stop - ts_.start) * dt_frame
        bw_bins = (fs_.stop - fs_.start) * df
        if duration < MIN_EVENT_SEC or bw_bins < MIN_EVENT_BW_HZ:
            continue
        ev = measure_event(f, t, S, fs_, ts_, comp, dt_frame)
        events.append(ev)

    events = merge_harmonics(events)
    # Drop faint scraps far below the loudest sound (echoes, distant noise)
    if events:
        loudest = max(e["energy_db"] for e in events)
        events = [e for e in events if e["energy_db"] > loudest - MAX_BELOW_LOUDEST_DB]
    events.sort(key=lambda e: (e["start"], e["peak_freq"]))
    for i, e in enumerate(events, 1):
        e["id"] = i
    return events, S_db, noise_floor


def measure_event(f, t, S, fs_, ts_, comp, dt_frame):
    sub = S[fs_, ts_] * comp
    freqs = f[fs_]
    times = t[ts_]

    spectrum = sub.sum(axis=1)
    peak_freq = float(freqs[np.argmax(spectrum)])

    # Bandwidth: frequencies holding the central 90 % of the event's energy
    cum = np.cumsum(spectrum) / spectrum.sum()
    fmin = float(freqs[np.searchsorted(cum, 0.05)])
    fmax = float(freqs[min(np.searchsorted(cum, 0.95), len(freqs) - 1)])

    # Dominant frequency per frame = the pitch contour
    active = sub.sum(axis=0) > 0
    track_t = times[active]
    track_f = freqs[np.argmax(sub[:, active], axis=0)]
    shape, slope = pitch_shape(track_t, track_f, peak_freq)

    # Tonality: spectral flatness of the full-band spectrum during the event
    full = S[:, ts_].mean(axis=1)
    lo, hi = max(fs_.start - 5, 0), min(fs_.stop + 5, S.shape[0])
    band = full[lo:hi] + 1e-20
    flatness = float(np.exp(np.mean(np.log(band))) / np.mean(band))

    return dict(
        start=float(times[0]), end=float(times[-1] + dt_frame),
        duration=float((ts_.stop - ts_.start) * dt_frame),
        peak_freq=peak_freq, min_freq=fmin, max_freq=fmax,
        bandwidth=max(fmax - fmin, float(f[1] - f[0])),
        shape=shape, slope_hz_per_s=slope, flatness=flatness,
        energy_db=float(10 * np.log10(sub.sum() + 1e-12)),
        harmonic_voice=voice_harmonicity(f, full),
        track_t=track_t, track_f=track_f,
        box=(fs_, ts_),
    )


def pitch_shape(track_t, track_f, peak_freq):
    if len(track_f) < 3:
        return "flat", 0.0
    track_f = signal.medfilt(track_f, 3)
    span = np.percentile(track_f, 95) - np.percentile(track_f, 5)
    slope = float(np.polyfit(track_t, track_f, 1)[0])
    if span < 0.08 * peak_freq:
        return "flat", slope
    r = np.corrcoef(track_t, track_f)[0, 1]
    if r > 0.7:
        return "rising", slope
    if r < -0.7:
        return "falling", slope
    return "mixed", slope


def voice_harmonicity(f, spectrum):
    """Human voice = a stack of harmonics on a low fundamental (80-350 Hz).
    Returns the best harmonic-to-average energy ratio."""
    band = (f >= 80) & (f <= 3000)
    if band.sum() < 10:
        return 0.0
    fb, sb = f[band], spectrum[band]
    mean_e = sb.mean() + 1e-20
    best = 0.0
    for f0 in np.arange(80, 350, 5):
        harm = [sb[np.argmin(np.abs(fb - k * f0))] for k in range(1, 6) if k * f0 < 3000]
        best = max(best, float(np.mean(harm) / mean_e))
    return best


def merge_harmonics(events):
    """A whistle often shows up as a fundamental plus fainter copies at 2x, 3x.
    Merge those copies into the main event instead of calling them new birds."""
    keep = []
    for e in sorted(events, key=lambda x: -x["energy_db"]):
        is_harm = False
        for k in keep:
            overlap = min(e["end"], k["end"]) - max(e["start"], k["start"])
            if overlap < 0.6 * min(e["duration"], k["duration"]):
                continue
            ratio = e["peak_freq"] / k["peak_freq"]
            if ratio > 1.5 and abs(ratio - round(ratio)) < 0.08 and e["energy_db"] < k["energy_db"]:
                is_harm = True
                break
        if not is_harm:
            keep.append(e)
    return keep


# ----------------------------------------------------------------------------
# Bird or not?
# ----------------------------------------------------------------------------
def classify_bird_or_not(e):
    score, reasons = 0.0, []
    pf = e["peak_freq"]
    if 1000 <= pf <= 10000:
        score += 0.40; reasons.append("pitch in main bird range (1-10 kHz)")
    elif 200 <= pf < 1000:
        score += 0.15; reasons.append("low pitch (doves/owls, but also voices)")
    else:
        reasons.append("pitch outside usual bird range")

    if e["flatness"] < 0.25:
        score += 0.25; reasons.append("tonal")
    elif e["flatness"] < 0.5:
        score += 0.10; reasons.append("somewhat harsh")
    else:
        reasons.append("noise-like")

    if 0.02 <= e["duration"] <= 2.0:
        score += 0.15
    else:
        reasons.append("unusual duration for a single call")

    if e["shape"] != "flat" or e["bandwidth"] > 300:
        score += 0.10; reasons.append(f"frequency-modulated ({e['shape']})")

    guess = None
    if e["harmonic_voice"] > 6 and pf < 1500:
        score -= 0.35; guess = "human voice"
    if e["flatness"] > 0.5 and e["bandwidth"] > 5000:
        score -= 0.35; guess = "broadband noise (clap, tap, keyboard, wind, traffic)"
    if e["shape"] == "flat" and e["duration"] > 2.0 and e["bandwidth"] < 150:
        score -= 0.30; guess = "steady tone / machine hum / beep"

    if score >= 0.55:
        label = "bird"
    elif score >= 0.35:
        label = "possible bird"
    else:
        label = "not a bird" + (f" ({guess})" if guess else "")
    e["bird_score"] = round(max(score, 0.0), 2)
    e["bird_label"] = label
    e["bird_reasons"] = reasons
    return e


# ----------------------------------------------------------------------------
# Group events into sources (Bird A, Bird B, ...)
# ----------------------------------------------------------------------------
def build_notes(events):
    """Pieces that overlap in time and sit within an octave are parts of ONE note."""
    parent = list(range(len(events)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(events):
        for j in range(i + 1, len(events)):
            b = events[j]
            if b["start"] > a["end"]:
                continue
            overlaps = min(a["end"], b["end"]) > max(a["start"], b["start"])
            near_pitch = abs(np.log2(a["peak_freq"] / b["peak_freq"])) < 1.0
            # Two full-length sounds at once = two voices; only a small scrap or a
            # piece that starts together with another is part of the same note.
            fragment = min(a["duration"], b["duration"]) < 0.06 or abs(a["start"] - b["start"]) < 0.03
            if overlaps and near_pitch and fragment:
                parent[find(j)] = find(i)

    groups = {}
    for i, e in enumerate(events):
        groups.setdefault(find(i), []).append(e)
    notes = []
    for parts in groups.values():
        main = max(parts, key=lambda x: x["energy_db"])
        notes.append(dict(
            parts=parts,
            start=min(p["start"] for p in parts), end=max(p["end"] for p in parts),
            peak_freq=main["peak_freq"], shape=main["shape"],
            min_freq=min(p["min_freq"] for p in parts), max_freq=max(p["max_freq"] for p in parts),
            flatness=float(np.median([p["flatness"] for p in parts])),
        ))
    for n in notes:
        n["duration"] = n["end"] - n["start"]
        n["bandwidth"] = max(n["max_freq"] - n["min_freq"], 50.0)
    return sorted(notes, key=lambda n: n["start"])


def unlike(note, group):
    """How different a note is from a group's typical note (0 = identical)."""
    pf = np.median([m["peak_freq"] for m in group["notes"]])
    du = np.median([m["duration"] for m in group["notes"]])
    return abs(np.log2(note["peak_freq"] / pf)) + 0.5 * abs(np.log2(note["duration"] / du))


def group_sources(events):
    """One bird sings notes one after another. So a new note joins a bird that is
    free at that moment and singing in a similar pitch region; a note that overlaps
    every compatible bird must come from ANOTHER bird."""
    for e in events:
        e["source"] = None
    bird_events = [e for e in events if not e["bird_label"].startswith("not")]
    notes = build_notes(bird_events)

    groups = []
    for n in notes:
        free = [g for g in groups
                if n["start"] >= g["last_end"] - 0.03
                and abs(np.log2(n["peak_freq"] / np.median([m["peak_freq"] for m in g["notes"]]))) < 1.2]
        free = [g for g in free if unlike(n, g) < NEW_BIRD_UNLIKENESS]
        if free:
            g = min(free, key=lambda g: unlike(n, g))  # the bird whose usual note is most similar
        else:
            g = {"notes": [], "last_end": 0.0}
            groups.append(g)
        g["notes"].append(n)
        g["last_end"] = max(g["last_end"], n["end"])

    sources = []
    for g in groups:
        if len(g["notes"]) < MIN_NOTES_PER_BIRD:
            for n in g["notes"]:  # a lone sound is too little to call a bird
                for p in n["parts"]:
                    if p["bird_label"] == "bird":
                        p["bird_label"] = "possible bird (isolated sound)"
            continue
        name = f"Bird {chr(ord('A') + len(sources))}" if len(sources) < 26 else f"Bird {len(sources) + 1}"
        ns = g["notes"]
        evs = [p for n in ns for p in n["parts"]]
        for p in evs:
            p["source"] = name
        starts = [n["start"] for n in ns]
        span = ns[-1]["end"] - ns[0]["start"]
        shapes = [n["shape"] for n in ns]
        sources.append(dict(
            name=name, notes=ns, events=evs, times=starts,
            features=dict(
                peak_freq=float(np.median([n["peak_freq"] for n in ns])),
                duration=float(np.median([n["duration"] for n in ns])),
                bandwidth=float(np.median([n["bandwidth"] for n in ns])),
                shape=max(set(shapes), key=shapes.count),
                rate=(len(ns) / span) if span > 0 else None,
                flatness=float(np.median([n["flatness"] for n in ns])),
            ),
        ))
    return sources


# ----------------------------------------------------------------------------
# Species identification by step-by-step narrowing
# ----------------------------------------------------------------------------
def param_match(feat, value, spec, tol=0.10):
    """Return closeness 0..1 (1 = inside the reference range)."""
    if value is None or spec is None:
        return None
    if feat == "shape":
        return 1.0 if value in spec else 0.0
    lo, hi = spec
    width = max(hi - lo, 1e-6)
    lo_t, hi_t = lo - tol * width, hi + tol * width
    if lo_t <= value <= hi_t:
        return 1.0
    dist = (lo_t - value) if value < lo_t else (value - hi_t)
    return float(np.exp(-dist / width))


def identify(features, db, verbose=True):
    candidates = list(db.keys())
    steps = []
    for feat in NARROWING_ORDER:
        value = features.get(feat)
        if value is None:
            steps.append((feat, None, candidates, "skipped (not measurable - need 2+ calls)"))
            continue
        passed = [sp for sp in candidates if param_match(feat, value, db[sp].get(feat)) in (1.0, None)]
        if not passed:
            steps.append((feat, value, candidates, "no candidate fits this value - parameter ignored"))
            continue
        candidates = passed
        steps.append((feat, value, candidates, ""))
        if len(candidates) == 1:
            break

    # Rank every species by how well ALL parameters fit (also used if the filter emptied out)
    # Inside a range, values near the centre score higher, which breaks ties
    # between species that all "fit".
    ranking = []
    for sp, spec in db.items():
        scores = []
        for k in NARROWING_ORDER:
            m = param_match(k, features.get(k), spec.get(k))
            if m is None:
                continue
            if m == 1.0 and k != "shape":
                lo, hi = spec[k]
                half = max((hi - lo) / 2, 1e-6)
                m = 1.0 - 0.3 * min(abs(features[k] - (lo + hi) / 2) / half, 1.0)
            scores.append(m)
        ranking.append((sp, float(np.mean(scores)) if scores else 0.0))
    ranking.sort(key=lambda x: -x[1])
    in_cands = [r for r in ranking if r[0] in candidates]

    if verbose:
        for i, (feat, value, cands, note) in enumerate(steps, 1):
            vtxt = "-" if value is None else (value if isinstance(value, str) else f"{value:.2f}")
            print(f"    Step {i}: {PARAM_LABEL[feat]} = {vtxt}  ->  {len(cands)} candidate(s)"
                  + (f"  [{note}]" if note else ""))
            if len(cands) <= 6:
                print(f"            {', '.join(cands)}")
    return in_cands, ranking, steps


def format_features(feat):
    r = "n/a" if feat["rate"] is None else f"{feat['rate']:.2f}/s"
    return (f"peak {feat['peak_freq']:.0f} Hz | dur {feat['duration']:.2f} s | "
            f"bw {feat['bandwidth']:.0f} Hz | {feat['shape']} | rate {r} | flatness {feat['flatness']:.2f}")


# ----------------------------------------------------------------------------
# Optional: BirdNET second opinion
# ----------------------------------------------------------------------------
@contextlib.contextmanager
def silence_everything():
    """Hide messages printed by TensorFlow's C++ code (Python redirects can't catch those)."""
    sys.stdout.flush(); sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1); os.dup2(null, 2)
    try:
        yield
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(saved[0], 1); os.dup2(saved[1], 2)
        for fd in (*saved, null):
            os.close(fd)


_BIRDNET_ANALYZER = None


def run_birdnet(path, lat=None, lon=None, details=False):
    try:
        with silence_everything():
            from birdnetlib import Recording
            from birdnetlib.analyzer import Analyzer
    except ImportError:
        print("\n(BirdNET is not available - using the rule-based method only.)")
        return None
    print("Identifying species with BirdNET...")
    kwargs = dict(min_conf=BIRDNET_MIN_CONF)
    if lat is not None and lon is not None:
        kwargs.update(lat=lat, lon=lon, date=dt.datetime.now())
    with silence_everything():
        global _BIRDNET_ANALYZER
        if _BIRDNET_ANALYZER is None:
            _BIRDNET_ANALYZER = Analyzer()
        rec = Recording(_BIRDNET_ANALYZER, path, **kwargs)
        rec.analyze()
    if details:
        print("\nBirdNET detections:")
        if not rec.detections:
            print(f"  none above {BIRDNET_MIN_CONF * 100:.0f} % confidence")
        for d in rec.detections:
            print(f"  {d['start_time']:5.1f}-{d['end_time']:5.1f} s  {d['common_name']:<28} "
                  f"({d['confidence'] * 100:.0f} %)")
    return rec.detections


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
# Categorical colours in fixed order (one per detected bird source)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
NOT_BIRD = "#8a8a85"


def plot_results(audio, sr, f, t, S_db, events, sources, title, save_path, show=True):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    color_of = {s["name"]: SERIES[i % len(SERIES)] for i, s in enumerate(sources)}
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "font.size": 9})
    fig = plt.figure(figsize=(13, 10), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, width_ratios=[4, 1.3], height_ratios=[1, 2.2, 1.6, 1.4])
    fig.suptitle(title, fontsize=12, fontweight="bold")

    # 1. Waveform
    ax0 = fig.add_subplot(gs[0, 0])
    tt = np.arange(len(audio)) / sr
    ax0.plot(tt, audio, color="#5a5a55", linewidth=0.5)
    ax0.set_ylabel("Amplitude")
    ax0.set_title("Waveform", loc="left")
    ax0.set_xlim(0, tt[-1] if len(tt) else 1)

    # 2. Spectrogram with event boxes
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    vmax = np.percentile(S_db, 99.8)
    ax1.pcolormesh(t, f / 1000, S_db, shading="auto", cmap="magma", vmin=vmax - 70, vmax=vmax)
    for e in events:
        fs_, ts_ = e["box"]
        c = color_of.get(e["source"], NOT_BIRD)
        ls = "-" if e["source"] else "--"
        ax1.add_patch(Rectangle((e["start"], f[fs_.start] / 1000), e["end"] - e["start"],
                                (f[fs_.stop - 1] - f[fs_.start]) / 1000,
                                fill=False, edgecolor=c, linewidth=1.5, linestyle=ls))
        ax1.text(e["start"], f[fs_.stop - 1] / 1000 + 0.15, e["source"].split()[-1] if e["source"] else "x",
                 color=c, fontsize=8, fontweight="bold")
    ax1.set_ylabel("Frequency (kHz)")
    ax1.set_title("Spectrogram (boxes = detected sounds; letter = bird source, x = not a bird)", loc="left")

    # 3. Frequency vs time (pitch contours)
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
    for e in events:
        c = color_of.get(e["source"], NOT_BIRD)
        ax2.plot(e["track_t"], e["track_f"] / 1000, ".", color=c, markersize=3,
                 alpha=1.0 if e["source"] else 0.4)
    for s in sources:
        ax2.plot([], [], "o", color=color_of[s["name"]], label=s["name"])
    if any(e["source"] is None for e in events):
        ax2.plot([], [], "o", color=NOT_BIRD, label="not a bird")
    if events:
        ax2.legend(loc="upper right", fontsize=8, frameon=False, ncol=4)
    ax2.set_ylabel("Dominant freq (kHz)")
    ax2.set_title("Frequency vs time (pitch contour of each sound)", loc="left")

    # 4. Loudness envelope in the bird band
    ax3 = fig.add_subplot(gs[3, 0], sharex=ax0)
    band = (f >= 1000) & (f <= 10000)
    env = 10 * np.log10(np.mean(10 ** (S_db[band] / 10), axis=0) + 1e-12)
    ax3.plot(t, env, color=SERIES[0], linewidth=1.2)
    ax3.set_ylabel("Level (dB)")
    ax3.set_xlabel("Time (s)")
    ax3.set_title("Energy in 1-10 kHz bird band over time", loc="left")

    # Right column: average spectrum + per-source summary
    ax4 = fig.add_subplot(gs[1:3, 1])
    mean_spec = 10 * np.log10(np.mean(10 ** (S_db / 10), axis=1) + 1e-12)
    ax4.plot(mean_spec, f / 1000, color="#5a5a55", linewidth=1.2)
    for s in sources:
        ax4.axhline(s["features"]["peak_freq"] / 1000, color=color_of[s["name"]], linewidth=1.5, linestyle=":")
    ax4.set_xlabel("Power (dB)")
    ax4.set_ylabel("Frequency (kHz)")
    ax4.set_title("Average spectrum", loc="left")

    ax5 = fig.add_subplot(gs[3, 1])
    ax5.axis("off")
    lines = []
    for s in sources:
        best = s.get("best", ("?", 0))
        lines.append(f"{s['name']}: {best[0]} ({best[1] * 100:.0f}%)")
    n_not = sum(1 for e in events if e["source"] is None)
    if n_not:
        lines.append(f"{n_not} non-bird sound(s)")
    ax5.text(0, 1, "\n".join(lines) or "No sounds detected", va="top", fontsize=9)

    fig.savefig(save_path, dpi=130)
    print(f"Saved plot: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ----------------------------------------------------------------------------
# CSV export
# ----------------------------------------------------------------------------
def export_csv(base, f, t, S_db, noise_floor, events):
    # Time -> dominant frequency for every frame of the recording
    frames_path = base + "_time_frequency.csv"
    above = S_db - noise_floor
    with open(frames_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "dominant_freq_hz", "level_db", "db_above_noise", "event_id", "source"])
        owner = {}
        for e in events:
            for tt in e["track_t"]:
                owner[round(float(tt), 5)] = (e["id"], e["source"] or "not a bird")
        for j, tt in enumerate(t):
            i = int(np.argmax(S_db[:, j]))
            eid, src = owner.get(round(float(tt), 5), ("", ""))
            w.writerow([f"{tt:.4f}", f"{f[i]:.1f}", f"{S_db[i, j]:.1f}", f"{above[i, j]:.1f}", eid, src])

    events_path = base + "_events.csv"
    with open(events_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "start_s", "end_s", "duration_s", "peak_freq_hz", "min_freq_hz", "max_freq_hz",
                    "bandwidth_hz", "shape", "slope_hz_per_s", "flatness", "bird_score", "classification", "source"])
        for e in events:
            w.writerow([e["id"], f"{e['start']:.3f}", f"{e['end']:.3f}", f"{e['duration']:.3f}",
                        f"{e['peak_freq']:.0f}", f"{e['min_freq']:.0f}", f"{e['max_freq']:.0f}",
                        f"{e['bandwidth']:.0f}", e["shape"], f"{e['slope_hz_per_s']:.0f}",
                        f"{e['flatness']:.3f}", e["bird_score"], e["bird_label"], e["source"] or ""])
    print(f"Saved data:  {frames_path}\n             {events_path}")


# ----------------------------------------------------------------------------
# Main analysis pipeline
# ----------------------------------------------------------------------------
def confidence_word(score):
    if score >= 0.85:
        return "LIKELY"
    if score >= 0.70:
        return "POSSIBLE"
    return "UNCERTAIN"


def is_birdnet_nonbird(d):
    """BirdNET also knows sounds like 'Human vocal', 'Dog', 'Engine', 'Siren'.
    Those labels have no scientific name (it equals the common name)."""
    return d["scientific_name"] == d["common_name"]


def final_answer(events, sources, birdnet_hits):
    """Pick ONE answer.
    1. BirdNET (6,500+ species) if it is installed.
    2. Otherwise the rule-based match from the small built-in list."""
    if birdnet_hits is not None:
        birds = [d for d in birdnet_hits if not is_birdnet_nonbird(d)]
        others = [d for d in birdnet_hits if is_birdnet_nonbird(d)]
        if birds:
            d = max(birds, key=lambda d: d["confidence"])
            return d["common_name"], d["confidence"], f"BirdNET - {d['scientific_name']}"
        if others and not sources:
            d = max(others, key=lambda d: d["confidence"])
            return f"NOT A BIRD - {d['common_name']}", d["confidence"], "BirdNET"
        if sources:
            return "BIRD DETECTED - SPECIES UNKNOWN", None, \
                "bird-like sounds found, but BirdNET could not match a species; record longer/closer"
    elif sources:
        s = max(sources, key=lambda s: (s["best"][1], len(s["notes"])))
        return s["best"][0], s["best"][1], "rule-based match (BirdNET not available)"
    not_bird = [e["bird_label"] for e in events if e["bird_label"].startswith("not")]
    heard = sorted({x.replace("not a bird", "").strip(" ()") for x in not_bird} - {""})
    return "NO BIRD DETECTED", None, ("heard: " + "; ".join(heard)) if heard else "only background noise"


def print_final_answer(events, sources, birdnet_hits):
    answer, conf, source = final_answer(events, sources, birdnet_hits)
    bar = "=" * 60
    print("\n" + bar)
    print(f"  FINAL ANSWER:  {answer.upper()}")
    if conf is not None:
        print(f"  Confidence:    {conf * 100:.0f}%  ({source})")
        if conf < 0.5:
            print("  Low confidence - record again, closer to the bird and with less noise.")
    else:
        print(f"  ({source})")
    print(bar)


def print_breakdown(events, sources, birdnet_hits):
    """Full breakdown, shown with --details."""
    bar = "#" * 78
    print("\n" + bar)
    print("#" + "DETAILED BREAKDOWN".center(76) + "#")
    print(bar)

    not_bird = sorted({e["bird_label"] for e in events if e["bird_label"].startswith("not")})
    maybe = [e for e in events if e["bird_label"].startswith("possible")]

    if birdnet_hits:
        best = {}
        for d in birdnet_hits:
            name = d["common_name"]
            if name not in best or d["confidence"] > best[name]["confidence"]:
                best[name] = d
        ranked = sorted(best.values(), key=lambda d: -d["confidence"])
        rule_species = {s["best"][0] for s in sources}
        sure = [d for d in ranked if d["confidence"] >= 0.70] or ranked[:1]
        print(f"  Is there a bird?   YES")
        print("  ANSWER:            " + " + ".join(f"{d['common_name'].upper()} ({d['confidence'] * 100:.0f}%)"
                                                   for d in sure))
        print(f"  Species detected:  {len(ranked)}")
        for i, d in enumerate(ranked, 1):
            c = d["confidence"]
            level = "CONFIRMED" if c >= 0.70 else "POSSIBLE " if c >= 0.40 else "WEAK     "
            times = sorted({f"{x['start_time']:.0f}-{x['end_time']:.0f}s" for x in birdnet_hits
                            if x["common_name"] == d["common_name"]})
            agree = "  [rule-based method agrees]" if d["common_name"] in rule_species else ""
            print(f"   {i}. {level} {d['common_name']} ({d['scientific_name']}) - "
                  f"{c * 100:.0f}%, heard at {', '.join(times)}{agree}")
        print("  (Source: BirdNET neural network. Treat POSSIBLE/WEAK as unconfirmed.)")
    elif sources:
        sure = []
        for s in sorted(sources, key=lambda s: -s["best"][1]):
            if s["best"][1] >= 0.85 and s["best"][0] not in sure:
                sure.append(s["best"][0])
        if not sure:
            top = max(sources, key=lambda s: s["best"][1])
            sure = [f"{top['best'][0]} ({confidence_word(top['best'][1]).lower()})"]
        print(f"  Is there a bird?   YES")
        print(f"  ANSWER:            {' + '.join(sure).upper()}")
        print(f"  Distinct call types: {len(sources)}  (could be {len(sources)} birds, or fewer birds "
              "with varied songs)")
        for s in sources:
            sp, sc = s["best"]
            print(f"  {s['name']} ({s['times'][0]:.1f}-{s['notes'][-1]['end']:.1f} s, {len(s['notes'])} notes): "
                  f"{sp} - {confidence_word(sc)} ({sc * 100:.0f}% match)")
            if sc < 0.70:
                print("       -> weak match: this species is probably not in the database yet.")
        print("  (Source: rule-based parameter matching - an educated guess, not a certainty.)")
    elif maybe:
        print("  Is there a bird?   MAYBE - only short or isolated bird-like sounds were found.")
        print("  Try recording longer or closer to the bird.")
    else:
        print("  Is there a bird?   NO")
    if not_bird:
        print("  Non-bird sounds:   " + "; ".join(x.replace("not a bird", "").strip(" ()") or "unclassified noise"
                                               for x in not_bird))
    print(bar)


def analyze(audio, sr, path=None, show=True, birdnet=True, lat=None, lon=None, details=False):
    db = load_species_db()
    f, t, S = compute_spectrogram(audio, sr)
    events, S_db, noise_floor = detect_events(f, t, S)
    for e in events:
        classify_bird_or_not(e)
    sources = group_sources(events)

    if details:
        print("\n" + "=" * 78)
        print(f"Analyzed {len(audio) / sr:.1f} s of audio at {sr} Hz - found {len(events)} sound piece(s)")
        print("=" * 78)
        print(f"{'#':>3} {'time (s)':>13} {'peak Hz':>8} {'dur s':>6} {'bw Hz':>6} {'shape':>8} {'flat':>5}  verdict")
        for e in events:
            print(f"{e['id']:>3} {e['start']:6.2f}-{e['end']:<6.2f} {e['peak_freq']:8.0f} {e['duration']:6.2f} "
                  f"{e['bandwidth']:6.0f} {e['shape']:>8} {e['flatness']:5.2f}  {e['bird_label']}"
                  + (f" -> {e['source']}" if e["source"] else ""))
    for s in sources:
        if details:
            times = ", ".join(f"{x:.2f}" for x in s["times"][:12]) + (" ..." if len(s["times"]) > 12 else "")
            print(f"\n--- {s['name']}: {len(s['notes'])} note(s) at t = {times} s")
            print(f"    {format_features(s['features'])}")
        cands, ranking, _ = identify(s["features"], db, verbose=details)
        top = cands[:3] if cands else ranking[:3]
        s["best"] = top[0] if top else ("Unknown", 0.0)
        if details:
            print("    Ranked guesses: " + "; ".join(f"{sp} ({sc * 100:.0f}% match)" for sp, sc in top))

    birdnet_hits = run_birdnet(path, lat, lon, details) if (birdnet and path) else None

    os.makedirs(OUT_DIR, exist_ok=True)
    name = os.path.splitext(os.path.basename(path))[0] if path else f"clip_{dt.datetime.now():%Y%m%d_%H%M%S}"
    base = os.path.join(OUT_DIR, name)
    with contextlib.redirect_stdout(io.StringIO()) if not details else contextlib.nullcontext():
        export_csv(base, f, t, S_db, noise_floor, events)
    if details:
        print_breakdown(events, sources, birdnet_hits)
    print_final_answer(events, sources, birdnet_hits)
    print(f"(Graphs and data saved in the results folder. Add --details for the full breakdown.)")
    with contextlib.redirect_stdout(io.StringIO()):
        plot_results(audio, sr, f, t, S_db, events, sources, f"Bird sound analysis - {name}",
                     base + "_plot.png", show=show)
    answer, conf, source = final_answer(events, sources, birdnet_hits)
    return dict(events=events, sources=sources, birdnet_hits=birdnet_hits,
                answer=answer, confidence=conf, answer_source=source,
                name=name, plot=base + "_plot.png",
                csv_frames=base + "_time_frequency.csv", csv_events=base + "_events.csv",
                duration=len(audio) / sr, sample_rate=sr)


def learn(species, path):
    """Measure a clean recording of a KNOWN species and store its parameter ranges."""
    audio, sr = load_audio(path)
    f, t, S = compute_spectrogram(audio, sr)
    events, _, _ = detect_events(f, t, S)
    events = [classify_bird_or_not(e) for e in events]
    events = [e for e in events if not e["bird_label"].startswith("not")]
    if not events:
        sys.exit("No bird-like sounds found in that file.")

    def rng(key):
        v = np.array([e[key] for e in events])
        med = float(np.median(v))
        spread = max(float(v.std()) * 2, 0.15 * med)
        return [max(med - spread, 0.0), med + spread]

    starts = sorted(e["start"] for e in events)
    rate = (len(starts) - 1) / (starts[-1] - starts[0]) if len(starts) > 1 and starts[-1] > starts[0] else None
    profile = dict(peak_freq=rng("peak_freq"), duration=rng("duration"), bandwidth=rng("bandwidth"),
                   shape=sorted(set(e["shape"] for e in events)), flatness=rng("flatness"))
    if rate:
        profile["rate"] = [rate * 0.6, rate * 1.5]
    custom = {}
    if os.path.exists(CUSTOM_DB):
        with open(CUSTOM_DB) as fh:
            custom = json.load(fh)
    custom[species] = profile
    with open(CUSTOM_DB, "w") as fh:
        json.dump(custom, fh, indent=2)
    print(f"Learned '{species}' from {len(events)} call(s):")
    print(json.dumps(profile, indent=2))
    print(f"Saved to {CUSTOM_DB}")


def make_demo_audio(sr=SAMPLE_RATE, seconds=8.0):
    """Synthetic scene: two 'birds' overlapping + a human-voice-like sound + a clap."""
    rng = np.random.default_rng(0)
    n = int(sr * seconds)
    x = 0.004 * rng.standard_normal(n)
    tt = np.arange(n) / sr

    def tone(t0, dur, f0, f1, amp):
        i0, i1 = int(t0 * sr), int((t0 + dur) * sr)
        tl = tt[: i1 - i0]
        freq = np.linspace(f0, f1, len(tl))
        phase = 2 * np.pi * np.cumsum(freq) / sr
        env = np.sin(np.pi * np.linspace(0, 1, len(tl))) ** 0.5
        x[i0:i1] += amp * env * np.sin(phase)

    for k in range(5):            # chickadee-like falling whistle, every 1.3 s
        tone(0.5 + 1.3 * k, 0.40, 4000, 3600, 0.25)
    for k in range(14):           # sparrow-ish quick chirps, overlapping in time
        tone(1.0 + 0.32 * k, 0.12, 5200, 3000, 0.20)
    i0 = int(6.4 * sr)            # voice-like: 140 Hz with harmonics
    for h in range(1, 12):
        x[i0:i0 + int(0.7 * sr)] += (0.12 / h) * np.sin(2 * np.pi * 140 * h * tt[: int(0.7 * sr)])
    i0 = int(7.5 * sr)            # clap: short noise burst
    x[i0:i0 + 600] += 0.5 * rng.standard_normal(600) * np.exp(-np.arange(600) / 120)
    return x.astype(np.float32)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def open_graph(png):
    """Open the saved graph in the Preview app so the menu doesn't have to wait for it."""
    if png and os.path.exists(png):
        import subprocess
        if sys.platform == "darwin":
            subprocess.run(["open", png])
        elif os.name == "nt":
            os.startfile(png)
        else:
            subprocess.run(["xdg-open", png])


def birdnet_species():
    """All species names BirdNET can recognise (read from its installed label file)."""
    try:
        with silence_everything():
            import birdnetlib
    except ImportError:
        return []
    root = os.path.dirname(birdnetlib.__file__)
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.endswith("_Labels.txt") and "GLOBAL" in fn:
                with open(os.path.join(dirpath, fn)) as fh:
                    rows = [line.strip().split("_", 1) for line in fh if "_" in line]
                return [(sci, com) for sci, com in rows if sci != com]
    return []


def search_species(term=""):
    labels = birdnet_species()
    print(f"\nBirdNET can recognise {len(labels)} bird species "
          f"(the rule-based method has its own list of {len(load_species_db())}).")
    if not term:
        term = input("Search a bird name (e.g. myna, kingfisher), or press Enter to skip: ").strip()
    if term:
        hits = [(sci, com) for sci, com in labels if term.lower() in (sci + " " + com).lower()]
        for sci, com in hits[:40]:
            print(f"  - {com} ({sci})")
        print(f"  {len(hits)} match(es)" + (" - showing first 40" if len(hits) > 40 else ""))


def menu():
    last_png = None
    while True:
        print("\n=========== Bird Sound Analyzer ===========")
        print("  1) Record from mic and identify")
        print("  2) Analyze a .wav file")
        print("  3) Demo (synthetic test, no mic)")
        print("  4) Open the last graph")
        print("  5) Search the species BirdNET knows")
        print("  q) Quit")
        try:
            choice = input("Choose: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        try:
            if choice == "1":
                secs = float(input("Seconds to record [10]: ") or 10)
                audio, sr, path = record_audio(secs)
                analyze(audio, sr, path, show=False)
            elif choice == "2":
                path = input("Path to .wav (you can drag the file into this window): ").strip().strip("'\"")
                audio, sr = load_audio(path)
                analyze(audio, sr, path, show=False)
            elif choice == "3":
                os.makedirs(REC_DIR, exist_ok=True)
                path = os.path.join(REC_DIR, "demo_scene.wav")
                audio = make_demo_audio()
                save_wav(path, audio, SAMPLE_RATE)
                analyze(audio, SAMPLE_RATE, path, show=False, birdnet=False)
            elif choice == "4":
                if last_png:
                    open_graph(last_png)
                else:
                    print("No graph yet - run 1, 2 or 3 first.")
                continue
            elif choice == "5":
                search_species()
                continue
            elif choice in ("q", "quit", "exit", "6"):
                return
            else:
                print("Please type 1-5 or q.")
                continue
            last_png = os.path.join(OUT_DIR, os.path.splitext(os.path.basename(path))[0] + "_plot.png")
            print("(Type 4 to see the graphs.)")
        except KeyboardInterrupt:
            print("\nCancelled.")
        except (Exception, SystemExit) as err:
            print(f"Error: {err}")


def main():
    p = argparse.ArgumentParser(description="Record and identify bird sounds.")
    sub = p.add_subparsers(dest="cmd")
    r = sub.add_parser("record", help="record from mic then analyze")
    r.add_argument("--seconds", type=float, default=10)
    a = sub.add_parser("analyze", help="analyze a .wav file")
    a.add_argument("file")
    sub.add_parser("demo", help="analyze a synthetic test scene (no mic needed)")
    l = sub.add_parser("learn", help="add a species profile from a recording of that species")
    l.add_argument("species")
    l.add_argument("file")
    ls = sub.add_parser("list", help="search the species BirdNET can recognise")
    ls.add_argument("search", nargs="?", default="", help="part of a bird name, e.g. myna")
    for sp in (r, a, sub.choices["demo"]):
        sp.add_argument("--no-show", action="store_true", help="save the plot without opening a window")
        sp.add_argument("--no-birdnet", action="store_true", help="skip BirdNET even if installed")
        sp.add_argument("--lat", type=float, help="latitude, helps BirdNET filter local species")
        sp.add_argument("--lon", type=float, help="longitude")
        sp.add_argument("--details", action="store_true", help="print every detected sound piece")
    args = p.parse_args()

    if args.cmd is None:  # interactive menu - keeps coming back until you quit
        menu()
        return

    if args.cmd == "list":
        search_species(args.search or " ")
    elif args.cmd == "learn":
        learn(args.species, args.file)
    else:
        if args.cmd == "record":
            audio, sr, path = record_audio(args.seconds)
        elif args.cmd == "analyze":
            audio, sr = load_audio(args.file)
            path = args.file
        else:  # demo
            os.makedirs(REC_DIR, exist_ok=True)
            path = os.path.join(REC_DIR, "demo_scene.wav")
            audio, sr = make_demo_audio(), SAMPLE_RATE
            save_wav(path, audio, sr)
        analyze(audio, sr, path, show=not args.no_show, birdnet=not args.no_birdnet,
                lat=args.lat, lon=args.lon, details=args.details)


if __name__ == "__main__":
    main()
