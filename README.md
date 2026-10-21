# Bird Sound ID

Record a sound (laptop mic or audio file) and find out:

- **Is it a bird?** (or a voice, a clap, noise, a hum)
- **Which bird?** One final answer from **BirdNET**, a neural network that knows ~6,500 bird species.
- **The data behind it:** frequency-over-time graphs, a spectrogram, and CSV files with
  time → frequency data and measurements for every sound.

**Try it online:** see the website link at the top of this page (hosted free on Render; the first
visit after a quiet period takes about a minute to wake up).

You can also run it on your own laptop, as a **website** or a **Terminal menu**.

---

## 1. Setup (once, a few minutes, ~300 MB download)

You need **Python 3.10 or newer** ([python.org/downloads](https://www.python.org/downloads/)).

| Mac | Windows |
|---|---|
| Double-click **`Setup (Mac).command`** <br>(if macOS blocks it: right-click → Open → Open) | Double-click **`setup_windows.bat`** |
| or in Terminal: `sh setup.sh` | |

This creates a private `.venv` folder with all the libraries. It doesn't touch your system Python.

## 2. Start the website

| Mac | Windows |
|---|---|
| Double-click **`Start Bird ID.command`** | Double-click **`start_windows.bat`** |
| or: `python3 web_app.py` | or: `.venv\Scripts\python web_app.py` |

The site opens at **http://127.0.0.1:5050**. Keep the Terminal window open while you use it.
Press **Ctrl+C** in that window to stop.

> Don't open `web/index.html` directly. It only works through the server address above.

On the page:
1. Pick a length (5–30 s) and press **Record**. Allow microphone access when the browser asks.
   Or use **Upload** for a .wav / .mp3 / .m4a file.
2. Optionally tick **Use my location** so BirdNET only considers local birds.
3. You get the **final answer** with a confidence level:
   **Confirmed** (≥ 70 %), **Possible** (40–70 %), **Low confidence** (< 40 %).

## 3. Terminal version (optional)

```bash
python3 bird_sound_analyzer.py                       # menu
python3 bird_sound_analyzer.py record --seconds 15   # record + identify
python3 bird_sound_analyzer.py analyze clip.wav      # analyze a file
python3 bird_sound_analyzer.py list myna             # search BirdNET's species
python3 bird_sound_analyzer.py learn "Common Myna" clip.wav   # teach the rule-based method
```
Add `--details` to see every step, and `--lat 17.4 --lon 78.5` for your location.

---

## How it works

1. **Spectrogram:** the audio is split into ~6 ms slices and the loudness of every pitch is measured.
2. **Sound detection:** anything clearly louder than the background is a "sound". Time *and*
   pitch are used, so two birds calling at once in different pitch ranges stay separate.
3. **Bird or not:** each sound is scored on pitch (birds are mostly 1–10 kHz), tonality
   (whistle vs noise), length and pitch change. Voices, claps and hums are recognised.
4. **Measurements:** peak frequency → duration → bandwidth → pitch shape → calls per second →
   tonality. A rule-based method narrows a small built-in species list one parameter at a time.
5. **BirdNET** (Cornell Lab of Ornithology) gives the final species answer. Its model runs through a
   small TFLite interpreter (`ai-edge-litert`), so no full TensorFlow install is needed.
6. **Outputs** are saved in `recordings/` (audio) and `results/` (graph PNG + 2 CSV files).

| File | What it is |
|---|---|
| `bird_sound_analyzer.py` | All the analysis code + Terminal menu |
| `web_app.py` | Local web server connecting the page to the analyzer |
| `web/index.html` | The website |
| `requirements.txt` | Python libraries |
| `deploy/`, `render.yaml` | Server setup for hosting the website (Render) |
| `species_db_custom.json` | Species you taught it with `learn` (created when used) |

## Tips for good results
- Get close to the bird. Record at least 10 seconds and don't talk.
- Laptop mics are weak above ~8 kHz, so very high-pitched birds come through faintly.
- Under 40 % confidence means "not sure". Record again.

## Troubleshooting
- **"Microphone blocked":** click the mic icon in the browser's address bar → Allow → reload.
  On Mac also check System Settings → Privacy & Security → Microphone.
- **"Server not reachable":** the Terminal window running `web_app.py` was closed. Start it again.
- **Port 5050 in use:** another copy is already running. Use that one, or close it.
- **Setup fails installing ai-edge-litert:** use Python 3.10–3.13 from python.org.

## Hosting your own copy (free)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/itssanjay206-cell/bird-sound-id)

Sign in to Render with GitHub, click the button, then **Apply**. Render builds `deploy/Dockerfile`
and gives you a link like `https://bird-sound-id.onrender.com`. On the free plan, uploads are
trimmed to 30 seconds and visitors' recordings are deleted after an hour.
