"""
SubSync backend
================
Upload a video -> extract its audio track -> transcribe with faster-whisper
-> write a timestamp-synced .srt -> optionally translate that transcript
to English -> serve everything for download.

Setup
-----
1. Install ffmpeg on the system (not via pip):
     macOS:   brew install ffmpeg
     Ubuntu:  sudo apt install ffmpeg
     Windows: https://ffmpeg.org/download.html  (add it to PATH)

2. pip install -r requirements.txt

3. python app.py
   Server runs on http://localhost:5000

Notes
-----
- Jobs are tracked in memory (JOBS dict). That's fine for a single-process
  dev server. For production, move job state to Redis/a DB and run
  transcription in a real task queue (Celery/RQ) instead of a bare thread.
- The Whisper model loads once, lazily, and is reused across requests.
- MODEL_SIZE trades speed for accuracy: tiny/base/small/medium/large-v3.
    "tiny" is the fast default on CPU. Set WHISPER_MODEL_SIZE=small for more
    accuracy at the cost of substantially longer processing time.
"""

import os
import re
import uuid
import threading
import subprocess
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from faster_whisper import WhisperModel

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
MAX_CONTENT_LENGTH = 1500 * 1024 * 1024  # 1500 MiB upload cap

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
CORS(app)  # allow the frontend (served from a different origin) to call this API

# ---------------------------------------------------------------------------
# Job tracking
# ---------------------------------------------------------------------------
JOBS = {}
JOBS_LOCK = threading.Lock()


def set_job(job_id, **fields):
    with JOBS_LOCK:
        JOBS[job_id].update(fields)


def get_job(job_id):
    with JOBS_LOCK:
        return dict(JOBS[job_id]) if job_id in JOBS else None


# ---------------------------------------------------------------------------
# Whisper model (loaded once, shared across requests)
# ---------------------------------------------------------------------------
MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "tiny")
DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")            # "cuda" if you have a GPU
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
BEAM_SIZE = int(os.environ.get("WHISPER_BEAM_SIZE", "1"))
CONDITION_ON_PREVIOUS_TEXT = os.environ.get(
    "WHISPER_CONDITION_ON_PREVIOUS_TEXT", "false"
).lower() == "true"

_model = None
_model_lock = threading.Lock()


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    return _model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def format_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def segments_to_srt(segments) -> str:
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}")
        lines.append(seg.text.strip())
        lines.append("")
    return "\n".join(lines)


def format_ass_timestamp(seconds: float) -> str:
    """ASS wants H:MM:SS.cc (centiseconds), not SRT's HH:MM:SS,mmm."""
    if seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int(round((seconds - int(seconds)) * 100))
    if centis == 100:
        centis = 0
        secs += 1
        if secs == 60:
            secs, minutes = 0, minutes + 1
            if minutes == 60:
                minutes, hours = 0, hours + 1
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def ass_escape(text: str) -> str:
    # Curly braces are override-tag delimiters in ASS, so swap them out
    # rather than risk a stray "{" from the transcript breaking the style.
    return text.strip().replace("\n", " ").replace("{", "(").replace("}", ")")


def subtitle_name(video_filename: str, job_id: str) -> str:
    stem = Path(video_filename).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "video"
    return f"{stem}-{job_id[:8]}"


# ASS colours are &HAABBGGRR. Alpha is inverted: 00 = fully opaque, FF = fully
# transparent. &H66 gives a translucent black background.
ASS_STYLE_TEMPLATE = """[Script Info]
Title: SubSync generated subtitles
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
PlayResX: 1280
PlayResY: 720

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,22,&H00FFFFFF,&H000000FF,&H00000000,&H66000000,0,0,0,0,100,100,0,0,3,1,1,2,40,40,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def segments_to_ass(segments) -> str:
    """
    Builds a .ass file with the style baked into the header:
    - PrimaryColour &H00FFFFFF -> white text, alpha 00 (fully opaque)
    - BackColour   &H66000000 -> translucent black box
    - BorderStyle 3            -> renders BackColour as a box behind the text
    - Outline 1, Shadow 1      -> thin outline and soft shadow
    - Fontname Arial, Fontsize 22 -> small sans-serif text
    Change Fontname/Fontsize/the two colour values here if you want a
    different look; every player that reads this file will follow it.
    """
    lines = [ASS_STYLE_TEMPLATE]
    for seg in segments:
        start = format_ass_timestamp(seg.start)
        end = format_ass_timestamp(seg.end)
        text = ass_escape(seg.text)
        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")
    return "\n".join(lines) + "\n"


def extract_audio(video_path: Path, audio_path: Path):
    """
    Pull the audio track out of the video as a standalone WAV file.
    This is a straight demux (no re-timing), so the audio's timeline
    matches the source video exactly — which is what keeps the .srt
    generated from it in sync when applied back to the original video.
    """
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn",                      # drop video stream
        "-acodec", "pcm_s16le",     # uncompressed PCM, easiest for Whisper
        "-ar", "16000", "-ac", "1", # 16kHz mono is what Whisper expects
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr[-2000:]}")


# ---------------------------------------------------------------------------
# Background job
# ---------------------------------------------------------------------------
def process_job(
    job_id: str, video_path: Path, subtitle_base: str, want_translation: bool
):
    audio_path = OUTPUT_DIR / f"{job_id}.wav"
    try:
        set_job(job_id, status="extracting_audio", progress=10)
        extract_audio(video_path, audio_path)

        set_job(job_id, status="transcribing", progress=35)
        model = get_model()

        segments_iter, info = model.transcribe(
            str(audio_path),
            task="transcribe",
            vad_filter=True,
            beam_size=BEAM_SIZE,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
        )
        segments = list(segments_iter)
        srt_path = OUTPUT_DIR / f"{subtitle_base}.srt"
        srt_path.write_text(segments_to_srt(segments), encoding="utf-8")
        ass_path = OUTPUT_DIR / f"{subtitle_base}.ass"
        ass_path.write_text(segments_to_ass(segments), encoding="utf-8")

        result = {
            "status": "done",
            "progress": 100,
            "detected_language": info.language,
            "srt_file": srt_path.name,
            "ass_file": ass_path.name,
            "translated_srt_file": None,
            "translated_ass_file": None,
        }

        # Whisper's "translate" task always targets English, which is
        # exactly the "any language -> English" behavior requested.
        if want_translation and info.language != "en":
            set_job(job_id, status="translating", progress=70)
            t_segments_iter, _ = model.transcribe(
                str(audio_path),
                task="translate",
                vad_filter=True,
                beam_size=BEAM_SIZE,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
            )
            t_segments = list(t_segments_iter)
            t_srt_path = OUTPUT_DIR / f"{subtitle_base}.en.srt"
            t_srt_path.write_text(segments_to_srt(t_segments), encoding="utf-8")
            t_ass_path = OUTPUT_DIR / f"{subtitle_base}.en.ass"
            t_ass_path.write_text(segments_to_ass(t_segments), encoding="utf-8")
            result["translated_srt_file"] = t_srt_path.name
            result["translated_ass_file"] = t_ass_path.name

        set_job(job_id, **result)

    except Exception as exc:
        set_job(job_id, status="error", progress=0, error=str(exc))
    finally:
        # Uploaded source video isn't needed after processing; the audio
        # and subtitle files in OUTPUT_DIR are what get served/downloaded.
        try:
            video_path.unlink(missing_ok=True)
            audio_path.unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health_check():
    return jsonify({"status": "ok"}), 200


@app.post("/api/upload")
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file included in the request."}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    want_translation = request.form.get("translate", "false").lower() == "true"

    job_id = uuid.uuid4().hex
    video_path = UPLOAD_DIR / f"{job_id}{ext}"
    subtitle_base = subtitle_name(file.filename, job_id)
    file.save(video_path)

    with JOBS_LOCK:
        JOBS[job_id] = {"status": "queued", "progress": 0}

    thread = threading.Thread(
        target=process_job,
        args=(job_id, video_path, subtitle_base, want_translation),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id}), 202


@app.get("/api/status/<job_id>")
def job_status(job_id):
    job = get_job(job_id)
    if job is None:
        return jsonify({"error": "Unknown job id."}), 404
    return jsonify(job)


@app.get("/api/download/<job_id>/<kind>")
def download_file(job_id, kind):
    job = get_job(job_id)
    if job is None or job.get("status") != "done":
        return jsonify({"error": "File not ready yet."}), 404

    filename_map = {
        "srt": job.get("srt_file"),                 # original-language, plain text
        "srt-en": job.get("translated_srt_file"),    # English translation, plain text
        "ass": job.get("ass_file"),                  # original-language, styled
        "ass-en": job.get("translated_ass_file"),    # English translation, styled
    }
    filename = filename_map.get(kind)
    if not filename:
        return jsonify({"error": "Requested file is not available for this job."}), 404

    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
