#!/usr/bin/env python3
"""
Flask demo server for SepFormer speech separation.

Loads the trained checkpoint at startup and exposes a simple REST API
that the frontend uses to separate uploaded audio files.

Run:
    python app.py
Then open http://localhost:7860 in your browser.
"""

import os
import sys
import uuid
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from flask import Flask, jsonify, render_template, request, send_file
from hyperpyyaml import load_hyperpyyaml
from werkzeug.utils import secure_filename

# ---------------------------------------------------------------------------
# Paths (all relative to this file so the script works from any cwd)
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).parent
SEP_DIR = THIS_DIR.parent  # recipes/WSJ0Mix/separation/

HPARAMS_PATH = SEP_DIR / "hparams" / "sepformer-customdataset.yaml"
CHECKPOINT_DIR = SEP_DIR / "results" / "sepformer-custom" / "42" / "save" / "bscpt"

UPLOAD_DIR = THIS_DIR / "uploads"
OUTPUT_DIR = THIS_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# ---------------------------------------------------------------------------
# Global model state (loaded once at startup)
# ---------------------------------------------------------------------------
_model = {}


def load_model():
    """Load encoder / masknet / decoder from YAML hparams + saved checkpoints."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[demo] Using device: {device}")

    if not HPARAMS_PATH.exists():
        sys.exit(f"[ERROR] hparams file not found: {HPARAMS_PATH}")

    for name, path in [
        ("encoder", CHECKPOINT_DIR / "encoder.ckpt"),
        ("decoder", CHECKPOINT_DIR / "decoder.ckpt"),
        ("masknet", CHECKPOINT_DIR / "masknet.ckpt"),
    ]:
        if not path.exists():
            sys.exit(f"[ERROR] Checkpoint not found: {path}")

    overrides = "data_folder: ."
    with open(HPARAMS_PATH, encoding="utf-8") as f:
        hparams = load_hyperpyyaml(f, overrides)

    encoder = hparams["Encoder"].to(device)
    masknet = hparams["MaskNet"].to(device)
    decoder = hparams["Decoder"].to(device)

    encoder.load_state_dict(
        torch.load(CHECKPOINT_DIR / "encoder.ckpt", map_location=device)
    )
    decoder.load_state_dict(
        torch.load(CHECKPOINT_DIR / "decoder.ckpt", map_location=device)
    )
    masknet.load_state_dict(
        torch.load(CHECKPOINT_DIR / "masknet.ckpt", map_location=device)
    )

    encoder.eval()
    masknet.eval()
    decoder.eval()

    _model["encoder"] = encoder
    _model["masknet"] = masknet
    _model["decoder"] = decoder
    _model["num_spks"] = hparams["num_spks"]
    _model["sample_rate"] = hparams["sample_rate"]
    _model["device"] = device

    print(
        f"[demo] Model ready — num_spks={hparams['num_spks']}, "
        f"sample_rate={hparams['sample_rate']} Hz"
    )


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def _run_separation(waveform: torch.Tensor) -> torch.Tensor:
    """
    Args:
        waveform: [1, T] mono waveform on CPU
    Returns:
        est_sources: [num_spks, T] on CPU
    """
    device = _model["device"]
    encoder = _model["encoder"]
    masknet = _model["masknet"]
    decoder = _model["decoder"]
    num_spks = _model["num_spks"]

    mix = waveform.to(device)

    mix_w = encoder(mix)                            # [1, N, L]
    est_mask = masknet(mix_w)                       # [num_spks, 1, N, L]
    mix_w = torch.stack([mix_w] * num_spks)         # [num_spks, 1, N, L]
    sep_h = mix_w * est_mask                        # [num_spks, 1, N, L]

    est_sources = torch.cat(
        [decoder(sep_h[i]) for i in range(num_spks)], dim=0
    )                                               # [num_spks, T']

    # Trim / pad to original length
    T_origin = mix.size(1)
    T_est = est_sources.size(1)
    if T_origin > T_est:
        est_sources = F.pad(est_sources, (0, T_origin - T_est))
    else:
        est_sources = est_sources[:, :T_origin]

    return est_sources.cpu()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/separate", methods=["POST"])
def separate():
    if not _model:
        return jsonify({"error": "Model not loaded yet. Please wait."}), 503

    if "audio" not in request.files:
        return jsonify({"error": "No audio file in request."}), 400

    audio_file = request.files["audio"]
    if not audio_file.filename:
        return jsonify({"error": "Empty filename."}), 400

    suffix = Path(secure_filename(audio_file.filename)).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return jsonify(
            {"error": f"Unsupported format '{suffix}'. Use: {', '.join(ALLOWED_EXTENSIONS)}"}
        ), 400

    session_id = uuid.uuid4().hex[:10]
    upload_path = UPLOAD_DIR / f"{session_id}{suffix}"
    audio_file.save(str(upload_path))

    try:
        waveform, sr = torchaudio.load(str(upload_path))

        # Stereo → mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        model_sr = _model["sample_rate"]
        if sr != model_sr:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=model_sr
            )(waveform)

        duration = round(waveform.shape[1] / model_sr, 2)

        est_sources = _run_separation(waveform)
        num_spks = _model["num_spks"]

        # Normalise to prevent clipping
        for i in range(num_spks):
            max_val = est_sources[i].abs().max()
            if max_val > 0:
                est_sources[i] = est_sources[i] / max_val * 0.95

        sources = []
        for i in range(num_spks):
            fname = f"{session_id}_source{i + 1}.wav"
            out_path = OUTPUT_DIR / fname
            torchaudio.save(str(out_path), est_sources[i].unsqueeze(0), model_sr)
            sources.append(fname)

        return jsonify(
            {
                "session_id": session_id,
                "sources": sources,
                "num_spks": num_spks,
                "sample_rate": model_sr,
                "duration": duration,
            }
        )

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    finally:
        upload_path.unlink(missing_ok=True)


@app.route("/audio/<filename>")
def serve_audio(filename):
    """Serve a separated output WAV file."""
    safe_name = secure_filename(filename)
    file_path = OUTPUT_DIR / safe_name
    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404
    return send_file(str(file_path), mimetype="audio/wav")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": bool(_model)})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[demo] Loading SepFormer model…")
    load_model()
    print("[demo] Starting server on http://0.0.0.0:7860")
    app.run(host="0.0.0.0", port=7860, debug=False, threaded=False)
