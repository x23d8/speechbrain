#!/usr/bin/env python3
"""
Flask demo server for SepFormer speech separation.

New in this version
-------------------
- --checkpoint_dir / --hparams / --port CLI flags so you can point at any
  checkpoint without editing the file.
- POST /load_model  — hot-swap the checkpoint from the UI at any time.
- GET  /model_info  — returns current checkpoint paths + model metadata.
- GET  /audio/<filename> also serves the saved mixture so the frontend can
  play back what was sent to the model.

Run:
    python app.py
    python app.py --checkpoint_dir path/to/save --hparams path/to/hparams.yaml
Then open http://localhost:7860
"""

import argparse
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
# Default paths (relative to this file)
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).parent
SEP_DIR  = THIS_DIR.parent          # recipes/WSJ0Mix/separation/

DEFAULT_HPARAMS   = SEP_DIR / "hparams" / "sepformer-customdataset.yaml"
DEFAULT_CKPT_DIR  = SEP_DIR / "results" / "sepformer-custom" / "42" / "save"

UPLOAD_DIR = THIS_DIR / "uploads"
OUTPUT_DIR = THIS_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS  = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
MAX_CONTENT_LENGTH  = 50 * 1024 * 1024   # 50 MB

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------
_model: dict = {}


def load_model(hparams_path: Path, checkpoint_dir: Path) -> dict:
    """
    Load encoder / masknet / decoder from a YAML hparams file and a
    directory that contains encoder.ckpt, decoder.ckpt, masknet.ckpt.

    Updates the global _model dict in-place and returns it.
    Raises on any error so callers can surface the message to the UI.
    """
    hparams_path   = Path(hparams_path)
    checkpoint_dir = Path(checkpoint_dir)

    if not hparams_path.exists():
        raise FileNotFoundError(f"hparams file not found: {hparams_path}")

    for name in ("encoder.ckpt", "decoder.ckpt", "masknet.ckpt"):
        p = checkpoint_dir / name
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {p}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[demo] Loading model on {device} ...")
    print(f"[demo]   hparams : {hparams_path}")
    print(f"[demo]   ckpt dir: {checkpoint_dir}")

    overrides = "data_folder: ."
    with open(hparams_path, encoding="utf-8") as f:
        hparams = load_hyperpyyaml(f, overrides)

    encoder = hparams["Encoder"].to(device)
    masknet = hparams["MaskNet"].to(device)
    decoder = hparams["Decoder"].to(device)

    encoder.load_state_dict(
        torch.load(checkpoint_dir / "encoder.ckpt", map_location=device)
    )
    decoder.load_state_dict(
        torch.load(checkpoint_dir / "decoder.ckpt", map_location=device)
    )
    masknet.load_state_dict(
        torch.load(checkpoint_dir / "masknet.ckpt", map_location=device)
    )

    encoder.eval()
    masknet.eval()
    decoder.eval()

    _model.clear()
    _model.update(
        {
            "encoder":        encoder,
            "masknet":        masknet,
            "decoder":        decoder,
            "num_spks":       hparams["num_spks"],
            "sample_rate":    hparams["sample_rate"],
            "device":         device,
            "hparams_path":   str(hparams_path),
            "checkpoint_dir": str(checkpoint_dir),
        }
    )

    print(
        f"[demo] Model ready — num_spks={hparams['num_spks']}, "
        f"sample_rate={hparams['sample_rate']} Hz"
    )
    return _model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def _run_separation(waveform: torch.Tensor) -> torch.Tensor:
    """
    Args:  waveform [1, T] mono on CPU
    Returns: est_sources [num_spks, T] on CPU
    """
    device   = _model["device"]
    encoder  = _model["encoder"]
    masknet  = _model["masknet"]
    decoder  = _model["decoder"]
    num_spks = _model["num_spks"]

    mix   = waveform.to(device)
    mix_w = encoder(mix)
    est_mask = masknet(mix_w)
    mix_w = torch.stack([mix_w] * num_spks)
    sep_h = mix_w * est_mask

    est_sources = torch.cat(
        [decoder(sep_h[i]) for i in range(num_spks)], dim=0
    )

    T_origin = mix.size(1)
    T_est    = est_sources.size(1)
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


@app.route("/model_info")
def model_info():
    if not _model:
        return jsonify(
            {
                "loaded":         False,
                "default_hparams":   str(DEFAULT_HPARAMS),
                "default_ckpt_dir":  str(DEFAULT_CKPT_DIR),
            }
        )
    return jsonify(
        {
            "loaded":         True,
            "num_spks":       _model["num_spks"],
            "sample_rate":    _model["sample_rate"],
            "device":         str(_model["device"]),
            "hparams_path":   _model["hparams_path"],
            "checkpoint_dir": _model["checkpoint_dir"],
        }
    )


@app.route("/load_model", methods=["POST"])
def api_load_model():
    """Hot-swap the model checkpoint from the UI."""
    body = request.get_json(silent=True) or {}
    ckpt_dir    = body.get("checkpoint_dir", "").strip()
    hparams_pth = body.get("hparams_path",   "").strip()

    if not ckpt_dir:
        return jsonify({"error": "checkpoint_dir is required"}), 400
    if not hparams_pth:
        return jsonify({"error": "hparams_path is required"}), 400

    try:
        load_model(Path(hparams_pth), Path(ckpt_dir))
        return jsonify(
            {
                "status":       "ok",
                "num_spks":     _model["num_spks"],
                "sample_rate":  _model["sample_rate"],
                "device":       str(_model["device"]),
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/separate", methods=["POST"])
def separate():
    if not _model:
        return jsonify({"error": "No model loaded. Use the Model Settings panel first."}), 503

    if "audio" not in request.files:
        return jsonify({"error": "No audio file in request."}), 400

    audio_file = request.files["audio"]
    if not audio_file.filename:
        return jsonify({"error": "Empty filename."}), 400

    suffix = Path(secure_filename(audio_file.filename)).suffix.lower()
    # Recorded blobs arrive as .webm or .ogg; accept them too
    if suffix not in ALLOWED_EXTENSIONS | {".webm", ".opus"}:
        return jsonify(
            {"error": f"Unsupported format '{suffix}'. Use: {', '.join(ALLOWED_EXTENSIONS)}"}
        ), 400

    session_id  = uuid.uuid4().hex[:10]
    upload_path = UPLOAD_DIR / f"{session_id}{suffix}"
    audio_file.save(str(upload_path))

    try:
        waveform, sr = torchaudio.load(str(upload_path))

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        model_sr = _model["sample_rate"]
        if sr != model_sr:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=model_sr
            )(waveform)

        duration = round(waveform.shape[1] / model_sr, 2)

        # Save the (mono, resampled) mixture so the frontend can play it back
        mix_fname = f"{session_id}_mix.wav"
        torchaudio.save(str(OUTPUT_DIR / mix_fname), waveform, model_sr)

        est_sources = _run_separation(waveform)
        num_spks    = _model["num_spks"]

        for i in range(num_spks):
            max_val = est_sources[i].abs().max()
            if max_val > 0:
                est_sources[i] = est_sources[i] / max_val * 0.95

        sources = []
        for i in range(num_spks):
            fname    = f"{session_id}_source{i + 1}.wav"
            torchaudio.save(
                str(OUTPUT_DIR / fname),
                est_sources[i].unsqueeze(0),
                model_sr,
            )
            sources.append(fname)

        return jsonify(
            {
                "session_id":  session_id,
                "mix_audio":   mix_fname,
                "sources":     sources,
                "num_spks":    num_spks,
                "sample_rate": model_sr,
                "duration":    duration,
            }
        )

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    finally:
        upload_path.unlink(missing_ok=True)


@app.route("/audio/<filename>")
def serve_audio(filename):
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
    parser = argparse.ArgumentParser(
        description="SepFormer separation demo server"
    )
    parser.add_argument(
        "--checkpoint_dir",
        default=str(DEFAULT_CKPT_DIR),
        help="Path to directory containing encoder.ckpt / decoder.ckpt / masknet.ckpt",
    )
    parser.add_argument(
        "--hparams",
        default=str(DEFAULT_HPARAMS),
        help="Path to the hyperpyyaml hparams file",
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Port to listen on (default: 7860)",
    )
    parser.add_argument(
        "--no_load", action="store_true",
        help="Start server without loading a model (load later via UI)",
    )
    args = parser.parse_args()

    if not args.no_load:
        print("[demo] Loading SepFormer model at startup …")
        try:
            load_model(Path(args.hparams), Path(args.checkpoint_dir))
        except Exception as e:
            print(f"[demo] WARNING: could not load model at startup: {e}")
            print("[demo] Server will start anyway — use the Model Settings panel to load.")

    print(f"[demo] Starting server on http://0.0.0.0:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=False)
