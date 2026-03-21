#!/usr/bin/env python3
"""
Inference script for SepFormer speech separation.

Loads the best SpeechBrain checkpoint and separates a mixed audio file
into individual speaker sources.

Usage:
    python inference.py --audio mixture.wav
    python inference.py --audio mixture.wav --output_dir ./separated
    python inference.py --audio mixture.wav --hparams hparams/sepformer-customdataset.yaml
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import torchaudio
from hyperpyyaml import load_hyperpyyaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="SepFormer Inference — separate a mixed audio file into individual sources"
    )
    parser.add_argument(
        "--audio", type=str, required=True,
        help="Path to the input mixture audio file (wav, flac, mp3, etc.)"
    )
    parser.add_argument(
        "--hparams", type=str,
        default=os.path.join(os.path.dirname(__file__), "hparams", "sepformer-customdataset.yaml"),
        help="Path to the YAML hparams file (default: hparams/sepformer-customdataset.yaml)"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str,
        default=os.path.join(
            os.path.dirname(__file__),
            "results", "sepformer-custom", "results", "best_model", "checkpoint",
        ),
        help="Path to the checkpoint directory containing encoder.ckpt, decoder.ckpt, masknet.ckpt"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save separated audio files (default: same dir as input audio)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use (default: cuda if available, else cpu)"
    )
    return parser.parse_args()


def load_model(hparams_path, checkpoint_dir, device):
    """Load the SepFormer model from a SpeechBrain YAML config and checkpoint."""

    # We only need a minimal set of YAML keys — supply a dummy data_folder
    # so the !PLACEHOLDER doesn't raise an error.
    overrides = "data_folder: ."

    with open(hparams_path, encoding="utf-8") as f:
        hparams = load_hyperpyyaml(f, overrides)

    encoder = hparams["Encoder"].to(device)
    masknet = hparams["MaskNet"].to(device)
    decoder = hparams["Decoder"].to(device)
    num_spks = hparams["num_spks"]
    sample_rate = hparams["sample_rate"]

    # Load checkpoint weights
    enc_ckpt = os.path.join(checkpoint_dir, "encoder.ckpt")
    dec_ckpt = os.path.join(checkpoint_dir, "decoder.ckpt")
    mask_ckpt = os.path.join(checkpoint_dir, "masknet.ckpt")

    for path in (enc_ckpt, dec_ckpt, mask_ckpt):
        if not os.path.isfile(path):
            sys.exit(f"[ERROR] Checkpoint file not found: {path}")

    encoder.load_state_dict(torch.load(enc_ckpt, map_location=device))
    decoder.load_state_dict(torch.load(dec_ckpt, map_location=device))
    masknet.load_state_dict(torch.load(mask_ckpt, map_location=device))

    encoder.eval()
    masknet.eval()
    decoder.eval()

    print(f"  ✔ Model loaded from: {checkpoint_dir}")
    print(f"  ✔ num_spks={num_spks}, sample_rate={sample_rate}")

    return encoder, masknet, decoder, num_spks, sample_rate


@torch.no_grad()
def separate(encoder, masknet, decoder, num_spks, mix, device):
    """
    Run the SepFormer forward pass on a mixture waveform.

    Args:
        mix: Tensor of shape [1, T] (mono waveform, single batch)

    Returns:
        est_sources: Tensor of shape [num_spks, T]
    """
    mix = mix.to(device)

    # Encoder
    mix_w = encoder(mix)                           # [1, N, L]

    # Mask estimation
    est_mask = masknet(mix_w)                      # [num_spks, 1, N, L]

    # Apply masks
    mix_w = torch.stack([mix_w] * num_spks)        # [num_spks, 1, N, L]
    sep_h = mix_w * est_mask                       # [num_spks, 1, N, L]

    # Decoder
    est_sources = []
    for i in range(num_spks):
        est_source = decoder(sep_h[i])             # [1, T']
        est_sources.append(est_source)
    est_sources = torch.cat(est_sources, dim=0)    # [num_spks, T']

    # Fix length to match original
    T_origin = mix.size(1)
    T_est = est_sources.size(1)
    if T_origin > T_est:
        est_sources = F.pad(est_sources, (0, T_origin - T_est))
    else:
        est_sources = est_sources[:, :T_origin]

    return est_sources


def main():
    args = parse_args()

    # Validate input
    if not os.path.isfile(args.audio):
        sys.exit(f"[ERROR] Audio file not found: {args.audio}")

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print("\nLoading model...")
    encoder, masknet, decoder, num_spks, model_sr = load_model(
        args.hparams, args.checkpoint_dir, device
    )

    # Load audio
    print(f"\nLoading audio: {args.audio}")
    waveform, sr = torchaudio.load(args.audio)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
        print(f"  Converted to mono")

    # Resample if needed
    if sr != model_sr:
        print(f"  Resampling {sr} Hz → {model_sr} Hz")
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=model_sr)
        waveform = resampler(waveform)
        sr = model_sr

    duration_sec = waveform.shape[1] / sr
    print(f"  Duration: {duration_sec:.2f}s, Sample rate: {sr} Hz")

    # Separate
    print("\nSeparating sources...")
    est_sources = separate(encoder, masknet, decoder, num_spks, waveform, device)
    est_sources = est_sources.cpu()

    # Normalize each source to prevent clipping
    for i in range(num_spks):
        max_val = est_sources[i].abs().max()
        if max_val > 0:
            est_sources[i] = est_sources[i] / max_val * 0.95

    # Output directory
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.dirname(os.path.abspath(args.audio))
    os.makedirs(out_dir, exist_ok=True)

    # Save separated sources
    base_name = os.path.splitext(os.path.basename(args.audio))[0]
    saved_files = []
    for i in range(num_spks):
        out_path = os.path.join(out_dir, f"{base_name}_source{i + 1}.wav")
        torchaudio.save(out_path, est_sources[i].unsqueeze(0), sr)
        saved_files.append(out_path)
        print(f"  ✔ Source {i + 1} saved: {out_path}")

    print(f"\nDone! Separated {num_spks} sources from '{args.audio}'")


if __name__ == "__main__":
    main()
