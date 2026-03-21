# SepFormer Demo

Interactive web demo for speech separation using the custom-trained SepFormer checkpoint.

## Quick start

```bash
# From the repo root
pip install flask werkzeug

# Run from the demo directory
cd recipes/WSJ0Mix/separation/demo
python app.py
```

Then open **http://localhost:7860** in your browser.

## What it uses

| Item | Path |
|------|------|
| Model hparams | `hparams/sepformer-customdataset.yaml` |
| Checkpoint | `results/sepformer-custom/42/save/{encoder,decoder,masknet}.ckpt` |

## Notes

- The model is loaded **once** at startup (the 113 MB masknet may take a few seconds).
- Inference runs on CUDA if available, otherwise CPU.
- Uploaded files are deleted after processing; separated WAVs are kept in `demo/outputs/`.
- Supports WAV, FLAC, MP3, OGG, M4A (max 50 MB).
