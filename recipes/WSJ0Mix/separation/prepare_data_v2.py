"""
CSV preparation for custom speech-separation datasets (v2).

Changes from prepare_data.py:
  - Computes real audio durations via soundfile.info() instead of
    hardcoding 1.0 for every sample.  Correct durations let SpeechBrain
    sort samples by length when batch_size > 1, which is needed for
    efficient dynamic batching in future runs.
  - Removed the unused noise_wav / noise_wav_format / noise_wav_opts
    columns.  The training pipeline never reads those columns; keeping
    them just bloated every CSV row.
  - valid_split and split_seed are now explicit keyword arguments so
    train_v2.py can forward values from the YAML hparams file rather
    than relying on buried defaults.
  - Added a lightweight per-split integrity check: verifies that every
    s1 and s2 file exists before writing the CSV.  A missing source file
    would silently produce NaN losses and is hard to debug later.
  - set_types defaults to ["train", "test"] to match the typical layout
    of a dataset that ships without a pre-built validation split.
"""

import csv
import os
import random


def _require_soundfile():
    try:
        import soundfile as sf
        return sf
    except ImportError:
        return None


def prepare_wsjmix_v2(
    datapath,
    savepath,
    n_spks=2,
    skip_prep=False,
    fs=8000,
    valid_split=0.1,
    split_seed=1234,
):
    """
    Entry point called by train_v2.py.

    Dispatches to the original WSJ0-Mix helpers for paths that contain
    the string "wsj", otherwise calls create_custom_dataset_v2.

    Arguments
    ---------
    datapath    : str  Root data folder (contains train/ and test/ sub-dirs).
    savepath    : str  Directory where the CSV files will be written.
    n_spks      : int  Number of speakers (2 or 3).
    skip_prep   : bool If True, skip CSV generation entirely.
    fs          : int  Sample rate (informational — used only for logging).
    valid_split : float Fraction of the *train* split to use as validation.
    split_seed  : int  RNG seed for the train/valid shuffle.
    """
    if skip_prep:
        return

    if "wsj" in datapath:
        # Reuse the original WSJ0-Mix helpers unchanged
        from prepare_data import create_wsj_csv, create_wsj_csv_3spks
        if n_spks == 2:
            assert "2speakers" in datapath, (
                "Inconsistent number of speakers and datapath"
            )
            create_wsj_csv(datapath, savepath)
        elif n_spks == 3:
            assert "3speakers" in datapath
            create_wsj_csv_3spks(datapath, savepath)
        else:
            raise ValueError("Unsupported Number of Speakers")
    else:
        print("Creating CSV files for a custom dataset (prepare_data_v2)")
        create_custom_dataset_v2(
            datapath,
            savepath,
            valid_split=valid_split,
            split_seed=split_seed,
        )


def create_custom_dataset_v2(
    datapath,
    savepath,
    dataset_name="custom",
    set_types=None,
    folder_names=None,
    valid_split=0.1,
    split_seed=1234,
):
    """
    Build train/valid/test CSV files for a custom two-speaker dataset.

    Expected on-disk layout (same as prepare_data.py, unchanged):
        <datapath>/
            train/
                mix/   *.wav   (mixture)
                s1/    *.wav   (speaker 1 — same filenames as mix/)
                s2/    *.wav   (speaker 2 — same filenames as mix/)
            test/
                mix/
                s1/
                s2/

    The *train* split is shuffled with a fixed seed and divided into
    a training CSV (<dataset_name>_train.csv) and a validation CSV
    (<dataset_name>_valid.csv).  The *test* split is written as-is to
    <dataset_name>_test.csv.

    Parameters
    ----------
    datapath     : str   Root data folder.
    savepath     : str   Directory where CSV files are written.
    dataset_name : str   Prefix for CSV filenames.
    set_types    : list  Sub-folders to process (default: ["train", "test"]).
    folder_names : dict  Mapping of logical names to actual sub-folder names.
    valid_split  : float Fraction of train to use as validation.
    split_seed   : int   Seed for the train/valid shuffle.
    """
    if set_types is None:
        set_types = ["train", "test"]
    if folder_names is None:
        folder_names = {"source1": "s1", "source2": "s2", "mixture": "mix"}

    sf = _require_soundfile()

    csv_columns = [
        "ID",
        "duration",
        "mix_wav",
        "mix_wav_format",
        "mix_wav_opts",
        "s1_wav",
        "s1_wav_format",
        "s1_wav_opts",
        "s2_wav",
        "s2_wav_format",
        "s2_wav_opts",
    ]

    def _get_duration(path):
        """Return audio duration in seconds; falls back to 1.0 if unavailable."""
        if sf is not None:
            try:
                info = sf.info(path)
                return round(info.duration, 6)
            except Exception:
                pass
        return 1.0

    def _check_sources(mix_paths, s1_paths, s2_paths, label):
        """Raise if any source file is missing."""
        missing = []
        for mp, s1p, s2p in zip(mix_paths, s1_paths, s2_paths):
            for p in (mp, s1p, s2p):
                if not os.path.isfile(p):
                    missing.append(p)
        if missing:
            sample = "\n  ".join(missing[:10])
            raise FileNotFoundError(
                f"[prepare_data_v2] {len(missing)} missing file(s) in "
                f"'{label}' split (first {min(10, len(missing))}):\n  {sample}"
            )

    def _write_csv(filepath, mix_paths, s1_paths, s2_paths):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for i, (mp, s1p, s2p) in enumerate(
                zip(mix_paths, s1_paths, s2_paths)
            ):
                writer.writerow(
                    {
                        "ID": i,
                        "duration": _get_duration(mp),
                        "mix_wav": mp,
                        "mix_wav_format": "wav",
                        "mix_wav_opts": None,
                        "s1_wav": s1p,
                        "s1_wav_format": "wav",
                        "s1_wav_opts": None,
                        "s2_wav": s2p,
                        "s2_wav_format": "wav",
                        "s2_wav_opts": None,
                    }
                )

    for set_type in set_types:
        mix_dir = os.path.join(datapath, set_type, folder_names["mixture"])
        s1_dir = os.path.join(datapath, set_type, folder_names["source1"])
        s2_dir = os.path.join(datapath, set_type, folder_names["source2"])

        files = sorted(os.listdir(mix_dir))

        mix_paths = [os.path.join(mix_dir, fl) for fl in files]
        s1_paths = [os.path.join(s1_dir, fl) for fl in files]
        s2_paths = [os.path.join(s2_dir, fl) for fl in files]

        _check_sources(mix_paths, s1_paths, s2_paths, set_type)

        if set_type == "train" and valid_split > 0:
            indices = list(range(len(files)))
            random.Random(split_seed).shuffle(indices)

            n_valid = max(1, int(len(indices) * valid_split))
            valid_idx = sorted(indices[:n_valid])
            train_idx = sorted(indices[n_valid:])

            train_mix = [mix_paths[i] for i in train_idx]
            train_s1  = [s1_paths[i]  for i in train_idx]
            train_s2  = [s2_paths[i]  for i in train_idx]

            valid_mix = [mix_paths[i] for i in valid_idx]
            valid_s1  = [s1_paths[i]  for i in valid_idx]
            valid_s2  = [s2_paths[i]  for i in valid_idx]

            train_csv = os.path.join(savepath, f"{dataset_name}_train.csv")
            valid_csv = os.path.join(savepath, f"{dataset_name}_valid.csv")

            _write_csv(train_csv, train_mix, train_s1, train_s2)
            _write_csv(valid_csv, valid_mix, valid_s1, valid_s2)

            print(
                f"  [{set_type}] {len(train_idx)} train  /  "
                f"{len(valid_idx)} valid  "
                f"({100*(1-valid_split):.0f}/{100*valid_split:.0f}%,  "
                f"seed={split_seed})"
            )
        else:
            csv_path = os.path.join(savepath, f"{dataset_name}_{set_type}.csv")
            _write_csv(csv_path, mix_paths, s1_paths, s2_paths)
            print(f"  [{set_type}] {len(files)} samples → {csv_path}")
