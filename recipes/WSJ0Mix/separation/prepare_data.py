"""
The .csv preparation functions for WSJ0-Mix.

Author
 * Cem Subakan 2020

"""

import csv
import os
import random


def prepare_wsjmix(
    datapath,
    savepath,
    n_spks=2,
    skip_prep=False,
    librimix_addnoise=False,
    fs=8000,
):
    """
    Prepared wsj2mix if n_spks=2 and wsj3mix if n_spks=3.

    Arguments:
    ----------
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file.
        n_spks (int): number of speakers
        skip_prep (bool): If True, skip data preparation
        librimix_addnoise: If True, add whamnoise to librimix datasets
    """

    if skip_prep:
        return

    if "wsj" in datapath:
        if n_spks == 2:
            assert "2speakers" in datapath, (
                "Inconsistent number of speakers and datapath"
            )
            create_wsj_csv(datapath, savepath)
        elif n_spks == 3:
            assert "3speakers" in datapath, (
                "Inconsistent number of speakers and datapath"
            )
            create_wsj_csv_3spks(datapath, savepath)
        else:
            raise ValueError("Unsupported Number of Speakers")
    else:
        print("Creating a csv file for a custom dataset")
        create_custom_dataset(datapath, savepath)


def create_custom_dataset(
    datapath,
    savepath,
    dataset_name="custom",
    set_types=["train", "test"],
    folder_names={
        "source1": "s1",
        "source2": "s2",
        "mixture": "mix"
    },
    valid_split=0.1,
    split_seed=1234,
):
    """
    This function creates the csv files for a custom source separation dataset.

    For the 'train' set_type, the data is shuffled (with a fixed seed for
    reproducibility) and split into train (90%) and valid (10%) CSVs.
    """

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
        "noise_wav",
        "noise_wav_format",
        "noise_wav_opts",
    ]

    def _write_csv(filepath, mix_paths, s1_paths, s2_paths):
        """Helper: write a single CSV file from parallel path lists."""
        with open(filepath, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for i, (mp, s1p, s2p) in enumerate(zip(mix_paths, s1_paths, s2_paths)):
                writer.writerow({
                    "ID": i,
                    "duration": 1.0,
                    "mix_wav": mp,
                    "mix_wav_format": "wav",
                    "mix_wav_opts": None,
                    "s1_wav": s1p,
                    "s1_wav_format": "wav",
                    "s1_wav_opts": None,
                    "s2_wav": s2p,
                    "s2_wav_format": "wav",
                    "s2_wav_opts": None,
                })

    for set_type in set_types:
        mix_dir = os.path.join(datapath, set_type, folder_names["mixture"])
        s1_dir = os.path.join(datapath, set_type, folder_names["source1"])
        s2_dir = os.path.join(datapath, set_type, folder_names["source2"])

        files = sorted(os.listdir(mix_dir))

        mix_fl_paths = [os.path.join(mix_dir, fl) for fl in files]
        s1_fl_paths = [os.path.join(s1_dir, fl) for fl in files]
        s2_fl_paths = [os.path.join(s2_dir, fl) for fl in files]

        if set_type == "train" and valid_split > 0:
            # Shuffle with fixed seed then split
            indices = list(range(len(files)))
            random.Random(split_seed).shuffle(indices)

            n_valid = max(1, int(len(indices) * valid_split))
            valid_idx = sorted(indices[:n_valid])
            train_idx = sorted(indices[n_valid:])

            train_mix = [mix_fl_paths[i] for i in train_idx]
            train_s1  = [s1_fl_paths[i] for i in train_idx]
            train_s2  = [s2_fl_paths[i] for i in train_idx]

            valid_mix = [mix_fl_paths[i] for i in valid_idx]
            valid_s1  = [s1_fl_paths[i] for i in valid_idx]
            valid_s2  = [s2_fl_paths[i] for i in valid_idx]

            train_csv = os.path.join(savepath, f"{dataset_name}_train.csv")
            valid_csv = os.path.join(savepath, f"{dataset_name}_valid.csv")

            _write_csv(train_csv, train_mix, train_s1, train_s2)
            _write_csv(valid_csv, valid_mix, valid_s1, valid_s2)

            print(f"  Train/Valid split: {len(train_idx)} train, {len(valid_idx)} valid "
                  f"({100*(1-valid_split):.0f}/{100*valid_split:.0f}%, seed={split_seed})")
        else:
            csv_path = os.path.join(savepath, f"{dataset_name}_{set_type}.csv")
            _write_csv(csv_path, mix_fl_paths, s1_fl_paths, s2_fl_paths)


def create_wsj_csv(datapath, savepath):
    """
    This function creates the csv files to get the speechbrain data loaders for the wsj0-2mix dataset.

    Arguments:
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file
    """
    for set_type in ["tr", "cv", "tt"]:
        mix_path = os.path.join(datapath, "wav8k/min/" + set_type + "/mix/")
        s1_path = os.path.join(datapath, "wav8k/min/" + set_type + "/s1/")
        s2_path = os.path.join(datapath, "wav8k/min/" + set_type + "/s2/")

        files = os.listdir(mix_path)

        mix_fl_paths = [mix_path + fl for fl in files]
        s1_fl_paths = [s1_path + fl for fl in files]
        s2_fl_paths = [s2_path + fl for fl in files]

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

        with open(
            savepath + "/wsj_" + set_type + ".csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for i, (mix_path, s1_path, s2_path) in enumerate(
                zip(mix_fl_paths, s1_fl_paths, s2_fl_paths)
            ):
                row = {
                    "ID": i,
                    "duration": 1.0,
                    "mix_wav": mix_path,
                    "mix_wav_format": "wav",
                    "mix_wav_opts": None,
                    "s1_wav": s1_path,
                    "s1_wav_format": "wav",
                    "s1_wav_opts": None,
                    "s2_wav": s2_path,
                    "s2_wav_format": "wav",
                    "s2_wav_opts": None,
                }
                writer.writerow(row)


def create_wsj_csv_3spks(datapath, savepath):
    """
    This function creates the csv files to get the speechbrain data loaders for the wsj0-3mix dataset.

    Arguments:
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file
    """
    for set_type in ["tr", "cv", "tt"]:
        mix_path = os.path.join(datapath, "wav8k/min/" + set_type + "/mix/")
        s1_path = os.path.join(datapath, "wav8k/min/" + set_type + "/s1/")
        s2_path = os.path.join(datapath, "wav8k/min/" + set_type + "/s2/")
        s3_path = os.path.join(datapath, "wav8k/min/" + set_type + "/s3/")

        files = os.listdir(mix_path)

        mix_fl_paths = [mix_path + fl for fl in files]
        s1_fl_paths = [s1_path + fl for fl in files]
        s2_fl_paths = [s2_path + fl for fl in files]
        s3_fl_paths = [s3_path + fl for fl in files]

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
            "s3_wav",
            "s3_wav_format",
            "s3_wav_opts",
        ]

        with open(
            savepath + "/wsj_" + set_type + ".csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for i, (mix_path, s1_path, s2_path, s3_path) in enumerate(
                zip(mix_fl_paths, s1_fl_paths, s2_fl_paths, s3_fl_paths)
            ):
                row = {
                    "ID": i,
                    "duration": 1.0,
                    "mix_wav": mix_path,
                    "mix_wav_format": "wav",
                    "mix_wav_opts": None,
                    "s1_wav": s1_path,
                    "s1_wav_format": "wav",
                    "s1_wav_opts": None,
                    "s2_wav": s2_path,
                    "s2_wav_format": "wav",
                    "s2_wav_opts": None,
                    "s3_wav": s3_path,
                    "s3_wav_format": "wav",
                    "s3_wav_opts": None,
                }
                writer.writerow(row)
