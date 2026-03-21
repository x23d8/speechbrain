#!/usr/bin/env/python3
"""
Fine-tuning recipe for SepFormer (v2) — fixes catastrophic-forgetting and
overfitting issues observed in the seed-42 run.

Key changes vs train.py
-----------------------
1. Uses prepare_data_v2.py which computes real audio durations and exposes
   valid_split / split_seed as YAML-controllable hparams.
2. Progressive encoder unfreezing: the encoder is frozen for the first
   `freeze_encoder_epochs` epochs (default 5) so the high-capacity MaskNet
   adapts first without overwriting pretrained encoder features.  After that
   the encoder is unfrozen and trained jointly at the (already-reduced) LR.
3. Passes valid_split and split_seed from hparams into prepare_data_v2 so
   the YAML is the single source of truth for all hyperparameters.

Usage
-----
python train_v2.py hparams/sepformer-finetune.yaml --data_folder F:/ducdataset
"""

import csv
import os
import shutil
import signal
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm

import speechbrain as sb
import speechbrain.nnet.schedulers as schedulers
from speechbrain.dataio import audio_io
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.logger import get_logger

import wandb

os.environ["WANDB_PROJECT"] = "sepformer-speech-separation"
os.environ["WANDB_ENTITY"] = "slp301_ai1802"

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
)
from loss import pit_sisnr_loss


class Separation(sb.Brain):
    # ------------------------------------------------------------------
    # Encoder freeze / unfreeze
    # ------------------------------------------------------------------
    def _set_encoder_frozen(self, frozen: bool):
        """Freeze or unfreeze encoder parameters."""
        for p in self.hparams.Encoder.parameters():
            p.requires_grad_(not frozen)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def compute_forward(self, mix, targets, stage, noise=None):
        """Forward pass: mixture → separated sources."""
        mix, mix_lens = mix
        mix, mix_lens = mix.to(self.device), mix_lens.to(self.device)

        targets = torch.cat(
            [targets[i][0].unsqueeze(-1) for i in range(self.hparams.num_spks)],
            dim=-1,
        ).to(self.device)

        if stage == sb.Stage.TRAIN:
            with torch.no_grad():
                if self.hparams.use_speedperturb:
                    mix, targets = self.add_speed_perturb(targets, mix_lens)
                    mix = targets.sum(-1)

                if self.hparams.use_wavedrop:
                    mix = self.hparams.drop_chunk(mix, mix_lens)
                    mix = self.hparams.drop_freq(mix)

                if self.hparams.limit_training_signal_len:
                    mix, targets = self.cut_signals(mix, targets)

        mix_w = self.hparams.Encoder(mix)
        est_mask = self.hparams.MaskNet(mix_w)
        mix_w = torch.stack([mix_w] * self.hparams.num_spks)
        sep_h = mix_w * est_mask

        est_source = torch.cat(
            [
                self.hparams.Decoder(sep_h[i]).unsqueeze(-1)
                for i in range(self.hparams.num_spks)
            ],
            dim=-1,
        )

        T_origin = mix.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]

        return est_source, targets

    def compute_objectives(self, predictions, targets):
        """PIT SI-SNR loss — expects [B, T, C], returns scalar."""
        est = predictions.permute(0, 2, 1)
        tgt = targets.permute(0, 2, 1)
        return pit_sisnr_loss(est, tgt)

    # ------------------------------------------------------------------
    # fit_batch — applies encoder freezing per-epoch
    # ------------------------------------------------------------------
    def fit_batch(self, batch):
        mixture = batch.mix_sig
        targets = [batch.s1_sig, batch.s2_sig]
        if self.hparams.num_spks == 3:
            targets.append(batch.s3_sig)

        # ── Progressive encoder unfreezing ─────────────────────────────
        freeze_until = getattr(self.hparams, "freeze_encoder_epochs", 0)
        current_epoch = self.hparams.epoch_counter.current
        encoder_is_frozen = current_epoch <= freeze_until
        self._set_encoder_frozen(encoder_is_frozen)
        # ───────────────────────────────────────────────────────────────

        with self.training_ctx:
            predictions, targets = self.compute_forward(
                mixture, targets, sb.Stage.TRAIN
            )
            loss = self.compute_objectives(predictions, targets)

        grad_norm = 0.0
        if loss.item() < self.hparams.loss_upper_lim:
            self.scaler.scale(loss).backward()
            if self.hparams.clip_grad_norm >= 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.modules.parameters(),
                    self.hparams.clip_grad_norm,
                ).item()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.nonfinite_count += 1
            logger.info(
                f"infinite loss! it happened {self.nonfinite_count} times "
                f"so far - skipping this batch"
            )
            loss = torch.tensor(0.0, device=self.device)
        self.optimizer.zero_grad()

        if not hasattr(self, "_grad_norm_accum"):
            self._grad_norm_accum = []
        self._grad_norm_accum.append(grad_norm)

        step_loss = loss.detach().item()
        if wandb.run is not None:
            wandb.log({
                "train_step_loss":    step_loss,
                "train_step_si-snr":  -step_loss,
                "encoder_frozen":     int(encoder_is_frozen),
            })

        return loss.detach().cpu()

    def evaluate_batch(self, batch, stage):
        snt_id = batch.id
        mixture = batch.mix_sig
        targets = [batch.s1_sig, batch.s2_sig]
        if self.hparams.num_spks == 3:
            targets.append(batch.s3_sig)

        with torch.no_grad():
            predictions, targets = self.compute_forward(mixture, targets, stage)
            loss = self.compute_objectives(predictions, targets)

        if stage == sb.Stage.TEST and self.hparams.save_audio:
            if hasattr(self.hparams, "n_audio_to_save"):
                if self.hparams.n_audio_to_save > 0:
                    self.save_audio(snt_id[0], mixture, targets, predictions)
                    self.hparams.n_audio_to_save += -1
            else:
                self.save_audio(snt_id[0], mixture, targets, predictions)

        return loss.detach()

    def on_stage_end(self, stage, stage_loss, epoch):
        stage_stats = {"si-snr": stage_loss}

        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
            self._grad_norm_accum = getattr(self, "_grad_norm_accum", [])
            self._epoch_grad_norm = (
                float(np.mean(self._grad_norm_accum))
                if self._grad_norm_accum
                else 0.0
            )
            self._grad_norm_accum = []

            # Log encoder freeze state
            freeze_until = getattr(self.hparams, "freeze_encoder_epochs", 0)
            if epoch == freeze_until + 1:
                logger.info(
                    f"Epoch {epoch}: encoder UNFROZEN — "
                    f"all parameters now being trained."
                )
            elif epoch <= freeze_until:
                logger.info(
                    f"Epoch {epoch}: encoder FROZEN "
                    f"(will unfreeze at epoch {freeze_until + 1})"
                )

        if stage == sb.Stage.VALID:
            if isinstance(
                self.hparams.lr_scheduler, schedulers.ReduceLROnPlateau
            ):
                current_lr, next_lr = self.hparams.lr_scheduler(
                    [self.optimizer], epoch, stage_loss
                )
                schedulers.update_learning_rate(self.optimizer, next_lr)
            else:
                current_lr = self.hparams.optimizer.optim.param_groups[0]["lr"]

            train_loss = self.train_stats["si-snr"]
            val_loss   = stage_stats["si-snr"]
            grad_norm  = getattr(self, "_epoch_grad_norm", 0.0)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": current_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            if wandb.run is not None:
                wandb.log({
                    "epoch":      epoch,
                    "train_loss": train_loss,
                    "val_loss":   val_loss,
                    "lr":         current_lr,
                    "grad_norm":  grad_norm,
                })

            self.checkpointer.save_and_keep_only(
                meta={"si-snr": val_loss}, min_keys=["si-snr"]
            )

            is_better = (
                not hasattr(self, "_best_val_loss")
                or val_loss < self._best_val_loss
            )
            if is_better:
                self._best_val_loss = val_loss
                self._no_improve_count = 0
                self._save_best_to_results(epoch, val_loss)

                if wandb.run is not None:
                    wandb.run.summary["best_val_loss"] = val_loss
                    wandb.run.summary["best_si_snr_dB"] = -val_loss
                    wandb.run.summary["best_epoch"] = epoch
                    wandb.run.summary["best_train_loss"] = train_loss
                    wandb.run.summary["best_lr"] = current_lr
            else:
                self._no_improve_count = getattr(self, "_no_improve_count", 0) + 1

            patience = getattr(self.hparams, "early_stop_patience", 15)
            if self._no_improve_count >= patience:
                logger.info(
                    f"Early stopping triggered: no improvement for "
                    f"{patience} epochs."
                )
                self._emergency_save(epoch, reason="early_stopping")
                self.hparams.epoch_counter.current = (
                    self.hparams.epoch_counter.limit
                )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

    # ------------------------------------------------------------------
    def _save_best_to_results(self, epoch, val_loss):
        results_dir = getattr(
            self.hparams, "results_folder",
            os.path.join(self.hparams.output_folder, "..", "results"),
        )
        best_dir = os.path.join(results_dir, "best_model")
        os.makedirs(best_dir, exist_ok=True)

        ckpt_dir = self.checkpointer.checkpoints_dir
        ckpt_entries = [
            os.path.join(ckpt_dir, d)
            for d in os.listdir(ckpt_dir)
            if os.path.isdir(os.path.join(ckpt_dir, d))
        ]
        if ckpt_entries:
            latest_ckpt = max(ckpt_entries, key=os.path.getmtime)
            dest = os.path.join(best_dir, "checkpoint")
            if os.path.exists(dest):
                shutil.rmtree(dest)
            shutil.copytree(latest_ckpt, dest)

        with open(os.path.join(best_dir, "best_info.txt"), "w", encoding="utf-8") as f:
            f.write(f"epoch:     {epoch}\n")
            f.write(f"val_loss:  {val_loss:.6f}  (-SI-SNR, lower=better)\n")
            f.write(f"si_snr:    {-val_loss:.6f} dB\n")
            f.write(f"saved_at:  {datetime.now().isoformat()}\n")

        logger.info(
            f"Best model saved to {best_dir} "
            f"(epoch={epoch}, SI-SNR={-val_loss:.4f} dB)"
        )

    def _emergency_save(self, epoch, reason="interrupt"):
        results_dir = getattr(
            self.hparams, "results_folder",
            os.path.join(self.hparams.output_folder, "..", "results"),
        )
        save_dir = os.path.join(
            results_dir,
            f"emergency_save_{reason}_epoch{epoch}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        os.makedirs(save_dir, exist_ok=True)
        try:
            torch.save(
                {k: v.state_dict() for k, v in self.modules.items()},
                os.path.join(save_dir, "modules.pt"),
            )
            torch.save(
                self.optimizer.state_dict(),
                os.path.join(save_dir, "optimizer.pt"),
            )
            ckpt_dir = self.checkpointer.checkpoints_dir
            ckpt_entries = [
                os.path.join(ckpt_dir, d)
                for d in os.listdir(ckpt_dir)
                if os.path.isdir(os.path.join(ckpt_dir, d))
            ]
            if ckpt_entries:
                latest_ckpt = max(ckpt_entries, key=os.path.getmtime)
                shutil.copytree(latest_ckpt, os.path.join(save_dir, "sb_checkpoint"))

            with open(os.path.join(save_dir, "info.txt"), "w", encoding="utf-8") as f:
                f.write(f"reason:   {reason}\n")
                f.write(f"epoch:    {epoch}\n")
                f.write(f"saved_at: {datetime.now().isoformat()}\n")
                best_loss = getattr(self, "_best_val_loss", None)
                if best_loss is not None:
                    f.write(f"best_val_loss: {best_loss:.6f}\n")
                    f.write(f"best_si_snr:   {-best_loss:.6f} dB\n")

            logger.info(f"Emergency save to {save_dir} (reason={reason})")
        except Exception as e:
            logger.error(f"Emergency save failed: {e}")

    # ------------------------------------------------------------------
    # Augmentation helpers (unchanged from train.py)
    # ------------------------------------------------------------------
    def add_speed_perturb(self, targets, targ_lens):
        min_len = -1
        recombine = False

        if self.hparams.use_speedperturb or self.hparams.use_rand_shift:
            new_targets = []
            recombine = True

            for i in range(targets.shape[-1]):
                new_target = self.hparams.speed_perturb(targets[:, :, i])
                new_targets.append(new_target)
                if i == 0:
                    min_len = new_target.shape[-1]
                elif new_target.shape[-1] < min_len:
                    min_len = new_target.shape[-1]

            if self.hparams.use_rand_shift:
                recombine = True
                for i in range(targets.shape[-1]):
                    rand_shift = torch.randint(
                        self.hparams.min_shift, self.hparams.max_shift, (1,)
                    )
                    new_targets[i] = new_targets[i].to(self.device)
                    new_targets[i] = torch.roll(
                        new_targets[i], shifts=(rand_shift[0],), dims=1
                    )

            if recombine:
                if self.hparams.use_speedperturb:
                    targets = torch.zeros(
                        targets.shape[0],
                        min_len,
                        targets.shape[-1],
                        device=targets.device,
                        dtype=torch.float,
                    )
                for i, new_target in enumerate(new_targets):
                    targets[:, :, i] = new_targets[i][:, 0:min_len]

        mix = targets.sum(-1)
        return mix, targets

    def cut_signals(self, mixture, targets):
        randstart = torch.randint(
            0,
            1 + max(0, mixture.shape[1] - self.hparams.training_signal_len),
            (1,),
        ).item()
        targets = targets[
            :, randstart: randstart + self.hparams.training_signal_len, :
        ]
        mixture = mixture[
            :, randstart: randstart + self.hparams.training_signal_len
        ]
        return mixture, targets

    def reset_layer_recursively(self, layer):
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        for child_layer in layer.modules():
            if layer != child_layer:
                self.reset_layer_recursively(child_layer)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def save_results(self, test_data):
        from mir_eval.separation import bss_eval_sources

        save_file = os.path.join(self.hparams.output_folder, "test_results.csv")
        all_sdrs, all_sdrs_i, all_sisnrs, all_sisnrs_i = [], [], [], []
        csv_columns = ["snt_id", "sdr", "sdr_i", "si-snr", "si-snr_i"]

        test_loader = sb.dataio.dataloader.make_dataloader(
            test_data, **self.hparams.dataloader_opts
        )
        with open(save_file, "w", newline="", encoding="utf-8") as results_csv:
            writer = csv.DictWriter(results_csv, fieldnames=csv_columns)
            writer.writeheader()

            with tqdm(test_loader, dynamic_ncols=True) as t:
                for i, batch in enumerate(t):
                    mixture, mix_len = batch.mix_sig
                    snt_id = batch.id
                    targets = [batch.s1_sig, batch.s2_sig]
                    if self.hparams.num_spks == 3:
                        targets.append(batch.s3_sig)

                    with torch.no_grad():
                        predictions, targets = self.compute_forward(
                            batch.mix_sig, targets, sb.Stage.TEST
                        )

                    sisnr = self.compute_objectives(predictions, targets)

                    mixture_signal = torch.stack(
                        [mixture] * self.hparams.num_spks, dim=-1
                    ).to(targets.device)
                    sisnr_baseline = self.compute_objectives(
                        mixture_signal, targets
                    )
                    sisnr_i = sisnr - sisnr_baseline

                    sdr, _, _, _ = bss_eval_sources(
                        targets[0].t().cpu().numpy(),
                        predictions[0].t().detach().cpu().numpy(),
                    )
                    sdr_baseline, _, _, _ = bss_eval_sources(
                        targets[0].t().cpu().numpy(),
                        mixture_signal[0].t().detach().cpu().numpy(),
                    )
                    sdr_i = sdr.mean() - sdr_baseline.mean()

                    writer.writerow(
                        {
                            "snt_id": snt_id[0],
                            "sdr": sdr.mean(),
                            "sdr_i": sdr_i,
                            "si-snr": -sisnr.item(),
                            "si-snr_i": -sisnr_i.item(),
                        }
                    )

                    all_sdrs.append(sdr.mean())
                    all_sdrs_i.append(sdr_i.mean())
                    all_sisnrs.append(-sisnr.item())
                    all_sisnrs_i.append(-sisnr_i.item())

                writer.writerow(
                    {
                        "snt_id": "avg",
                        "sdr": np.array(all_sdrs).mean(),
                        "sdr_i": np.array(all_sdrs_i).mean(),
                        "si-snr": np.array(all_sisnrs).mean(),
                        "si-snr_i": np.array(all_sisnrs_i).mean(),
                    }
                )

        logger.info(f"Mean SISNR  is {np.array(all_sisnrs).mean()}")
        logger.info(f"Mean SISNRi is {np.array(all_sisnrs_i).mean()}")
        logger.info(f"Mean SDR    is {np.array(all_sdrs).mean()}")
        logger.info(f"Mean SDRi   is {np.array(all_sdrs_i).mean()}")

    def save_audio(self, snt_id, mixture, targets, predictions):
        save_path = os.path.join(self.hparams.save_folder, "audio_results")
        os.makedirs(save_path, exist_ok=True)

        for ns in range(self.hparams.num_spks):
            signal = predictions[0, :, ns]
            signal = signal / signal.abs().max()
            audio_io.save(
                os.path.join(save_path, f"item{snt_id}_source{ns + 1}hat.wav"),
                signal.unsqueeze(0).cpu(),
                self.hparams.sample_rate,
            )
            signal = targets[0, :, ns]
            signal = signal / signal.abs().max()
            audio_io.save(
                os.path.join(save_path, f"item{snt_id}_source{ns + 1}.wav"),
                signal.unsqueeze(0).cpu(),
                self.hparams.sample_rate,
            )

        signal = mixture[0][0, :]
        signal = signal / signal.abs().max()
        audio_io.save(
            os.path.join(save_path, f"item{snt_id}_mix.wav"),
            signal.unsqueeze(0).cpu(),
            self.hparams.sample_rate,
        )


# ── Data pipeline (unchanged) ───────────────────────────────────────────────

def dataio_prep(hparams):
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_data"],
        replacements={"data_root": hparams["data_folder"]},
    )
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_data"],
        replacements={"data_root": hparams["data_folder"]},
    )
    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    datasets = [train_data, valid_data, test_data]

    @sb.utils.data_pipeline.takes("mix_wav")
    @sb.utils.data_pipeline.provides("mix_sig")
    def audio_pipeline_mix(mix_wav):
        return sb.dataio.dataio.read_audio(mix_wav)

    @sb.utils.data_pipeline.takes("s1_wav")
    @sb.utils.data_pipeline.provides("s1_sig")
    def audio_pipeline_s1(s1_wav):
        return sb.dataio.dataio.read_audio(s1_wav)

    @sb.utils.data_pipeline.takes("s2_wav")
    @sb.utils.data_pipeline.provides("s2_sig")
    def audio_pipeline_s2(s2_wav):
        return sb.dataio.dataio.read_audio(s2_wav)

    if hparams["num_spks"] == 3:
        @sb.utils.data_pipeline.takes("s3_wav")
        @sb.utils.data_pipeline.provides("s3_sig")
        def audio_pipeline_s3(s3_wav):
            return sb.dataio.dataio.read_audio(s3_wav)

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_mix)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s1)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s2)
    if hparams["num_spks"] == 3:
        sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s3)
        sb.dataio.dataset.set_output_keys(
            datasets, ["id", "mix_sig", "s1_sig", "s2_sig", "s3_sig"]
        )
    else:
        sb.dataio.dataset.set_output_keys(
            datasets, ["id", "mix_sig", "s1_sig", "s2_sig"]
        )

    return train_data, valid_data, test_data


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    _cli = argparse.ArgumentParser(
        description="SepFormer Fine-tuning v2 — CLI overrides for YAML hparams",
        add_help=False,
    )

    _cli.add_argument("--epochs",               type=int,   default=None)
    _cli.add_argument("--lr",                   type=float, default=None)
    _cli.add_argument("--batch_size",           type=int,   default=None)
    _cli.add_argument("--clip_grad_norm",       type=float, default=None)
    _cli.add_argument("--seed",                 type=int,   default=None)
    _cli.add_argument("--precision",            type=str,   default=None,
                      choices=["fp32", "fp16", "bf16"])
    _cli.add_argument("--early_stop_patience",  type=int,   default=None)
    _cli.add_argument("--freeze_encoder_epochs",type=int,   default=None)
    _cli.add_argument("--valid_split",          type=float, default=None)
    _cli.add_argument("--split_seed",           type=int,   default=None)
    _cli.add_argument("--data_folder",          type=str,   default=None)
    _cli.add_argument("--experiment_name",      type=str,   default=None)
    _cli.add_argument("--num_spks",             type=int,   default=None,
                      choices=[2, 3])
    _cli.add_argument("--sample_rate",          type=int,   default=None)
    _cli.add_argument("--skip_prep",            action="store_true", default=None)
    _cli.add_argument("--use_wavedrop",         action="store_true", default=None)
    _cli.add_argument("--use_speedperturb",     action="store_true", default=None)
    _cli.add_argument("--use_rand_shift",       action="store_true", default=None)
    _cli.add_argument("--dynamic_mixing",       action="store_true", default=None)
    _cli.add_argument("--N_encoder_out",        type=int,   default=None)
    _cli.add_argument("--kernel_size",          type=int,   default=None)
    _cli.add_argument("--kernel_stride",        type=int,   default=None)
    _cli.add_argument("--wandb_project",        type=str,   default=None)
    _cli.add_argument("--wandb_run_name",       type=str,   default=None)

    _known, _sb_argv = _cli.parse_known_args(sys.argv[1:])

    _OVERRIDE_MAP = {
        "epochs":               "N_epochs",
        "lr":                   "lr",
        "batch_size":           "batch_size",
        "clip_grad_norm":       "clip_grad_norm",
        "seed":                 "seed",
        "precision":            "precision",
        "early_stop_patience":  "early_stop_patience",
        "freeze_encoder_epochs":"freeze_encoder_epochs",
        "valid_split":          "valid_split",
        "split_seed":           "split_seed",
        "data_folder":          "data_folder",
        "experiment_name":      "experiment_name",
        "num_spks":             "num_spks",
        "sample_rate":          "sample_rate",
        "skip_prep":            "skip_prep",
        "use_wavedrop":         "use_wavedrop",
        "use_speedperturb":     "use_speedperturb",
        "use_rand_shift":       "use_rand_shift",
        "dynamic_mixing":       "dynamic_mixing",
        "N_encoder_out":        "N_encoder_out",
        "kernel_size":          "kernel_size",
        "kernel_stride":        "kernel_stride",
    }

    _cli_overrides = []
    for _dest, _yaml_key in _OVERRIDE_MAP.items():
        _val = getattr(_known, _dest, None)
        if _val is not None:
            if isinstance(_val, str):
                _escaped = _val.replace("\\", "\\\\").replace('"', '\\"')
                _cli_overrides.append(f'{_yaml_key}: "{_escaped}"')
            elif isinstance(_val, bool):
                _cli_overrides.append(f"{_yaml_key}: {'True' if _val else 'False'}")
            else:
                _cli_overrides.append(f"{_yaml_key}: {_val}")

    _wandb_project  = _known.wandb_project  or "sepformer-speech-separation"
    _wandb_run_name = _known.wandb_run_name

    hparams_file, run_opts, overrides = sb.parse_arguments(_sb_argv)

    if _cli_overrides:
        overrides = "\n".join(_cli_overrides) + ("\n" + overrides if overrides else "")

    _data_folder_in_overrides = any(
        "data_folder" in line for line in (_cli_overrides + [overrides or ""])
    )
    if not _data_folder_in_overrides:
        sys.exit(
            "\n[ERROR] 'data_folder' is required.\n"
            "Example: python train_v2.py hparams/sepformer-finetune.yaml "
            "--data_folder F:/ducdataset\n"
        )

    with open(hparams_file, encoding="utf-8") as fin:
        try:
            hparams = load_hyperpyyaml(fin, overrides)
        except ValueError as _e:
            _msg = str(_e)
            if "PLACEHOLDER" in _msg:
                _key = _msg.split("'")[1] if "'" in _msg else "unknown"
                sys.exit(
                    f"\n[ERROR] YAML key '{_key}' is required but not set.\n"
                    f"Pass it as:  --{_key} VALUE\n"
                )
            raise

    sb.utils.distributed.ddp_init_group(run_opts)
    logger = get_logger(__name__)

    wandb.init(
        project=_wandb_project,
        name=_wandb_run_name,
        config={k: v for k, v in hparams.items() if isinstance(v, (int, float, str, bool))},
    )

    wandb.define_metric("epoch")
    wandb.define_metric("train_loss",         step_metric="epoch", summary="min")
    wandb.define_metric("val_loss",           step_metric="epoch", summary="min")
    wandb.define_metric("lr",                 step_metric="epoch", summary="last")
    wandb.define_metric("grad_norm",          step_metric="epoch", summary="mean")
    wandb.define_metric("train_step_loss",    summary="min")
    wandb.define_metric("train_step_si-snr",  summary="max")
    wandb.define_metric("encoder_frozen",     summary="last")

    wandb.save(
        os.path.join(hparams["output_folder"], "**", "*"),
        base_path=hparams["output_folder"],
        policy="live",
    )

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    if run_opts.device == "cpu" and hparams.get("precision") == "fp16":
        hparams["precision"] = "bf16"

    # ── Data preparation (v2) ────────────────────────────────────────────
    from prepare_data_v2 import prepare_wsjmix_v2

    run_on_main(
        prepare_wsjmix_v2,
        kwargs={
            "datapath":   hparams["data_folder"],
            "savepath":   hparams["save_folder"],
            "n_spks":     hparams["num_spks"],
            "skip_prep":  hparams["skip_prep"],
            "fs":         hparams["sample_rate"],
            "valid_split":hparams.get("valid_split", 0.15),
            "split_seed": hparams.get("split_seed", 1234),
        },
    )

    train_data, valid_data, test_data = dataio_prep(hparams)

    # ── Load pretrained weights ──────────────────────────────────────────
    if "pretrained_separator" in hparams:
        print("Loading pretrained separator weights …")
        run_on_main(hparams["pretrained_separator"].collect_files)
        hparams["pretrained_separator"].load_collected()

    # ── Brain ────────────────────────────────────────────────────────────
    separator = Separation(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    if "pretrained_separator" not in hparams:
        for module in separator.modules.values():
            separator.reset_layer_recursively(module)

    # ── Signal handlers ──────────────────────────────────────────────────
    def _handle_signal(signum, frame):
        logger.warning(f"Signal {signum} received — emergency save …")
        separator._emergency_save(
            separator.hparams.epoch_counter.current,
            reason=f"signal_{signum}",
        )
        if wandb.run is not None:
            wandb.finish()
        sys.exit(1)

    signal.signal(signal.SIGINT, _handle_signal)
    try:
        signal.signal(signal.SIGTERM, _handle_signal)
    except (OSError, ValueError):
        pass

    # ── Pre-training summary ─────────────────────────────────────────────
    n_train = len(train_data)
    n_valid = len(valid_data)
    n_test  = len(test_data)

    _sample_lens = []
    for _i in range(min(50, n_train)):
        try:
            _item = train_data[_i]
            _sig = _item["mix_sig"] if isinstance(_item, dict) else _item.mix_sig
            if isinstance(_sig, tuple):
                _sig = _sig[0]
            _sample_lens.append(_sig.shape[-1])
        except Exception:
            pass
    avg_waveform_len = float(np.mean(_sample_lens)) if _sample_lens else 0.0
    avg_duration_sec = avg_waveform_len / hparams["sample_rate"] if avg_waveform_len > 0 else 0.0
    total_params     = sum(p.numel() for p in separator.modules.parameters())
    trainable_params = sum(p.numel() for p in separator.modules.parameters() if p.requires_grad)

    freeze_until = hparams.get("freeze_encoder_epochs", 0)
    logger.info("=" * 60)
    logger.info("PRE-TRAINING SUMMARY (train_v2.py)")
    logger.info("=" * 60)
    logger.info(f"  Train / Valid / Test   : {n_train} / {n_valid} / {n_test}")
    logger.info(f"  Sample rate            : {hparams['sample_rate']} Hz")
    logger.info(f"  Avg duration           : {avg_duration_sec:.2f} s")
    logger.info(f"  Num speakers           : {hparams['num_spks']}")
    logger.info(f"  Batch size             : {hparams['dataloader_opts'].get('batch_size')}")
    logger.info(f"  Epochs                 : {hparams['N_epochs']}")
    logger.info(f"  Learning rate          : {hparams['lr']}")
    logger.info(f"  Freeze encoder epochs  : {freeze_until}")
    logger.info(f"  Early stop patience    : {hparams.get('early_stop_patience', 15)}")
    logger.info(f"  Speed perturb          : {hparams.get('use_speedperturb', False)}")
    logger.info(f"  WaveDrop               : {hparams.get('use_wavedrop', False)}")
    logger.info(f"  Limit train len        : {hparams.get('limit_training_signal_len', False)}")
    logger.info(f"  Valid split            : {hparams.get('valid_split', 0.15)}")
    logger.info(f"  Total params           : {total_params:,}")
    logger.info(f"  Trainable params       : {trainable_params:,}")
    logger.info("=" * 60)

    if wandb.run is not None:
        wandb.config.update({
            "train_samples":      n_train,
            "valid_samples":      n_valid,
            "test_samples":       n_test,
            "sample_rate":        hparams["sample_rate"],
            "avg_duration_sec":   round(avg_duration_sec, 2),
            "num_spks":           hparams["num_spks"],
            "batch_size":         hparams["dataloader_opts"].get("batch_size"),
            "N_epochs":           hparams["N_epochs"],
            "lr":                 hparams["lr"],
            "freeze_encoder_epochs": freeze_until,
            "early_stop_patience":   hparams.get("early_stop_patience", 15),
            "valid_split":           hparams.get("valid_split", 0.15),
            "use_speedperturb":      hparams.get("use_speedperturb", False),
            "use_wavedrop":          hparams.get("use_wavedrop", False),
            "total_params":          total_params,
            "trainable_params":      trainable_params,
        }, allow_val_change=True)

    # ── Train ────────────────────────────────────────────────────────────
    try:
        separator.fit(
            separator.hparams.epoch_counter,
            train_data,
            valid_set=valid_data,
            train_loader_kwargs=hparams["dataloader_opts"],
            valid_loader_kwargs=hparams["dataloader_opts"],
        )
    except Exception as exc:
        logger.error(f"Training failed: {exc}")
        separator._emergency_save(
            separator.hparams.epoch_counter.current, reason="exception"
        )
        if wandb.run is not None:
            wandb.finish()
        raise

    # ── Eval ─────────────────────────────────────────────────────────────
    separator.evaluate(test_data, min_key="si-snr")
    separator.save_results(test_data)

    if wandb.run is not None:
        best_loss = getattr(separator, "_best_val_loss", None)
        if best_loss is not None:
            wandb.run.summary["final_best_val_loss"] = best_loss
            wandb.run.summary["final_best_si_snr_dB"] = -best_loss
        wandb.finish()
