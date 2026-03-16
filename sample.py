import argparse
import torch
import torchaudio
from torch.utils.data import DataLoader
from torchaudio.pipelines import CONVTASNET_BASE_LIBRI2MIX
from dataset import VIVOSMIX, pad_collate_fn
from loss import pit_sisnr_loss

import wandb
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# ==========================================
# Argument Parser
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune ConvTasNet on VIVOSMIX dataset")

    # Training hyperparameters
    parser.add_argument("--batch_size",     type=int,   default=1,      help="Batch size for training")
    parser.add_argument("--learning_rate",  type=float, default=1e-4,   help="Initial learning rate")
    parser.add_argument("--epochs",         type=int,   default=50,     help="Number of training epochs")
    parser.add_argument("--max_norm",       type=float, default=5.0,    help="Max gradient norm for clipping")
    parser.add_argument("--freeze_encoder", action="store_true",        help="Freeze ConvTasNet encoder during fine-tuning")
    parser.add_argument("--dropout", type=float, default=0.0,           help="Dropout probability applied to Conv1d/Linear layers in separator")

    # Dataset
    parser.add_argument("--data_root", type=str,
                        default=r"D:\Duc_Data\Study\FPT_University_Course\SPRING26_Semester_7\SLP301\code\SpeechSeparation\source_code\data\datasets",
                        help="Root directory of the dataset")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of DataLoader workers")

    # Scheduler (ReduceLROnPlateau)
    parser.add_argument("--scheduler_factor",  type=float, default=0.5, help="ReduceLROnPlateau: LR decay factor")
    parser.add_argument("--scheduler_min_lr",  type=float, default=1e-6, help="ReduceLROnPlateau: minimum LR")
    parser.add_argument("--patience",       type=int,   default=3)

    # Checkpoint
    parser.add_argument("--save_path", type=str, default=r"models\finetuned\convtasnet_best.pth", help="Path to save best model")

    # W&B
    parser.add_argument("--wandb_project",    type=str, default="speech-separation",  help="W&B project name")
    parser.add_argument("--wandb_run_name",   type=str, default="convtasnet-vivosmix-v1", help="W&B run name")
    parser.add_argument("--wandb_entity",     type=str, default="slp301_ai1802",                 help="W&B entity (team/user)")
    parser.add_argument("--wandb_disabled",   action="store_true",                    help="Disable W&B logging")

    return parser.parse_args()


# ==========================================
# Training for one epoch
# ==========================================
def train_one_epoch(model, loader, optimizer, device, max_norm, epoch, total_epochs):
    model.train()

    running_loss = 0.0
    total_grad_norm = 0.0

    pbar = tqdm(loader, desc=f"[Train] Epoch {epoch}/{total_epochs}", leave=False, dynamic_ncols=True)

    for batch_idx, (mixtures, sources) in enumerate(pbar):
        mixtures = mixtures.to(device)   # [B, 1, T]
        sources  = sources.to(device)    # [B, N_src, T]

        optimizer.zero_grad()

        # Forward
        estimates = model(mixtures)      # [B, N_src, T]

        # Loss
        loss = pit_sisnr_loss(estimates, sources)

        # Backward
        loss.backward()

        # Gradient clipping — capture norm before clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm).item()

        optimizer.step()

        running_loss   += loss.item()
        total_grad_norm += grad_norm
        avg_loss = running_loss / (batch_idx + 1)

        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{current_lr:.2e}")

    epoch_loss      = running_loss   / len(loader)
    epoch_grad_norm = total_grad_norm / len(loader)
    return epoch_loss, epoch_grad_norm


# ==========================================
# Evaluation for one epoch
# ==========================================
def evaluate(model, loader, device, epoch, total_epochs):
    model.eval()

    running_loss = 0.0
    pbar = tqdm(loader, desc=f"[Eval]  Epoch {epoch}/{total_epochs}", leave=False, dynamic_ncols=True)

    with torch.no_grad():
        for batch_idx, (mixtures, sources) in enumerate(pbar):
            mixtures = mixtures.to(device)
            sources  = sources.to(device)

            estimates = model(mixtures)
            loss      = pit_sisnr_loss(estimates, sources)

            running_loss += loss.item()
            avg_loss      = running_loss / (batch_idx + 1)
            pbar.set_postfix(val_loss=f"{avg_loss:.4f}")

    return running_loss / len(loader)


# ==========================================
# Dropout Hooks
# ==========================================

def register_dropout_hooks(module, p: float):
    """
    Inject dropout sau output của mỗi Conv1d/Linear trong module.
    Chỉ active khi model.training = True (tự động tắt lúc eval).
    Trả về list hooks để có thể remove sau nếu cần.
    """
    hooks = []

    def make_hook(dropout_layer):
        def hook(mod, inp, output):
            return dropout_layer(output)
        return hook

    for name, mod in module.named_modules():
        if isinstance(mod, (torch.nn.Conv1d, torch.nn.Linear)):
            dropout_layer = torch.nn.Dropout(p=p)
            h = mod.register_forward_hook(make_hook(dropout_layer))
            hooks.append(h)

    print(f"  Registered dropout (p={p}) on {len(hooks)} Conv1d/Linear layers in separator")
    return hooks


# ==========================================
# Main
# ==========================================
def main():
    args = parse_args()

    # ── W&B ────────────────────────────────
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        entity=args.wandb_entity,
        config=vars(args),
        mode="disabled" if args.wandb_disabled else "online",
    )

    # ── Device ─────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Model ──────────────────────────────
    print("Loading pre-trained ConvTasNet...")
    bundle      = CONVTASNET_BASE_LIBRI2MIX
    model       = bundle.get_model().to(device)
    sample_rate = bundle.sample_rate          # 8 000 Hz
    print(f"Sample rate: {sample_rate} Hz")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)

    if args.freeze_encoder:
        print("Freezing encoder...")
        base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        for param in base_model.encoder.parameters():
            param.requires_grad = False

    # ── Dropout hooks ──────────────────────
    dropout_hooks = []   
    if args.dropout > 0.0:
        print(f"\nInjecting dropout (p={args.dropout}) into mask_generator...")
        base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        dropout_hooks = register_dropout_hooks(base_model.mask_generator, p=args.dropout)
    
    def count_parameters(module):
        return sum(p.numel() for p in module.parameters())
    
    def count_trainable_parameters(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    
    print("\n===== Model Parameter Summary =====")
    
    encoder_params = count_parameters(base_model.encoder)
    separator_params = count_parameters(base_model.mask_generator)
    decoder_params = count_parameters(base_model.decoder)
    
    print(f"Encoder params   : {encoder_params:,}")
    print(f"Separator params : {separator_params:,}")
    print(f"Decoder params   : {decoder_params:,}")
    
    total_params = encoder_params + separator_params + decoder_params
    print(f"Total params     : {total_params:,}")

    # ── Datasets & Loaders ─────────────────
    train_dataset = VIVOSMIX(root=args.data_root, subset="train")
    test_dataset  = VIVOSMIX(root=args.data_root, subset="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=pad_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pad_collate_fn,
    )

    # ── Optimizer + Scheduler ──────────────
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.patience,
        min_lr=args.scheduler_min_lr,
    )

    # ── Training Loop ──────────────────────
    best_val_loss = float("inf")

    print(f"\nStarting fine-tuning on {device} for {args.epochs} epochs...\n")

    epoch_pbar = tqdm(range(1, args.epochs + 1), desc="Epochs", dynamic_ncols=True)

    for epoch in epoch_pbar:
        # Train
        train_loss, grad_norm = train_one_epoch(
            model, train_loader, optimizer, device, args.max_norm, epoch, args.epochs
        )

        # Eval
        val_loss = evaluate(model, test_loader, device, epoch, args.epochs)

        # Scheduler step on val_loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # ── tqdm outer bar postfix ──────────
        epoch_pbar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            lr=f"{current_lr:.2e}",
        )

        # ── W&B logging ────────────────────
        wandb.log({
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "lr":         current_lr,
            "grad_norm":  grad_norm,
        })

        # ── Checkpoint ─────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state_to_save = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_to_save, args.save_path)
            
            # Log best model.pt
            artifact = wandb.Artifact(
                "convtasnet-vivosmix",
                type="model",
                metadata={"epoch": epoch, "val_loss": val_loss}
            )
            artifact.add_file(args.save_path)
            wandb.log_artifact(artifact)
            tqdm.write(f"  ✔ Epoch {epoch}: new best val_loss={best_val_loss:.4f} — model saved to '{args.save_path}'")
        else:
            tqdm.write(f"  Epoch {epoch}: val_loss={val_loss:.4f}  (no improvement)")

    print(f"\nFine-tuning complete! Best val_loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {args.save_path}")
    wandb.finish()


if __name__ == "__main__":
    main()