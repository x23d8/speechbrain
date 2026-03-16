import torch

def pairwise_si_snr(estimates, targets, eps=1e-8):
    """
    estimates: [B, C, T]
    targets:   [B, C, T]
    return: pairwise SI-SNR matrix [B, C, C]
    """

    B, C, T = estimates.shape

    # zero-mean
    estimates = estimates - estimates.mean(dim=2, keepdim=True)
    targets = targets - targets.mean(dim=2, keepdim=True)

    # expand dims for pairwise computation
    est = estimates.unsqueeze(2)  # [B, C, 1, T]
    tgt = targets.unsqueeze(1)    # [B, 1, C, T]

    # projection
    dot = torch.sum(est * tgt, dim=3, keepdim=True)
    tgt_energy = torch.sum(tgt ** 2, dim=3, keepdim=True) + eps

    proj = dot * tgt / tgt_energy
    noise = est - proj

    ratio = torch.sum(proj ** 2, dim=3) / (torch.sum(noise ** 2, dim=3) + eps)

    return 10 * torch.log10(ratio + eps)

def pit_sisnr_loss(estimates, targets):
    """
    estimates: [B, C, T]
    targets:   [B, C, T]
    """

    pairwise_snr = pairwise_si_snr(estimates, targets)

    perms = torch.tensor([[0,1],[1,0]], device=estimates.device)

    snr_set = []

    for perm in perms:
        snr = pairwise_snr[:, torch.arange(2, device=estimates.device), perm]
        snr_set.append(torch.mean(snr, dim=1))

    snr_set = torch.stack(snr_set, dim=1)

    max_snr, _ = torch.max(snr_set, dim=1)

    loss = -max_snr.mean()

    return loss