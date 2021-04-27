import torch


def calculate_kl(
        drops_proba: torch.FloatTensor,
        prior_proba: float,
        kl_coeff: float = 1
) -> torch.FloatTensor:
    kl_div = kl_coeff * (
            drops_proba * torch.log(torch.clamp(drops_proba / prior_proba, min=1e-8))
            + (1 - drops_proba)
            * torch.log(torch.clamp((1 - drops_proba) / (1 - prior_proba), min=1e-8))
    ).sum(dim=1)

    return kl_div


def variational_translation_loss(
        src_drops_proba: torch.FloatTensor,
        tgt_drops_proba: torch.FloatTensor,
        src_possible_merges: (torch.FloatTensor, torch.FloatTensor),
        tgt_possible_merges: (torch.FloatTensor, torch.FloatTensor),
        kl_coeff: float = 1,
        src_prior_proba: float = None,
        tgt_prior_proba: float = None
) -> (torch.FloatTensor, torch.FloatTensor, torch.FloatTensor):
    # Invert tensor to get probability of skipping merge.
    # It needs because initially drops_proba is a tensor of leaving merge probabilities
    src_drops_proba = 1 - src_drops_proba
    tgt_drops_proba = 1 - tgt_drops_proba

    src_used_per_id = src_possible_merges[0]
    src_missed_per_id = src_possible_merges[1]

    tgt_used_per_id = tgt_possible_merges[0]
    tgt_missed_per_id = tgt_possible_merges[1]

    rl_loss = (
            src_missed_per_id * torch.log(torch.clamp(src_drops_proba, min=1e-8)) +
            src_used_per_id * torch.log(torch.clamp(1 - src_drops_proba, min=1e-8))
    ).sum(dim=1) + (
            tgt_missed_per_id * torch.log(torch.clamp(tgt_drops_proba, min=1e-8)) +
            tgt_used_per_id * torch.log(torch.clamp(1 - tgt_drops_proba, min=1e-8))
    ).sum(dim=1)

    src_kl_loss = calculate_kl(src_drops_proba, prior_proba=src_prior_proba, kl_coeff=kl_coeff)

    tgt_kl_loss = calculate_kl(tgt_drops_proba, prior_proba=tgt_prior_proba, kl_coeff=kl_coeff)

    return rl_loss, src_kl_loss, tgt_kl_loss
