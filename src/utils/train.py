import torch
import torch.nn.functional as F


def focal_loss_with_logits(
    logits,
    target,
    alpha=0.25,
    gamma=2.0,
    eps=1e-6
):
    """
        This is a classification loss used for the heatmap head
        (predicting neuron vs. background per voxel).
        It's an improved version of binary cross-entropy (BCE),
        designed for imbalanced data (many more background voxels
        than neuron voxels
    """
    # convert raw lagits into probabilities between 0 and 1
    p = torch.sigmoid(logits)

    # pt = model's probability for the correct class
    # if target=1 (neuron), pt=p; if target=0 (background), pt=1-p
    pt = p*target + (1-alpha)*(1-target)

    # weighting factor for imbalance: alpha for positives, (1-alpha) for negatives
    w = alpha*target + (1-alpha)*(1-target)

    # Focal loss formual:
    # - cross-entropy term (log of predicted prob)
    # - weighted by alpha
    # - downweighted for "easy" examples using (1-py)^gamma
    return (
        -w * (1-pt).pow(gamma) *
        (
            target*torch.log(p+eps) + 
            (1-target)*torch.log(1-p+eps)
        )
    ).mean()


def masked_smoothl1(pred, target, mask, beta=1.0):
    """
        This is a regression loss used for the offsets head 
        (predicting (Δz, Δy, Δx) inside neuron spheres).
        It's a Smooth L1 loss (Huber loss), applied only where
        the mask=1 (inside supervision spheres).
    """
    # compute difference only where mask = 1
    diff = (pred - target) * mask # same shape as pred (B, D, Z, Y, X)

    # Absolute value of difference
    absd = diff.abs()

    # SmoothL1 formula:
    #   if |x| < beta -> 0.5 * (x^2 / beta) (quadratic, smooth near 0)
    #   else -> |x| - 0.5*beta              (linear, robust to outliers)
    l = torch.where(absd < beta, 0.5*absd**2/beta, absd - 0.5*beta)

    # Normalisation: divide by number of supervised voxels
    demon = mask.sum().clamp_min(1.0)
    return l.sum() / demon
