# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Custom loss functions"""

import torch
from torch.nn import functional as F

"""Custom loss function from mean teacher project"""
def softmax_kl_loss(input_logits, target_logits, eps=False, k=2, tau=0, function='softmax', reduction='batchmean'):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax, reduction=reduction)

"""Our loss"""
def softmax_kl_loss_sl2(input_logits, target_logits, eps=0.35, k=-1, tau=1/7, function='softmax', reduction='batchmean'):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    #import pdb; pdb.set_trace()
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)

    if function == 'softmax':
        target = F.softmax(target_logits, dim=1)
    else:
        raise('Unsupported function')

    N, C = target_logits.shape
    smooth_labels = target.gt(tau).float() * target
    smooth_labels = smooth_labels / smooth_labels.sum(1).unsqueeze(1)
    smooth_labels = smooth_labels * (1 - eps)
    Ks = target.gt(tau).sum(1).unsqueeze(1)
    Ks = Ks + Ks.eq(0).int()
    small_mask = 1 - target.gt(tau).float()
    smooth_labels = smooth_labels + small_mask * (eps / (C - Ks.float()))

    return F.kl_div(input_log_softmax, smooth_labels, reduction=reduction)

