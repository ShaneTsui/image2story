def l2norm(input, p=2.0, dim=1, eps=1e-12):
    """
    Compute L2 norm, row-wise
    """
    return input / input.norm(p, dim).clamp(min=eps).view((-1, 1))