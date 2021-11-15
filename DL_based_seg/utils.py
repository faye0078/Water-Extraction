class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def compute_params(model):
    """Compute number of parameters"""
    n_total_params = 0
    n_aux_params = 0
    for name, m in model.named_parameters():
        n_elem = m.numel()
        if "aux" in name:
            n_aux_params += n_elem
        n_total_params += n_elem
    return n_total_params, n_total_params - n_aux_params