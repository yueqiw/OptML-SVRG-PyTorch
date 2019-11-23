from torch.optim import Optimizer

class SGD_Simple(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
    """
    def __init__(self, params, lr):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr)
        
        super(SGD_Simple, self).__init__(params, defaults)

    def step(self):
        """Performs a single optimization step.
        """

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data.add_(-group['lr'], d_p)