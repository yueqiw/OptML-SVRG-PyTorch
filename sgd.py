from torch.optim import Optimizer

# adapted from pytorch official implementation. 
class SGD_Simple(Optimizer):
    r"""Implements stochastic gradient descent.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
    """
    def __init__(self, params, lr, weight_decay=0):
        print("Using optimizer: SGD_Simple")
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight decay: {}".format(weight_decay))

        defaults = dict(lr=lr, weight_decay=weight_decay)
        
        super(SGD_Simple, self).__init__(params, defaults)

    def step(self):
        """Performs a single optimization step.
        """
        for group in self.param_groups:
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                    
                p.data.add_(-group['lr'], d_p)