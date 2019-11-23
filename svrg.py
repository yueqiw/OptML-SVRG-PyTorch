from torch.optim import Optimizer
import copy


#First optimization class for calculating the gradient of the one random selected sample.
class SVRG_k(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
    """
    def __init__(self, params, lr):
        self.u = None
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr)
        
        super(SVRG_k, self).__init__(params, defaults)
    
    def return_param(self):
          return self.param_groups
    
    def set_param(self, new_params):
          self.param_groups = new_params

    def set_u(self, new_u):
        if self.u is None:
          self.u = copy.deepcopy(new_u)
        for u_group, new_group in zip(self.u, new_u):  
          for u, new_u in zip(u_group['params'], new_group['params']):
            u.grad = new_u.grad.clone()
    
    def return_u(self):
          return self.u

    def step(self, params):
        """Performs a single optimization step.
        """

        for group, new_group, u_group in zip(self.param_groups, params, self.u):  
            for p, q, u in zip(group['params'], new_group['params'], u_group['params']):
                if p.grad is None:
                    continue
                if q.grad is None:
                    continue

                new_d = p.grad.data - q.grad.data + u.grad.data
                p.data.add_(-group['lr'], new_d)

#optimization class for calculating the mean gradient of all the sample.
class SVRG_0(Optimizer):
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
        
        super(SVRG_0, self).__init__(params, defaults)
      
    def return_param(self):
          return self.param_groups
    
    def set_param(self, new_params):
         for group, new_group in zip(self.param_groups, new_params):  #[1,2,3] [4,5,6] => [(1,4), (2,5)]
            for p, q in zip(group['params'], new_group['params']):
                  p.data[:] = q.data[:]