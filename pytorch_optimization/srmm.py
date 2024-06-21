import math
import torch
from torch.optim import Optimizer


class SRMM(Optimizer):
    """Implements Stochastic Regularized Majorization-Minimization algorithm.
    Current version only implements the "double-averaged PSGD" variant of SRMM.
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        ilr (float, optional): inverse learning rate (1/L)
        pr (float, optional): proximal regularization coefficient (lambda) # stepsize = 1/(ilr + pr)
        beta (Tuple[float], optional): adaptivity weight = step ** (-beta)
        L (float, optional): internal memory length for moving average; None for no refreshing
        lr (float, optional): dummay learning rate (not used)
    """

    def __init__(self, params, ilr=1e-3, pr=0, beta=(0.5), L=100, lr=0.1):
        if not 0.0 <= ilr: # Not used in SRMM
            raise ValueError("Invalid learning rate: {}".format(ilr))
        if not 0.0 <= pr: # Not used in SRMM
            raise ValueError("Invalid learning rate: {}".format(pr))
        if not 0.0 <= beta:
            raise ValueError("Invalid beta parameter at index 0: {}".format(beta))
        if (L is not None) and  (0.0 > L):
            raise ValueError("Invalid memory length parameter at index 0: {}".format(L))
        defaults = dict(ilr=ilr, pr=pr, beta=beta, L=L, lr=lr)
        super(SRMM, self).__init__(params, defaults)

        self.base_lrs = list(map(lambda group: group['ilr'], self.param_groups))

    def __setstate__(self, state):
        super(SRMM, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()


        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]


                # State initialization
                if (len(state) == 0):
                    state['step'] = 0
                    # moving average of gradient values
                    state['mov_avg_grad'] = torch.zeros_like(p.data)
                    # moving average of parameter values (averaged up to the previous iter)
                    state['mov_avg_param'] = torch.zeros_like(p.data)
                    # moving average of parameter values after proximal regularization
                    # (averaged up to the previous iter)
                    #state['mov_avg_param_reg'] = torch.zeros_like(p.data)

                mov_avg_grad = state['mov_avg_grad']
                mov_avg_param = state['mov_avg_param']
                #mov_avg_param_reg = state['mov_avg_param_reg']
                state['step'] += 1
                t = float(state['step']) # current iteration count
                beta = group['beta']
                if group['L'] is None:
                    wt = (t+1)**(-beta)
                else:
                    wt = (t+1 % group['L'])**(-beta)
                    #print("group['L']={}, t={}".format(group['L'], t))
                mov_avg_grad.mul_(1-wt)
                mov_avg_grad.add_(grad, alpha=wt)
                param = p.data # previous parameter
                mov_avg_param.mul_(1-wt)
                mov_avg_param.add_(param, alpha=wt)

                wt2 = group['pr']/(group['ilr']+group['pr']) # = 0 if pr = 0
                #print("pr={}".format(group['pr']))
                #print("lr={}".format(group['lr']))
                #print("wt2={}".format(wt2))
                #print("beta={}".format(beta))
                #print("wt={}".format(wt))
                #mov_avg_param0 = mov_avg_param.clone()
                mov_avg_param_reg = mov_avg_param.clone()
                mov_avg_param_reg.mul_(1-wt2)
                mov_avg_param_reg.add_(param, alpha=wt2)

                step_size = 1/(group['ilr']+group['pr']) # =1/group['ilr'] if pr = 0
                #print("step_size={}".format(step_size))
                p.data = mov_avg_param_reg.add_(-mov_avg_grad, alpha=step_size)
                #p.data = mov_avg_param0.add_(-mov_avg_grad, alpha=step_size)

                #state['mov_avg_grad'] = mov_avg_grad
                #state['mov_avg_param'] = mov_avg_param
                #state['mov_avg_param_reg'] = mov_avg_param_reg
        return loss


class SMM(Optimizer):
    """Implements Stochastic Majorization-Minimization algorithm.
    Current version only implements the "double-averaged PSGD" variant of SRMM.
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): constant step size
        pr (float, optional): proximal regularization coefficient
        beta (Tuple[float], optional): adaptivity weight = step ** (-beta)
        L (float, optional): internal memory length for moving average; None for no refreshing
    """

    def __init__(self, params, lr=1e-3, pr=0, beta=(0.5), L=100):
        if not 0.0 <= lr: # Not used in SRMM
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= pr: # Not used in SRMM
            raise ValueError("Invalid learning rate: {}".format(pr))
        if not 0.0 <= beta:
            raise ValueError("Invalid beta parameter at index 0: {}".format(beta))
        if (L is not None) and  (0.0 > L):
            raise ValueError("Invalid memory length parameter at index 0: {}".format(L))
        defaults = dict(lr=lr, pr=pr, beta=beta, L=L)
        super(SRMM, self).__init__(params, defaults)

        self.base_lrs = list(map(lambda group: group['lr'], self.param_groups))

    def __setstate__(self, state):
        super(SRMM, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()


        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]


                # State initialization
                if (len(state) == 0):
                    state['step'] = 0
                    # moving average of gradient values
                    state['mov_avg_grad'] = torch.zeros_like(p.data)
                    # moving average of parameter values (averaged up to the previous iter)
                    state['mov_avg_param'] = torch.zeros_like(p.data)
                    # moving average of parameter values after proximal regularization
                    # (averaged up to the previous iter)
                    #state['mov_avg_param_reg'] = torch.zeros_like(p.data)

                mov_avg_grad = state['mov_avg_grad']
                mov_avg_param = state['mov_avg_param']
                #mov_avg_param_reg = state['mov_avg_param_reg']
                state['step'] += 1
                t = float(state['step']) # current iteration count
                beta = group['beta']
                if group['L'] is None:
                    wt = (t+1)**(-beta)
                else:
                    wt = (t+1 % group['L'])**(-beta)
                    #print("group['L']={}, t={}".format(group['L'], t))
                mov_avg_grad.mul_(1-wt)
                mov_avg_grad.add_(grad, alpha=wt)
                param = p.data # previous parameter
                mov_avg_param.mul_(1-wt)
                mov_avg_param.add_(param, alpha=wt)

                #wt2 = group['pr']/(group['lr']+group['pr'])
                #print("pr={}".format(group['pr']))
                #print("lr={}".format(group['lr']))
                #print("wt2={}".format(wt2))
                #print("beta={}".format(beta))
                #print("wt={}".format(wt))
                #mov_avg_param_reg = mov_avg_param.clone()
                #mov_avg_param_reg.mul_(1-wt2)
                #mov_avg_param_reg.add_(param, alpha=wt2)

                step_size = group['lr']
                #p.data = mov_avg_param_reg.add_(-mov_avg_grad, alpha=step_size)
                p.data = mov_avg_param.add_(-mov_avg_grad, alpha=step_size)

                state['mov_avg_grad'] = mov_avg_grad
                state['mov_avg_param'] = mov_avg_param
                #state['mov_avg_param_reg'] = mov_avg_param_reg

        return loss