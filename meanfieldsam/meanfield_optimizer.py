from abc import ABC, abstractmethod
from copy import deepcopy
import torch
from torch.optim import Optimizer
from torch.nn.init import constant_
from torch.distributions import Normal
import numpy as np


class MeanFieldOptimizer(Optimizer, ABC):
    """Abstract base class for Mean-Field Type Optimizers, including
    Mean-Field Variational Inference, RandomSAM, MixSAM.
    Subclasses must implement the get_perturbation method that returns
    a D-dimensional parameter vector of norm sqrt(D), shaped in the same
    format as in param_groups. And the populate_gradients_for_Sigma
    method."""

    def __init__(self, params, base_optimizer, lr_sigma = 0.01, sigma_prior = 10, rho=0.05, kl_div_weight = 1, **kwargs):
       
        if not lr_sigma >= 0.0:
            raise ValueError(f"Invalid lr_sigma, should be non-negative: {lr_sigma}")
        if not sigma_prior >= 0.0:
            raise ValueError(f"Invalid sigma_prior, should be non-negative: {sigma_prior}")
        if not kl_div_weight >= 0.0:
            raise ValueError(f"Invalid kl_div_weight, should be non-negative: {kl_div_weight}")
          
        self.sigma_prior = sigma_prior
        self.kl_div_weight = kl_div_weight
        self.rho = rho
        defaults = dict(lr_sigma=lr_sigma, **kwargs)
        super(MeanFieldOptimizer, self).__init__(params, defaults)

        self.num_params = 0
        for param_group in self.param_groups:
            for param in param_group['params']:
                if param.requires_grad:
                    self.num_params += param.numel()
                
        self.M_param_groups = []
        for param_group in self.param_groups:
            M_param_group = param_group.copy()
            M_param_group["params"] = [
                torch.ones_like(tensor, requires_grad=tensor.requires_grad)
                for tensor in param_group['params']
            ]
            for M in M_param_group["params"]:
                constant_(M, self.rho / np.sqrt(self.num_params))
            M_param_group['lr'] = M_param_group['lr_sigma']
            M_param_group.pop('lr_sigma')
            param_group.pop('lr_sigma')
            self.M_param_groups.append(M_param_group)

        self.base_optimizer = base_optimizer(self.param_groups + self.M_param_groups, **kwargs)

        self.eps = max(torch.finfo(
            self.param_groups[0]['params'][0].dtype).eps, 1e-12)

        self.shared_device = self.param_groups[0]["params"][0].device

    def step(self, closure):        
        self._populate_gradients_for_mean(closure)
        self._populate_gradients_for_Sigma()
        self.base_optimizer.step()

    @torch.no_grad()
    def _populate_gradients_for_mean(self, closure):
        """This function populates the gradients of the mean parameter in
        `param_groups`. It first saves the original parameter values for later,
        applies the perturbation, zeroes out the gradients, calls the closure to
        backpropagate, then restores the mean parameters to their original values.
        """

        self.perturbation_groups = self._get_perturbation()
        for param_group, M_param_group, perturbation_group in zip(self.param_groups, self.M_param_groups, self.perturbation_groups):
            for param, M, perturbation in zip(param_group['params'], M_param_group['params'], perturbation_group['params']):
                if param.requires_grad:
                    self.state[param]["old_p"] = param.data.clone()
                    param.add_(torch.abs(M)*perturbation)
        
        self.base_optimizer.zero_grad()

        with torch.enable_grad():
          closure()
        
        for param_group in self.param_groups:
            for param in param_group["params"]:
                if param.requires_grad:
                    param.data = self.state[param]["old_p"]
                    param.grad.add_(2 * self.kl_div_weight * param.detach().clone() / self.sigma_prior**2)

    @torch.no_grad()
    @abstractmethod
    def _populate_gradients_for_Sigma(self):
        """Abstract method all subclasses must implement for calculating the
        gradients for Sigma. This may be nothing."""
        pass

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    @abstractmethod
    def _get_perturbation(self):
        """Abstract method all subclasses must implement for calculating the
        perturbation direction in the Mean-Field optimization algorithm."""
        pass


class MFVI(MeanFieldOptimizer):
    """Implements Mean Field Variational Optimization.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        base_optimizer (torch.optim.Optimizer): base optimizer to make gradient
            updates to mean and field parameters
        lr_Sigma (float, optional): learning rate for the field parameter
            (default: 1e-2)
        sigma_prior (float, optional): standard deviation of the prior, controls
            weight decay of mean parameter and strength of regularisation on
            the field parameter (default: 1)
        kl_div_weight (float, optional): weight given to the KL divergence term.
            Should be chosen roughly proportional to 1/N where N is the
            total number of datasets (default: 0.01), assuming loss function
            calculates mean log loss.
        """
    
    def __init__(self, params, base_optimizer, num_params, **kwargs):
        if not lr_sigma > 0.0:
            raise ValueError(f"Invalid lr_sigma, should be non-negative: {lr_sigma}")
        if not sigma_prior > 0.0:
            raise ValueError(f"Invalid sigma_prior, should be positive: {sigma_prior}")
        if not kl_div_weight > 0.0:
            raise ValueError(f"Invalid kl_div_weight, should be positive: {kl_div_weight}")
        super(MFVI, self).__init__(params, base_optimizer, num_params, **kwargs)

    def _get_perturbation(self):
        """Calculates a standard normal perturbation for each parameter."""
        perturbation_groups = []
        for param_group in self.param_groups:
            perturbation_group = {'params': []}
            for param in param_group['params']:
                if param.requires_grad:
                  perturbation_group['params'].append(torch.randn_like(param))
                else:
                  perturbation_group['params'].append(None)
            perturbation_groups.append(perturbation_group)
        return perturbation_groups
    
    @torch.no_grad()
    def _populate_gradients_for_Sigma(self):
        """This function computes the gradients with respect to the field parameters
        in `M_param_group`. It does so by multiplying the corresponding mean parameter
        gradients by the perturbations, and adding this value to the gradient of the KL
        divergence."""

        with torch.enable_grad():
            kl_div = torch.tensor(0.0, device=self.shared_device)
            for M_param_group in self.M_param_groups:
                for M in M_param_group['params']:
                    if M.requires_grad:
                        kl_div+= (M**2).sum()/2/self.sigma_prior**2 + torch.log(torch.abs(M)).sum()
            (self.kl_div_weight * kl_div).backward()

        for param_group, M_param_group, perturbation_group in zip(self.param_groups, self.M_param_groups, self.perturbation_groups):
            for param, M, perturbation in zip(param_group['params'], M_param_group['params'], perturbation_group['params']):
                if param.requires_grad:
                    M.grad.add_(param.grad * perturbation * torch.sign(M))


class RandomSAM(MeanFieldOptimizer):
    """Implements Mean Field Variational Optimization.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        base_optimizer (torch.optim.Optimizer): base optimizer to make gradient
            updates to mean and field parameters
        init_scale_M (float): controls the size of the variance of the Normal perturbation
        """
    def __init__(self, params, base_optimizer, lr_sigma=0.0, **kwargs):
        if lr_sigma != 0.0:
            raise ValueError('RandomSAM should not modify Sigma, lr_sigma should be 0.')
        super(RandomSAM, self).__init__(params, base_optimizer, lr_sigma=0.0, **kwargs)

    def _get_perturbation(self):
        """Calculates a standard normal perturbation for each parameter."""
        perturbation_groups = []
        for param_group in self.param_groups:
            perturbation_group = {'params': []}
            for param in param_group['params']:
                if param.requires_grad:
                  perturbation_group['params'].append(torch.randn_like(param))
                else:
                  perturbation_group['params'].append(None)
            perturbation_groups.append(perturbation_group)
        return perturbation_groups
      
    @torch.no_grad()
    def _populate_gradients_for_Sigma(self):
        pass


class MixSAM(MeanFieldOptimizer):
      def __init__(self, params, base_optimizer, kappa_scale=1.0, lr_sigma=0.0, **kwargs):
        if lr_sigma != 0.0:
            raise ValueError('MixSAM should not modify Sigma, lr_sigma should be 0.')
        self.kappa_scale = kappa_scale
        super(MixSAM, self).__init__(params, base_optimizer, lr_sigma=0.0, **kwargs)

      def _get_perturbation(self):
        perturbation_groups = []
        squared_norm = torch.tensor(0.0, device=self.shared_device)
        
        for param_group in self.param_groups:
            perturbation_group = {'params':[]}
            for param in param_group['params']:
                if param.requires_grad:
                    if param.grad is None:
                        raise ValueError('MixSAM requires gradients to be populated to take a step.')
                    perturbation = param.grad.detach().clone()
                    squared_norm.add_((perturbation**2).sum())
                    perturbation_group['params'].append(perturbation)
                else:
                    perturbation_group['params'].append(None)

            perturbation_groups.append(perturbation_group)

        scale = torch.sqrt(self.num_params) / (torch.sqrt(squared_norm) + self.eps)

        squared_norm = torch.tensor(0.0, device=self.shared_device)

        for perturbation_group in perturbation_groups:
            for perturbation in perturbation_group['params']:
                if perturbation is not None:
                    perturbation.mul_(scale)
                    perturbation.add_(self.kappa_scale * torch.randn_like(perturbation))
                    squared_norm.add_((perturbation**2).sum())
        
        scale = torch.sqrt(self.num_params) / (torch.sqrt(squared_norm) + self.eps)

        for perturbation_group in perturbation_groups:
            for perturbation in perturbation_group['params']:
                if perturbation is not None:
                    perturbation.mul_(scale)

        return perturbation_groups

      @torch.no_grad()
      def _populate_gradients_for_Sigma(self):
          pass


class VSAM(torch.optim.Optimizer):
    """
    An implementation of Variational SAM.
    """
    def __init__(self, params, base_optimizer, lr_M=0.01, rho=0.05, data_len=50000, sigma_prior=np.sqrt(2)/5, num_params=1e6, **kwargs):
        if not rho >= 0.0:
            raise ValueError(f"Invalid rho, should be non-negative: {rho}")
        if not lr_M >= 0.0:
            raise ValueError(f"Invalid eta2, should be non-negative: {lr_M}")
        self.rho = rho
        self.sigma_prior = sigma_prior
        self.data_len = data_len
        self.num_params = num_params
        defaults = dict(lr_M=lr_M, **kwargs)
        super(VSAM, self).__init__(params, defaults)
        self.M_param_groups = []

        for param_group in self.param_groups:
            M_param_group = param_group.copy()
            M_param_group["params"] = [
                torch.ones_like(tensor, requires_grad=tensor.requires_grad)
                for tensor in param_group['params']
            ]
            for M in M_param_group["params"]:
                constant_(M, np.sqrt(self.num_params)/self.rho)

            M_param_group['lr'] = M_param_group['lr_M']
            M_param_group.pop('lr_M')
            param_group.pop('lr_M')
            self.M_param_groups.append(M_param_group)

        self.base_optimizer = base_optimizer(
            self.param_groups + self.M_param_groups, **kwargs)

        self.eps = max(torch.finfo(
            self.param_groups[0]['params'][0].dtype).eps, 1e-12)

        self.shared_device = self.param_groups[0]["params"][0].device

    def mloss(self):
        squared_norm = self._grad_norm()
        return self.rho * torch.sqrt(squared_norm)

    def mpenalty(self):
        trace_Minv = torch.tensor(0.0, device=self.shared_device)
        logdet_M = torch.tensor(0.0, device=self.shared_device)
        for param_group, M_param_group in zip(self.param_groups, self.M_param_groups):
            for param, M in zip(param_group['params'], M_param_group['params']):
                if param.grad is None:
                    continue
                trace_Minv.add_((1/(M**2)).sum())
                logdet_M.add_(torch.log(M**2).sum())
        return 1/ (2*self.data_len) * (trace_Minv / self.sigma_prior + logdet_M)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        squared_norm = self._grad_norm()
        scale = self.rho / (torch.sqrt(squared_norm) + self.eps)
        for param_group, M_param_group in zip(self.param_groups, self.M_param_groups):
            for param, M in zip(param_group['params'], M_param_group['params']):
                if param.grad is None:
                    continue
                self.state[param]["old_p"] = param.data.clone()
                param.add_(scale * param.grad / (M**2))
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self):
        for p in self.param_groups[0]["params"]:
            if p.grad is None:
                continue
            p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"
        self.base_optimizer.step()

    @torch.no_grad()
    def step(self, closure):
        self._zero_M_grad()
        with torch.enable_grad():
            penalized_mloss = self.mloss() + self.mpenalty()
            penalized_mloss.backward()
        self.first_step(zero_grad=True)
        with torch.enable_grad():
            closure()
        self.second_step()

    def _grad_norm(self):
        squared_norm = torch.tensor(0.0, device=self.shared_device)
        for param_group, M_param_group in zip(self.param_groups, self.M_param_groups):
            for param, M in zip(param_group['params'], M_param_group['params']):
                if param.grad is None:
                    continue
                grad_tensor = param.grad.detach().to(self.shared_device)
                squared_norm.add_((grad_tensor**2/(M**2)).sum())
        return squared_norm

    def _zero_M_grad(self, set_to_none=False):
        for M_param_group in self.M_param_groups:
            for M in M_param_group['params']:
                if set_to_none:
                    M.grad = None
                else:
                    if M.grad is not None:
                        torch.zero_(M.grad)