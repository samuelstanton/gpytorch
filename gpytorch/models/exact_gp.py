from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import warnings
import torch
from torch.autograd import Variable
from ..module import Module
from ..functions import exact_predictive_mean, exact_predictive_covar
from ..random_variables import GaussianRandomVariable
from ..likelihoods import GaussianLikelihood


class ExactGP(Module):

    def __init__(self, train_inputs, train_targets, likelihood):
        if torch.is_tensor(train_inputs):
            train_inputs = train_inputs,
        if not all(torch.is_tensor(train_input) for train_input in train_inputs):
            raise RuntimeError(
                'Train inputs must be a tensor, or a list/tuple of tensors'
            )
        if not isinstance(likelihood, GaussianLikelihood):
            raise RuntimeError('ExactGP can only handle GaussianLikelihood')

        super(ExactGP, self).__init__()
        self.train_inputs = train_inputs
        self.train_targets = train_targets
        self.likelihood = likelihood

        self.mean_cache = None
        self.covar_cache = None

    def _apply(self, fn):
        self.train_inputs = tuple(fn(train_input) for train_input in self.train_inputs)
        self.train_targets = fn(self.train_targets)
        return super(ExactGP, self)._apply(fn)

    def marginal_log_likelihood(self, likelihood, output, target, n_data=None):
        from ..mlls import ExactMarginalLogLikelihood
        if not hasattr(self, '_has_warned') or not self._has_warned:
            import warnings
            warnings.warn(
                'model.marginal_log_likelihood is now deprecated. '
                'Please use gpytorch.mll.ExactMarginalLogLikelihood instead.'
            )
            self._has_warned = True
        return ExactMarginalLogLikelihood(likelihood, self)(output, target)

    def set_train_data(self, inputs=None, targets=None):
        """Set training data (does not re-fit model hyper-parameters)"""
        if inputs is not None:
            if torch.is_tensor(inputs):
                inputs = inputs,
            for input, t_input in zip(inputs, self.train_inputs):
                for attr in {'shape', 'dtype', 'device'}:
                    if getattr(input, attr) != getattr(t_input, attr):
                        raise RuntimeError(
                            'Cannot modify {attr} of inputs'.format(attr=attr)
                        )
            self.train_inputs = inputs
        if targets is not None:
            for attr in {'shape', 'dtype', 'device'}:
                if getattr(targets, attr) != getattr(self.train_targets, attr):
                    raise RuntimeError(
                        'Cannot modify {attr} of targets'.format(attr=attr)
                    )
            self.train_targets = targets
        self.mean_cache = None
        self.covar_cache = None

    def train(self, mode=True):
        if mode:
            self.mean_cache = None
            self.covar_cache = None
        return super(ExactGP, self).train(mode)

    def __call__(self, *args, **kwargs):
        train_inputs = tuple(Variable(train_input) for train_input in self.train_inputs)

        # Training mode: optimizing
        if self.training:
            if not all(torch.equal(train_input, input) for train_input, input in zip(train_inputs, args)):
                raise RuntimeError('You must train on the training inputs!')
            return super(ExactGP, self).__call__(*args, **kwargs)

        # Posterior mode
        else:
            if all(torch.equal(train_input, input) for train_input, input in zip(train_inputs, args)):
                warnings.warn('The input matches the stored training data. '
                              'Did you forget to call model.train()?', UserWarning)

            # Exact inference
            full_inputs = tuple(torch.cat([train_input, input]) for train_input, input in zip(train_inputs, args))
            full_output = super(ExactGP, self).__call__(*full_inputs, **kwargs)
            if not isinstance(full_output, GaussianRandomVariable):
                raise RuntimeError('ExactGP.forward must return a GaussianRandomVariable')
            full_mean, full_covar = full_output.representation()
	    if len(full_mean.size()) == 2:
		for i in range(full_mean.size(0)):
	            noise = self.likelihood.log_noise.exp()
		    if self.mean_cache is None:
			cache_arg = None
		    else:
			cache_arg = self.mean_cache[i, :]
            	    temp_mean, temp_mean_cache = exact_predictive_mean(full_covar[i, :, :], full_mean[i, :], Variable(self.train_targets[i, :]), noise, cache_arg)
		    if i == 0:
			predictive_mean = temp_mean.unsqueeze(0)
			mean_cache = temp_mean_cache.unsqueeze(0)
		    else:	
			predictive_mean = torch.cat((predictive_mean, temp_mean.unsqueeze(0)))
			mean_cache = torch.cat((mean_cache, temp_mean_cache.unsqueeze(0)))
	    else:
		noise = self.likelihood.log_noise.exp()
                predictive_mean, mean_cache = exact_predictive_mean(full_covar, full_mean, Variable(self.train_targets), noise, self.mean_cache)
            predictive_covar, covar_cache = exact_predictive_covar(full_covar, self.train_targets.size(-1),
                                                                            noise, self.covar_cache)

            self.mean_cache = mean_cache
            self.covar_cache = covar_cache
	    return GaussianRandomVariable(predictive_mean, predictive_covar)
