from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch import nn
from .mean import Mean
import pdb

class IdentityMean(Mean):
    def __init__(self, batch_size=None):
        super(IdentityMean, self).__init__()
	self.batch_size = batch_size
    def forward(self, input):
	if self.batch_size is None:
		return input
	else:
		return input.view(self.batch_size, input.size(1)*input.size(2))
