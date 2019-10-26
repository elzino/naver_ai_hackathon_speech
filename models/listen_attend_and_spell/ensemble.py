"""

Copyright 2017- IBM Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""


import time
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch

class Ensemble(nn.Module):
    """ Ensemble structure to get output

        Input: 4 models to get output
    """
    def __init__(self,model, vocab_size, max_len, device='cpu'):
        super(Ensemble, self).__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.model= model
        self.device = device

    def forward(self, input_variable, input_lengths=None, target_variable=None, teacher_forcing_ratio=0):
        x = torch.zeros(input_variable.size(0), self.max_len, self.vocab_size).to(self.device)
        for model in self.model:
            output = model(input_variable)
            output = torch.stack(output, dim=1).to(self.device)
            x += output
            # print(index,':',x[index])
        
        return x
