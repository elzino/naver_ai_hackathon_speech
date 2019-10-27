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
    def __init__(self, models, vocab_size, max_len, device='cpu'):
        super(Ensemble, self).__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.models= models
        for model in self.models:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
        self.device = device
        self.classifier = nn.Linear(vocab_size * len(models),vocab_size)
    def flatten_parameters(self):
       return 0

    def forward(self, input_variable, input_lengths=None, target_variable=None, function=F.log_softmax, teacher_forcing_ratio=0):
        results = []
        for model in self.models:
            output = model(input_variable, input_lengths, target_variable, teacher_forcing_ratio=teacher_forcing_ratio) # seq_len x (batch x vocab Tensor)
            output = torch.stack(output, dim=1).to(self.device)  # batch x seq_len x vocab_size
            results.append(output)
        x = self.classifier(torch.cat(results, dim=2))
        x = function(x, dim=-1)  # batch x length x vocab
        return x
