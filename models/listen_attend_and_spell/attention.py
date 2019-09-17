import torch
import torch.nn as nn
import torch.nn.functional as F


# Reference
# concept : https://hcnoh.github.io/2018-12-11-bahdanau-attention
# code : https://towardsdatascience.com/attention-seq2seq-with-pytorch-learning-to-invert-a-sequence-34faf4133e53


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        """
        Args
            hidden_dim: dimension of hidden unit in LSTM = numdirections * hidden_size

        Inputs
            encoder_outputs: (seq_len, batch, num_directions * hidden_size)
            last_hidden: last_hidden (batch, decoder_hidden_dim)

        Outputs
            context: (seq_len, batch, decoder_hidden_dim)

        """
        super().__init__()
        self.Wa = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Ua = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Va = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, encoder_outputs, last_hiddens, seq_lens=None):
        """
                Inputs
                    encoder_outputs: (batch, max_len, hidden_dim)
                    last_hidden: last_hidden (1 x batch x hidden_dim)
                    seq_lens: lengths of sequence (list of int)

                Outputs
                    context: (batch, hidden_dim)

        """

        last_hidden = last_hiddens.transpose(0, 1)  # (batch, 1, hidden_dim)
        attention_energy = self.Va(torch.tanh(self.Wa(encoder_outputs) + self.Ua(last_hidden)))  # (batch, max_len, 1)
        attention_energy = attention_energy.squeeze(-1)  # (batch, max_len)
        #  attention_energy = self.mask_3d(attention_energy, seq_lens, -float('inf'))
        #  -> input과 여기서 seq_len이 1대1 대응이 안되서 사용 불가능
        ai = F.softmax(attention_energy, -1)  # (batch, max_len)
        return ai

    def mask_3d(self, attention_energy, seq_lens, value):
        for idx, length in enumerate(seq_lens):
            attention_energy[idx, length:] = value
        return attention_energy

