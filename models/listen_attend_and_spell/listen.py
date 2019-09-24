
import torch.nn as nn
import math


class ListenRNN(nn.Module):
    r"""
    Applies a multi-layer RNN to an input sequence.

    Args:
        feature_size (int): mel_filter num
        hidden_size (int): the number of features in the hidden state `h`
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)

    Inputs: inputs, input_lengths
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is a list of token IDs.
        - **input_lengths** (list of int, optional): list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)

    Outputs: output, hidden
        - **output** (batch, seq_len, hidden_size): tensor containing the encoded features of the input sequence
        - **hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the features in the hidden state `h`

    Examples::

         >>> encoder = EncoderRNN(input_vocab, max_seq_length, hidden_size)
         >>> output, hidden = encoder(input)

    """

    SYM_MASK = "MASK"
    SYM_EOS = "EOS"

    def __init__(self, feature_size, hidden_size=256, input_dropout_p=0, dropout_p=0, n_layers=3,
                 rnn_cell='gru'):
        super().__init__()

        self.input_dropout_p = input_dropout_p
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        self.dropout_p = dropout_p

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        """
        Copied from https://github.com/SeanNaren/deepspeech.pytorch/blob/master/model.py
        Copyright (c) 2017 Sean Naren
        MIT License
        """
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20)
        )

        feature_size = math.ceil((feature_size - 11 + 1 + (5 * 2)) / 2)
        feature_size = math.ceil(feature_size - 11 + 1 + (5 * 2))
        feature_size *= 32

        self.rnn = self.rnn_cell(feature_size, hidden_size, n_layers, batch_first=True, bidirectional=True, dropout=dropout_p)

    def forward(self, input_var, input_lengths=None):
        """
        Applies a multi-layer RNN to an input sequence.

        Args:
            input_var (batch, mel_filter, seq_len): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch

        Returns: output, hidden
            - **output** (seq_len, batch, num_direction * hidden_size): variable containing the encoded features of the input sequence
            - **hidden** hn: variable containing the features in the hidden state hn
                    hn : (num_layers * num_directions, batch, hidden_size)
        """

        # Bx1xTxD
        input_var = input_var.unsqueeze(1)
        x = self.conv(input_var)

        # BxCxTxD - > BxTxCxD
        x = x.transpose(1, 2).contiguous()
        sizes = x.size()
        # BxTxCxD => BxTx(CxD)
        x = x.view(sizes[0], sizes[1], sizes[2] * sizes[3])

        #  if self.training:
        self.rnn.flatten_parameters()

        # x = BxTxH
        output, hidden = self.rnn(x)

        # output = BxTxH
        return output, hidden
