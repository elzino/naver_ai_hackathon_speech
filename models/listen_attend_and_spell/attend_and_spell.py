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

import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import Attention
from .beam_search import Beam

class AttendSpellRNN(nn.Module):
    r"""
    Provides functionality for decoding in a seq2seq framework, with an option for attention.
    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        sos_id (int): index of the start of sentence symbol
        eos_id (int): index of the end of sentence symbol
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
    Attributes:
        KEY_ATTN_SCORE (str): key used to indicate attention weights in `ret_dict`
        KEY_LENGTH (str): key used to indicate a list representing lengths of output sequences in `ret_dict`
        KEY_SEQUENCE (str): key used to indicate a list of sequences in `ret_dict`
    Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        - **inputs** (batch, seq_len, input_size): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default `None`)
        - **encoder_hidden** (num_layers * num_directions, batch_size, hidden_size): tensor containing the features in the
          hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
        - **encoder_outputs** (batch, seq_len, hidden_size): tensor with containing the outputs of the encoder.
          Used for attention mechanism (default is `None`).
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
          drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0).
    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (batch, seq_len, vocab_size): list of tensors with size (batch_size, vocab_size) containing
          the outputs of the decoding function.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs }.
    """

    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'
    BEAM_INDEX = 'beam_index'
    PROBABILITY = 'probability'

    def __init__(self, vocab_size, max_len, hidden_size, sos_id, eos_id, n_layers=2, rnn_cell='gru',
                 embedding_size=512, input_dropout_p=0, dropout_p=0, beam_width=1, device='cpu'):
        super().__init__()

        self.device = device

        self.hidden_size = hidden_size
        self.max_length = max_len

        self.eos_id = eos_id
        self.sos_id = sos_id
        self.init_input = None

        self.n_layers = n_layers
        self.beam_width = beam_width

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        assert n_layers > 1

        dropout_p = 0 if n_layers == 2 else dropout_p
        self.bottom_rnn = self.rnn_cell(hidden_size + embedding_size, hidden_size, batch_first=True)
        self.upper_rnn = self.rnn_cell(hidden_size, hidden_size, n_layers-1, batch_first=True, dropout=dropout_p)

        # TODO word embedding dimension parameter 추가하고 바꾸기
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.input_dropout = nn.Dropout(p=input_dropout_p)
        self.attention = Attention(self.hidden_size)
        self.out = nn.Linear(self.hidden_size, vocab_size)

    def forward_step(self, input_var, last_bottom_hidden, last_upper_hidden, encoder_outputs, function):
        # input_var = [list of int] = [B]
        # last_~~~_hidden = [layer x B x hidden_size]
        # encoder_outputs = [B x max_len x hidden_dim]

        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded).unsqueeze(1)  # B x 1 x H

        #  if self.training:
        self.bottom_rnn.flatten_parameters()
        self.upper_rnn.flatten_parameters()

        attn = self.attention(encoder_outputs, last_bottom_hidden)  # (batch, max_len)
        context = attn.unsqueeze(1).bmm(encoder_outputs)  # B x 1 x H
        x = torch.cat([embedded, context], 2)  # B x 1 x (2 * H)

        x, bottom_hidden = self.bottom_rnn(x, last_bottom_hidden)
        x, upper_hidden = self.upper_rnn(x, last_upper_hidden)  # B x 1 x H
        predicted_prob = function(self.out(x.squeeze(1)), dim=-1)  # B x vocab_size

        return predicted_prob, bottom_hidden, upper_hidden, attn

    def forward_step_beam(self, input_var, last_bottom_hidden, last_upper_hidden, encoder_outputs, function):
        # input_var = [list of int] = [B]
        # last_~~~_hidden = [layer x B x hidden_size]
        # encoder_outputs = [B x max_len x hidden_dim]

        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded).unsqueeze(1)  # B x 1 x H

        batch_size, encoder_length, encoder_hidden_dim = encoder_outputs.size()
        encoder_outputs = encoder_outputs.repeat(self.beam_width, 1, 1)
        encoder_outputs = encoder_outputs.view(self.beam_width, batch_size, encoder_length, encoder_hidden_dim)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        encoder_outputs = encoder_outputs.reshape(self.beam_width * batch_size, encoder_length, encoder_hidden_dim)

        #  if self.training:
        self.bottom_rnn.flatten_parameters()
        self.upper_rnn.flatten_parameters()

        attn = self.attention(encoder_outputs, last_bottom_hidden)  # (batch, max_len)
        context = attn.unsqueeze(1).bmm(encoder_outputs)  # B x 1 x H
        x = torch.cat([embedded, context], 2)  # B x 1 x (2 * H)

        x, bottom_hidden = self.bottom_rnn(x, last_bottom_hidden)
        x, upper_hidden = self.upper_rnn(x, last_upper_hidden)  # B x 1 x H
        predicted_prob = function(self.out(x.squeeze(1)), dim=-1)  # B x vocab_size

        return predicted_prob, bottom_hidden, upper_hidden, attn

    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None, function=F.log_softmax, teacher_forcing_ratio=0):
        ret_dict = dict()
        ret_dict[AttendSpellRNN.KEY_ATTN_SCORE] = list()
        ret_dict[AttendSpellRNN.BEAM_INDEX] = list()
        ret_dict[AttendSpellRNN.KEY_SEQUENCE] = list()
        ret_dict[AttendSpellRNN.PROBABILITY] = list()

        inputs, batch_size, max_length = self._validate_args(inputs, encoder_outputs, teacher_forcing_ratio)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        def decode(step, step_output, step_attn):
            decoder_outputs.append(step_output)
            ret_dict[AttendSpellRNN.KEY_ATTN_SCORE].append(step_attn)
            # TODO : BEAM Search 추가하기
            symbols = step_output.topk(1)[1]  # topk(n) [0]는 값 [1]은 index
            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)  # eos 처음 나타나는 곳에서 그 길이로 update
            return symbols

        if self.training:
            bottom_hidden, upper_hidden = self._init_state_zero(batch_size)

            if use_teacher_forcing:
                decoder_input = inputs[:, :-1]
                for di in range(max_length):
                    decoder_output, bottom_hidden, upper_hidden, attn \
                        = self.forward_step(decoder_input[:, di], bottom_hidden, upper_hidden, encoder_outputs,
                                            function)
                    decode(di, decoder_output, attn)
            else:
                decoder_input = inputs[:, 0]
                for di in range(max_length):
                    decoder_output, bottom_hidden, upper_hidden, step_attn \
                        = self.forward_step(decoder_input, bottom_hidden, upper_hidden, encoder_outputs, function)
                    symbols = decode(di, decoder_output, step_attn)  # batch x 1
                    decoder_input = symbols.squeeze(1)

            ret_dict[AttendSpellRNN.KEY_SEQUENCE] = sequence_symbols
            ret_dict[AttendSpellRNN.KEY_LENGTH] = lengths.tolist()
            decoder_outputs_temp = torch.stack(decoder_outputs, dim=1)  # batch x seq_len x vocab_size
            hyps = decoder_outputs_temp.max(-1)[1]

        else:
            bottom_hidden, upper_hidden = self._init_state_zero_beam(batch_size, self.beam_width)
            beam = [
                Beam(self.beam_width, self.sos_id, self.eos_id, cuda=True)
                for _ in range(batch_size)
            ]

            for di in range(max_length):
                # (a) Construct batch x beam_size nxt words.
                # Get all the pending current beam words and arrange for forward.

                decoder_input = torch.stack([b.current_predictions for b in beam]).to(self.device)
                decoder_input = decoder_input.view(-1)
                decoder_output, bottom_hidden, upper_hidden, step_attn \
                    = self.forward_step_beam(decoder_input, bottom_hidden, upper_hidden, encoder_outputs, function)
                decoder_output = decoder_output.view(batch_size, self.beam_width, -1)
                step_attn = step_attn.view(batch_size, self.beam_width, -1)

                select_indices_array = []
                # Loop over the batch_size number of beam
                for j, b in enumerate(beam):
                    b.advance(decoder_output[j, :], step_attn.data[j, :, :])
                    select_indices_array.append(
                        list(map(lambda x: x + j * self.beam_width, b.current_origin))
                    )
                select_indices = torch.tensor(select_indices_array, dtype=torch.int64).view(-1).to(self.device)
                bottom_hidden, upper_hidden = self._select_indices_hidden(select_indices, bottom_hidden, upper_hidden)

            for b in beam:
                _, ks = b.sort_finished()
                times, k = ks[0]
                hyp, beam_index, prob = b.get_hyp(times, k)

                prob = torch.stack(prob)
                prob = b.fill_empty_sequence(prob, max_length)  # make length max_length of sequence
                hyp = torch.stack(hyp)
                hyp = b.fill_empty_sequence(hyp, max_length)

                ret_dict[AttendSpellRNN.PROBABILITY].append(prob)
                ret_dict[AttendSpellRNN.KEY_SEQUENCE].append(hyp)

            hyps = torch.stack(ret_dict[AttendSpellRNN.KEY_SEQUENCE])
            probs = torch.stack(ret_dict[AttendSpellRNN.PROBABILITY])
            probs = torch.transpose(probs, 0, 1)
            for i in range(probs.size(0)):
                decoder_outputs.append(probs[i])

        # decoder_outputs = [seq_len, batch, vocab_size]
        return decoder_outputs, hyps, bottom_hidden, upper_hidden

    def _select_indices_hidden(self, select_indices, bottom_hidden, upper_hidden):
        return torch.index_select(bottom_hidden, 1, select_indices), torch.index_select(upper_hidden, 1, select_indices)

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        bottom_hidden = encoder_hidden[-self.n_layers, :, :].unsqueeze(0)
        upper_hidden = encoder_hidden[(-self.n_layers + 1):, :, :]
        return bottom_hidden, upper_hidden

    def _init_state_beam(self, encoder_hidden):
        """
            Initialize the encoder hidden state.
            encoder_hidden : [layer x B x hidden_size]
        """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)

        _, batch_size, hidden_size = encoder_hidden.size()
        bottom_hidden = encoder_hidden[-self.n_layers, :, :].unsqueeze(0)  # make beam * batch to batch * beam
        bottom_hidden = bottom_hidden.repeat(1, self.beam_width,1)
        bottom_hidden = bottom_hidden.view(1, self.beam_width, batch_size, hidden_size)
        bottom_hidden = torch.transpose(bottom_hidden, 1, 2).reshape(1, self.beam_width * batch_size, hidden_size)

        upper_hidden = encoder_hidden[(-self.n_layers + 1):, :, :]
        upper_hidden = upper_hidden.repeat(1, self.beam_width, 1)
        upper_hidden = upper_hidden.view(self.n_layers - 1, self.beam_width, batch_size, hidden_size)
        upper_hidden = torch.transpose(upper_hidden, 1, 2).reshape(self.n_layers - 1, self.beam_width * batch_size, hidden_size)

        return bottom_hidden, upper_hidden

    def _init_state_zero(self, batch_size):
        bottom_init = torch.zeros(1, batch_size, self.hidden_size).to(self.device)
        upper_init = torch.zeros(self.n_layers - 1, batch_size, self.hidden_size).to(self.device)
        return bottom_init, upper_init

    def _init_state_zero_beam(self, batch_size, beam_width):
        bottom_init = torch.zeros(1, batch_size*beam_width, self.hidden_size).to(self.device)
        upper_init = torch.zeros(self.n_layers - 1, batch_size*beam_width, self.hidden_size).to(self.device)
        return bottom_init, upper_init
    def _cat_directions(self, h):
        """
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def _validate_args(self, inputs, encoder_outputs, teacher_forcing_ratio):
        if encoder_outputs is None:
            raise ValueError("Argument encoder_outputs cannot be None.")

        batch_size = encoder_outputs.size(0)

        # set default input and max decoding length
        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.max_length
        else:
            max_length = inputs.size(1) - 1  # minus the start of sequence symbol

        return inputs, batch_size, max_length
