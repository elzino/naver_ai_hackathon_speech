import torch


class Beam(object):
    """Class for managing the internals of the beam search process.
    Takes care of beams, back pointers, and scores.
    Args:
        size (int): Number of beams to use.
        pad (int): Magic integer in output vocab.
        bos (int): Magic integer in output vocab.
        eos (int): Magic integer in output vocab.
        n_best (int): Don't stop until at least this many beams have
            reached EOS.
        cuda (bool): use gpu
        global_scorer (onmt.translate.GNMTGlobalScorer): Scorer instance.
        min_length (int): Shortest acceptable generation, not counting
            begin-of-sentence or end-of-sentence.
    """

    def __init__(self, size, bos, eos,
                 n_best=1, cuda=False,
                 min_length=0,
                 stepwise_penalty=False,
                 block_ngram_repeat=0,
                 exclusion_tokens=set()):

        self.size = size
        self.tt = torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.all_scores = []
        self.all_probs = []        # all_probs = sequence * beam * vocab

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [self.tt.LongTensor(size).fill_(bos)]

        # Has EOS topped the beam yet.
        self._eos = eos

        # Time and k pair for finished.
        self.finished = []

        # Minimum prediction length
        self.min_length = min_length

    @property
    def current_predictions(self):
        return self.next_ys[-1]

    @property
    def current_origin(self):
        """Get the backpointers for the current timestep."""
        return self.prev_ks[-1]

    def advance(self, word_probs, attn_out):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attn_out`: Compute and update the beam search.
        Args:
            word_probs (FloatTensor): probs of advancing from the last step
                ``(K, words)``              beam_width * words
            attn_out (FloatTensor): attention at the last step
        Returns:
            bool: True if beam search is complete.
        """
        word_probs_temp = word_probs.clone()
        num_words = word_probs.size(1)
        # force the output to be longer than self.min_length
        cur_len = len(self.next_ys)
        if cur_len <= self.min_length:
            # assumes there are len(word_probs) predictions OTHER
            # than EOS that are greater than -1e20
            for k in range(len(word_probs)):
                word_probs[k][self._eos] = -1e20

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_scores = word_probs + self.scores.unsqueeze(1)  # beam_width * words
            # Don't let EOS have children.
            for i in range(self.size):
                if self.next_ys[-1][i] == self._eos:
                    beam_scores[i] = -1e20
        else:
            beam_scores = word_probs[0]
        flat_beam_scores = beam_scores.view(-1)
        best_scores, best_scores_id = flat_beam_scores.topk(self.size, 0, True, True)
        self.all_probs.append(word_probs_temp)   
        self.all_scores.append(self.scores)
        self.scores = best_scores

        # best_scores_id is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = best_scores_id / num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append((best_scores_id - prev_k * num_words))

        for i in range(self.size):
            if self.next_ys[-1][i] == self._eos:
                length = len(self.next_ys) - 1
                score = self.scores[i] / length
                self.finished.append((score, length, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.next_ys[-1][0] == self._eos:
            self.all_scores.append(self.scores)

    def sort_finished(self):
        if len(self.finished) == 0:
            for i in range(self.n_best):
                length = len(self.next_ys) - 1
                score = self.scores[i] / length
                self.finished.append((score, length, i))

        self.finished = sorted(self.finished, key=lambda finished: finished[0], reverse=True)
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

    def get_hyp(self, timestep, k):
        """Walk back to construct the full hypothesis."""
        hyp, key_index, probability = [], [], []
        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            key_index.append(k)
            probability.append(self.all_probs[j][k])
            k = self.prev_ks[j][k]

        return hyp[::-1], key_index[::-1], probability[::-1]

    def fill_empty_sequence(self, stack, max_length):
        for i in range(stack.size(0), max_length):
            stack = torch.cat([stack, stack[0].unsqueeze(0)])
        return stack

