"""Sparse interpolated trigram tables built only from training tokens."""
from pathlib import Path
import torch
from common import ROOT


VOCAB = 2048
DEFAULT_TABLE = ROOT / 'assets' / 'trigram.pt'


def pack_pair(left, right):
    return (int(left) << 11) | int(right)


def build_trigram_table(ids, alpha=1.0, min_trigram_count=2):
    """Witten-Bell-style interpolation over seen n-grams only."""
    ids = ids.to(dtype=torch.int64).view(-1)
    uni = torch.bincount(ids, minlength=VOCAB).to(torch.float32)
    previous, token = ids[:-1], ids[1:]
    packed_bi = (previous << 11) | token
    bi_unique, bi_count = torch.unique(packed_bi, return_counts=True)
    packed_tri = (previous[:-1] << 11) | token[:-1]
    packed_tri3 = (packed_tri << 11) | token[1:]
    tri_unique, tri_count = torch.unique(packed_tri3, return_counts=True)
    keep = tri_count >= min_trigram_count
    tri_unique = tri_unique[keep]
    tri_count = tri_count[keep]
    return {
        'vocab': VOCAB,
        'alpha': float(alpha),
        'min_trigram_count': int(min_trigram_count),
        'unigram': uni,
        'unigram_total': float(uni.sum().item()),
        'bigram_context': (bi_unique >> 11).to(torch.int32),
        'bigram_token': (bi_unique & 2047).to(torch.int16),
        'bigram_count': bi_count.to(torch.float32),
        'trigram_pair': (tri_unique >> 11).to(torch.int32),
        'trigram_token': (tri_unique & 2047).to(torch.int16),
        'trigram_count': tri_count.to(torch.float32),
    }


class SparseTrigram:
    def __init__(self, table, device='cpu'):
        if isinstance(table, (str, Path)):
            table = torch.load(table, map_location='cpu', weights_only=True)
        device = torch.device(device)
        self.alpha = float(table['alpha'])
        self.vocab = int(table['vocab'])
        self.device = device
        unigram = table['unigram'].to(dtype=torch.float32)
        self.unigram = (unigram / unigram.sum().clamp_min(1.0)).to(device)
        self.bigram_count = torch.zeros(self.vocab, self.vocab, dtype=torch.float32)
        ctx = table['bigram_context'].to(torch.long)
        tok = table['bigram_token'].to(torch.long)
        cnt = table['bigram_count'].to(torch.float32)
        self.bigram_count[ctx, tok] = cnt
        self.bigram_total = self.bigram_count.sum(-1)
        pair = table['trigram_pair'] if 'trigram_pair' in table else table['trigram_context']
        self.tri_pair = pair.to(dtype=torch.int64)
        self.tri_tok = table['trigram_token'].to(dtype=torch.int64)
        self.tri_cnt = table['trigram_count'].to(dtype=torch.float32)
        if self.tri_pair.numel():
            order = torch.argsort(self.tri_pair)
            self.tri_pair = self.tri_pair[order]
            self.tri_tok = self.tri_tok[order]
            self.tri_cnt = self.tri_cnt[order]
            self.tri_keys, inverse = torch.unique_consecutive(self.tri_pair, return_inverse=True)
            self.tri_tot = torch.zeros(self.tri_keys.shape, dtype=torch.float32)
            self.tri_tot.scatter_add_(0, inverse, self.tri_cnt)
        else:
            self.tri_keys = torch.zeros(0, dtype=torch.int64)
            self.tri_tot = torch.zeros(0, dtype=torch.float32)
        self.bigram_count = self.bigram_count.to(device)
        self.bigram_total = self.bigram_total.to(device)
        self.tri_pair = self.tri_pair.to(device)
        self.tri_tok = self.tri_tok.to(device)
        self.tri_cnt = self.tri_cnt.to(device)
        self.tri_keys = self.tri_keys.to(device)
        self.tri_tot = self.tri_tot.to(device)
        self.n_bigrams = int(cnt.numel())
        self.n_trigrams = int(self.tri_tok.numel())

    def to(self, device):
        device = torch.device(device)
        if device == self.device:
            return self
        self.device = device
        self.unigram = self.unigram.to(device)
        self.bigram_count = self.bigram_count.to(device)
        self.bigram_total = self.bigram_total.to(device)
        self.tri_pair = self.tri_pair.to(device)
        self.tri_tok = self.tri_tok.to(device)
        self.tri_cnt = self.tri_cnt.to(device)
        self.tri_keys = self.tri_keys.to(device)
        self.tri_tot = self.tri_tot.to(device)
        return self

    def byte_size(self):
        return (self.vocab * 8 + self.n_bigrams * 10 + self.n_trigrams * 10 + int(self.tri_keys.numel()) * 8)

    def probabilities(self, ids):
        """Interpolated p(x_{t+1} | prefix through t), shape [B, T, V]."""
        ids = ids.to(device=self.device, dtype=torch.int64)
        batch, length = ids.shape
        alpha = self.alpha
        uni = self.unigram
        counts = self.bigram_count[ids]
        total = self.bigram_total[ids].unsqueeze(-1)
        probs = (counts + alpha * uni) / (total + alpha)
        if length < 2 or self.tri_pair.numel() == 0:
            return probs
        packed = (ids[:, :-1] << 11) | ids[:, 1:]
        uniq, inverse = packed.reshape(-1).unique(return_inverse=True)
        right = uniq & 2047
        p_bi = (self.bigram_count[right] + alpha * uni) / (self.bigram_total[right].unsqueeze(-1) + alpha)
        counts_u = uniq.new_zeros((uniq.numel(), self.vocab), dtype=torch.float32)
        loc = torch.searchsorted(uniq, self.tri_pair)
        in_range = loc < uniq.numel()
        matched = torch.zeros_like(self.tri_pair, dtype=torch.bool)
        matched[in_range] = uniq[loc[in_range]] == self.tri_pair[in_range]
        if matched.any():
            counts_u[loc[matched], self.tri_tok[matched]] = self.tri_cnt[matched]
        key_loc = torch.searchsorted(uniq, self.tri_keys)
        key_ok = torch.zeros_like(self.tri_keys, dtype=torch.bool)
        key_in = key_loc < uniq.numel()
        key_ok[key_in] = uniq[key_loc[key_in]] == self.tri_keys[key_in]
        totals_u = uniq.new_zeros((uniq.numel(),), dtype=torch.float32)
        if key_ok.any():
            totals_u[key_loc[key_ok]] = self.tri_tot[key_ok]
        p_tri = (counts_u + alpha * p_bi) / (totals_u.unsqueeze(-1) + alpha)
        suffix = p_tri[inverse].view(batch, length - 1, self.vocab)
        return torch.cat([probs[:, :1], suffix], dim=1)
