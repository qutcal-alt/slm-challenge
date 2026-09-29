"""Improved GPT with RMSNorm + RoPE + MoE.

Changes from baseline:
1. RMSNorm: More stable normalization (Llama standard)
2. RoPE: Rotary Position Embedding (better positional encoding)
3. MoE FFN: Mixture of Experts for capacity without parameter explosion

Design: Smaller width but MoE adds expressiveness.
"""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from common import ROOT
from ngram import SparseTrigram


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RoPE(nn.Module):
    """Rotary Position Embedding."""
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        if dim % 2:
            raise ValueError('RoPE head dimension must be even.')
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        
    def forward(self, x, seq_len):
        """Apply RoPE to x [batch, heads, seq_len, head_dim]."""
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        cos = freqs.cos().repeat_interleave(2, dim=-1)[None, None, :, :]
        sin = freqs.sin().repeat_interleave(2, dim=-1)[None, None, :, :]

        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
        return x * cos + rotated * sin


class BigramHash(nn.Module):
    """Hashed bigram features added to unigram embeddings.

    A full V×V table is impossible under the 64 MiB cap. Collisions are
    accepted in exchange for a compact table of `buckets` rows. Token pairs
    are packed as prev * vocab + curr, multiplied by the 32-bit golden
    ratio, then reduced with the high bits of that product. Using high bits
    rather than `% buckets` keeps previous-token information even when the
    table size is a power of two. Position 0 has no predecessor and
    receives no bigram vector.
    """

    _VOCAB = 2048
    _GOLDEN_32 = 0x9E3779B1

    def __init__(self, width, buckets, hash_dim=None):
        super().__init__()
        self.buckets = int(buckets)
        hash_dim = width if hash_dim is None else int(hash_dim)
        self.table = nn.Embedding(self.buckets, hash_dim)
        self.proj = None if hash_dim == width else nn.Linear(hash_dim, width, bias=False)

    def forward(self, ids, token_emb):
        previous = torch.zeros_like(ids)
        previous[:, 1:] = ids[:, :-1]
        pair = previous.to(torch.int64) * self._VOCAB + ids.to(torch.int64)
        mixed = (pair * self._GOLDEN_32) & 0xFFFFFFFF
        bucket = (mixed * self.buckets) >> 32
        bigram = self.table(bucket)
        if self.proj is not None:
            bigram = self.proj(bigram)
        bigram[:, 0].zero_()
        return token_emb + bigram


class SwiGLU(nn.Module):
    """Gated FFN: down(silu(gate(x)) * up(x))."""

    def __init__(self, width, hidden):
        super().__init__()
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoEFFN(nn.Module):
    """Mixture of Experts FFN with load balancing."""
    def __init__(self, width, num_experts=4, expert_hidden=None, top_k=1, ffn='gelu'):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.ffn = ffn
        expert_hidden = expert_hidden or int(1.5 * width)
        
        self.router = nn.Linear(width, num_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLU(width, expert_hidden) if ffn == 'swiglu' else nn.Sequential(
                nn.Linear(width, expert_hidden, bias=False),
                nn.GELU(),
                nn.Linear(expert_hidden, width, bias=False)
            ) for _ in range(num_experts)
        ])
        
    def forward(self, x, return_aux=False):
        batch, seq_len, width = x.shape
        x_flat = x.view(-1, width)
        router_logits = self.router(x_flat)
        probs = F.softmax(router_logits, dim=-1)
        routing_weights, selected_experts = torch.topk(probs, self.top_k, dim=-1)
        # Keep the softmax mass so language-model loss trains the router.
        # Top-1 must not be renormalized to 1; top-k>1 still shares mass
        # across the selected experts.
        if self.top_k > 1:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(x_flat)
        for i in range(self.num_experts):
            mask = (selected_experts == i).any(dim=-1)
            if mask.any():
                expert_input = x_flat[mask]
                expert_output = self.experts[i](expert_input)
                weight = (routing_weights[mask] * (selected_experts[mask] == i)).sum(dim=-1)
                output[mask] += expert_output * weight.unsqueeze(-1)

        output = output.view(batch, seq_len, width)
        self.last_selected_fraction = F.one_hot(
            selected_experts, self.num_experts
        ).float().sum(dim=1).mean(dim=0).detach()
        self.last_mean_probability = probs.mean(dim=0).detach()
        self.last_selected_experts = selected_experts.detach()
        if return_aux:
            return output, self._load_balance_loss(probs, selected_experts)
        return output

    def _load_balance_loss(self, probs, selected_experts):
        expert_count = self.num_experts
        selected_fraction = F.one_hot(selected_experts, expert_count).float().sum(dim=1).mean(dim=0)
        mean_probability = probs.mean(dim=0)
        return expert_count * (selected_fraction * mean_probability).sum()


class Block(nn.Module):
    def __init__(self, width=256, heads=8, moe_config=None, xsa_projection=False,
                 smear_gate=False, smear_init=-2.0, dropout=0.0, ffn='gelu', ffn_hidden=None):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.xsa_projection = xsa_projection
        self.attn_dropout = nn.Dropout(dropout)
        self.mlp_dropout = nn.Dropout(dropout)
        self.smear_gate = nn.Parameter(torch.full((width,), smear_init)) if smear_gate else None
        
        self.norm1 = RMSNorm(width)
        self.norm2 = RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.rope = RoPE(self.head_dim)
        
        if moe_config:
            self.mlp = MoEFFN(
                width,
                num_experts=moe_config.get('num_experts', 4),
                expert_hidden=moe_config.get('expert_hidden'),
                top_k=moe_config.get('top_k', 1),
                ffn=moe_config.get('ffn', ffn)
            )
        elif ffn == 'swiglu':
            hidden = int(ffn_hidden) if ffn_hidden is not None else int(8 * width / 3)
            self.mlp = SwiGLU(width, hidden)
        else:
            hidden = int(ffn_hidden) if ffn_hidden is not None else 4 * width
            self.mlp = nn.Sequential(
                nn.Linear(width, hidden, bias=False),
                nn.GELU(),
                nn.Linear(hidden, width, bias=False)
            )

    def forward(self, x, return_aux=False):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q = self.rope(q, length)
        k = self.rope(k, length)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        if self.xsa_projection:
            value_direction = F.normalize(v, dim=-1, eps=1e-6)
            self_projection = (attended * value_direction).sum(dim=-1, keepdim=True) * value_direction
            attended = attended - self_projection
        x = x + self.attn_dropout(self.proj(attended.transpose(1, 2).reshape(batch, length, width)))

        mlp_input = self.norm2(x)
        if isinstance(self.mlp, MoEFFN):
            mlp_result = self.mlp(mlp_input, return_aux=return_aux)
            mlp_output, aux_loss = mlp_result if return_aux else (mlp_result, x.new_zeros(()))
        else:
            mlp_output = self.mlp(mlp_input)
            aux_loss = x.new_zeros(()) if return_aux else None
        output = x + self.mlp_dropout(mlp_output)
        if self.smear_gate is not None:
            previous = torch.zeros_like(output)
            previous[:, 1:] = output[:, :-1]
            gate = torch.sigmoid(self.smear_gate).view(1, 1, -1)
            output = (1.0 - gate) * output + gate * previous
        return (output, aux_loss) if return_aux else output



class ImprovedGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width = config['width']
        moe_config = config.get('moe')
        
        self.token = nn.Embedding(config['vocab'], width)
        bigram_buckets = int(config.get('bigram_hash_buckets', 0) or 0)
        self.bigram_hash = (
            BigramHash(width, bigram_buckets, config.get('bigram_hash_dim'))
            if bigram_buckets > 0 else None
        )
        # No positional embedding (RoPE handles it)
        self.blocks = nn.ModuleList([
            Block(width, config['heads'], moe_config, config.get('xsa_projection', False),
                  config.get('smear_gate', False), config.get('smear_init', -2.0),
                  config.get('dropout', 0.0), config.get('ffn', 'gelu'),
                  config.get('ffn_hidden'))
            for _ in range(config['depth'])
        ])
        self.norm = RMSNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        
        self.apply(self.initialize)
        if config.get('tie_embeddings', True):
            self.head.weight = self.token.weight
        self._scale_residual_projections()
        self.cache_mix = float(config.get('cache_mix', 0.0) or 0.0)
        self.cache_theta = float(config.get('cache_theta', 10.0) or 10.0)
        self.ngram_mix = float(config.get('ngram_mix', 0.0) or 0.0)
        ngram_path = config.get('ngram_path')
        self.ngram = None
        if self.ngram_mix > 0.0:
            path = Path(ngram_path) if ngram_path else ROOT / 'assets' / 'trigram.pt'
            if not path.is_absolute():
                path = ROOT / path
            self.ngram = SparseTrigram(path)

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def _scale_residual_projections(self):
        if not self.config.get('residual_scale', False):
            return
        std = .02 * (2 * self.config['depth']) ** -0.5
        for block in self.blocks:
            nn.init.normal_(block.proj.weight, std=std)
            if isinstance(block.mlp, MoEFFN):
                for expert in block.mlp.experts:
                    down = expert.down if isinstance(expert, SwiGLU) else expert[2]
                    nn.init.normal_(down.weight, std=std)
            elif isinstance(block.mlp, SwiGLU):
                nn.init.normal_(block.mlp.down.weight, std=std)
            else:
                nn.init.normal_(block.mlp[2].weight, std=std)

    def features(self, ids, return_aux=False):
        x = self.token(ids)
        if self.bigram_hash is not None:
            x = self.bigram_hash(ids, x)
        total_aux = x.new_zeros(())
        for block in self.blocks:
            if return_aux:
                x, aux_loss = block(x, return_aux=True)
                total_aux = total_aux + aux_loss
            else:
                x = block(x)
        x = self.norm(x)
        return (x, total_aux / len(self.blocks)) if return_aux else x

    def forward(self, ids):
        """Training and evaluation interface: unnormalized next-token logits."""
        return self.head(self.features(ids))

    def forward_with_aux(self, ids):
        features, aux_loss = self.features(ids, return_aux=True)
        return self.head(features), aux_loss

    def expert_usage(self):
        layers = self.expert_usage_layers()
        if not layers:
            return None
        return torch.stack(layers).mean(dim=0)

    def expert_usage_layers(self):
        usages = []
        for block in self.blocks:
            if isinstance(block.mlp, MoEFFN) and hasattr(block.mlp, 'last_selected_fraction'):
                usages.append(block.mlp.last_selected_fraction)
        return usages

    def expert_routing(self, ids=None, top_tokens=8):
        """Per-layer routing snapshot from the most recent forward.

        `expert_usage()` averages layers and can hide a collapsed layer.
        Optional `ids` of shape [batch, time] summarizes whether experts
        see similar tokens on that batch.
        """
        layers = []
        for index, block in enumerate(self.blocks):
            if not isinstance(block.mlp, MoEFFN):
                continue
            moe = block.mlp
            if not hasattr(moe, 'last_selected_fraction'):
                continue
            usage = moe.last_selected_fraction.detach().float()
            mean_prob = getattr(moe, 'last_mean_probability', usage)
            row = {
                'layer': index,
                'usage': [round(float(value), 4) for value in usage.cpu().tolist()],
                'mean_prob': [round(float(value), 4) for value in mean_prob.detach().cpu().tolist()],
                'usage_max': round(float(usage.max().item()), 4),
                'usage_min': round(float(usage.min().item()), 4),
                'usage_entropy': round(float((-(usage * (usage.clamp_min(1e-8).log())).sum()).item()), 4),
            }
            if ids is not None and hasattr(moe, 'last_selected_experts'):
                row['token_overlap'] = _expert_token_overlap(
                    ids, moe.last_selected_experts, moe.num_experts, top_tokens
                )
            layers.append(row)
        return layers


    def predict_log_probs(self, ids):
        """Evaluation interface: normalized log probabilities, with no access to targets."""
        hidden = self.features(ids)
        model_logp = F.log_softmax(self.head(hidden).float(), dim=-1)
        if self.cache_mix <= 0.0 and self.ngram_mix <= 0.0:
            return model_logp
        probs = model_logp.exp()
        if self.cache_mix > 0.0:
            cache = self._window_neural_cache(hidden, ids)
            mixed = (1.0 - self.cache_mix) * probs + self.cache_mix * cache
            probs = torch.where(cache.sum(-1, keepdim=True) > 0, mixed, probs)
        if self.ngram_mix > 0.0 and self.ngram is not None:
            ngram = self.ngram.to(ids.device).probabilities(ids)
            probs = (1.0 - self.ngram_mix) * probs + self.ngram_mix * ngram
        return torch.log(probs.clamp_min(1e-12))

    def _window_neural_cache(self, hidden, ids):
        """Pointer over earlier in-window states; each row/window is independent.

        Predicting the token after position t uses keys i<t with values ids[:, i+1].
        Position 0 has no history and returns zeros, so mixing leaves the model.
        """
        batch, length, width = hidden.shape
        cache = torch.zeros(batch, length, self.config['vocab'], dtype=torch.float32, device=hidden.device)
        if length < 2:
            return cache
        keys = F.normalize(hidden.float(), dim=-1)
        scores = self.cache_theta * torch.bmm(keys, keys.transpose(1, 2))
        index = torch.arange(length, device=hidden.device)
        valid = index.view(1, -1, 1) > index.view(1, 1, -1)
        scores = scores.masked_fill(~valid, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        weights = weights * valid
        stored = torch.zeros(batch, length, dtype=torch.long, device=ids.device)
        stored[:, :-1] = ids[:, 1:]
        cache.scatter_add_(
            2,
            stored.unsqueeze(1).expand(batch, length, length),
            weights,
        )
        return cache


def _expert_token_overlap(ids, selected_experts, num_experts, top_tokens=8):
    tokens = ids.reshape(-1)
    if selected_experts.shape[0] != tokens.numel():
        return {'error': 'routing snapshot does not match token count'}
    counts = torch.zeros(num_experts, 2048, device=tokens.device, dtype=torch.float32)
    ones = torch.ones_like(tokens, dtype=torch.float32)
    for expert in range(num_experts):
        mask = (selected_experts == expert).any(dim=1)
        if mask.any():
            counts[expert].scatter_add_(0, tokens[mask], ones[mask])
    mass = counts.sum(-1).clamp_min(1.0)
    dist = counts / mass.unsqueeze(-1)
    cosine = dist @ dist.transpose(0, 1)
    off = cosine.clone()
    off.fill_diagonal_(0)
    top = []
    for expert in range(num_experts):
        values, tokens_ids = counts[expert].topk(top_tokens)
        top.append({
            'expert': expert,
            'tokens': [int(token) for token in tokens_ids.cpu().tolist()],
            'counts': [int(value) for value in values.cpu().tolist()],
        })
    return {
        'mean_offdiag_cosine': round(float(off.sum().item() / max(num_experts * (num_experts - 1), 1)), 4),
        'max_offdiag_cosine': round(float(off.max().item()), 4),
        'top_tokens': top,
    }


def build_model(config):
    return ImprovedGPT(config)
