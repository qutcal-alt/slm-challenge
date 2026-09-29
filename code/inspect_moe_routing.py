"""Per-layer MoE routing on a frozen checkpoint; training logits are unchanged."""
import argparse
import json
from pathlib import Path
import torch
from common import PROTOCOL, load_data, make_model, setup, windows
from student import MoEFFN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'validation'], default='validation')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batches', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    device, _ = setup(args.device, 'fp32', 4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['protocol'] != PROTOCOL:
        raise ValueError('Checkpoint belongs to a different course protocol.')
    model, _ = make_model(checkpoint['implementation'], checkpoint['config'], device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    tokens = load_data()[args.split][0]
    usage_sum = None
    prob_sum = None
    token_counts = None
    seen = 0
    with torch.no_grad():
        for x, _ in windows(tokens, args.batch_size):
            x = x.to(device)
            model.forward_with_aux(x)
            layers = []
            for block in model.blocks:
                if not isinstance(block.mlp, MoEFFN) or not hasattr(block.mlp, 'last_selected_fraction'):
                    continue
                layers.append(block.mlp)
            if not layers:
                raise ValueError('Checkpoint has no MoE layers.')
            usage = torch.stack([layer.last_selected_fraction for layer in layers])
            prob = torch.stack([layer.last_mean_probability for layer in layers])
            usage_sum = usage if usage_sum is None else usage_sum + usage
            prob_sum = prob if prob_sum is None else prob_sum + prob
            if token_counts is None:
                token_counts = [
                    torch.zeros(layer.num_experts, 2048, device=device)
                    for layer in layers
                ]
            flat = x.reshape(-1)
            ones = torch.ones_like(flat, dtype=torch.float32)
            for layer_index, layer in enumerate(layers):
                selected = layer.last_selected_experts
                for expert in range(layer.num_experts):
                    mask = (selected == expert).any(dim=1)
                    if mask.any():
                        token_counts[layer_index][expert].scatter_add_(0, flat[mask], ones[mask])
            seen += 1
            if seen >= args.batches:
                break
    rows = []
    for layer_index, counts in enumerate(token_counts):
        usage = usage_sum[layer_index] / seen
        mean_prob = prob_sum[layer_index] / seen
        dist = counts / counts.sum(-1, keepdim=True).clamp_min(1.0)
        cosine = dist @ dist.transpose(0, 1)
        off = cosine.clone()
        off.fill_diagonal_(0)
        experts = int(counts.shape[0])
        top = []
        for expert in range(experts):
            values, tokens_ids = counts[expert].topk(8)
            top.append({
                'expert': expert,
                'tokens': [int(token) for token in tokens_ids.cpu().tolist()],
                'counts': [int(value) for value in values.cpu().tolist()],
            })
        rows.append({
            'layer': layer_index,
            'usage': [round(float(value), 4) for value in usage.cpu().tolist()],
            'mean_prob': [round(float(value), 4) for value in mean_prob.cpu().tolist()],
            'usage_max': round(float(usage.max().item()), 4),
            'usage_min': round(float(usage.min().item()), 4),
            'mean_offdiag_cosine': round(float(off.sum().item() / max(experts * (experts - 1), 1)), 4),
            'max_offdiag_cosine': round(float(off.max().item()), 4),
            'top_tokens': top,
        })
    payload = {
        'checkpoint': str(args.checkpoint),
        'split': args.split,
        'batches': seen,
        'mean_usage': [round(float(value), 4) for value in (usage_sum.mean(0) / seen).cpu().tolist()],
        'layers': rows,
    }
    output = args.output or args.checkpoint.parent / f'routing_{args.split}.json'
    output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({key: payload[key] for key in ('checkpoint', 'split', 'batches', 'mean_usage')}, indent=2))
    print(json.dumps([{'layer': row['layer'], 'usage': row['usage'],
                       'mean_offdiag_cosine': row['mean_offdiag_cosine'],
                       'max_offdiag_cosine': row['max_offdiag_cosine']} for row in rows], indent=2))


if __name__ == '__main__':
    main()
