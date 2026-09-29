"""Validation sweep for frozen-checkpoint neural cache or trigram mix."""
import argparse
import json
import math
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from common import PROTOCOL, ROOT, autocast, load_data, make_model, setup, windows
from ngram import SparseTrigram


def parse_floats(text):
    return [float(part) for part in text.split(',') if part.strip()]


def mix_nll(model_probs, aux_probs, mix, targets):
    if mix <= 0.0:
        mixed = model_probs
    else:
        mixed = (1.0 - mix) * model_probs + mix * aux_probs
        if aux_probs is not None:
            has_aux = aux_probs.sum(-1, keepdim=True) > 0
            mixed = torch.where(has_aux, mixed, model_probs)
    logp = torch.log(mixed.clamp_min(1e-12))
    losses = -logp.gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    losses.masked_fill_(targets == -100, 0)
    return float(losses.double().sum().item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--precision', choices=['auto', 'fp32', 'bf16'], default='fp32')
    parser.add_argument('--mode', choices=['cache', 'trigram'], required=True)
    parser.add_argument('--lambdas', default='0,0.05,0.1,0.2,0.3')
    parser.add_argument('--thetas', default='1,5,10,20')
    parser.add_argument('--ngram-path', type=Path, default=ROOT / 'assets' / 'trigram.pt')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    device, precision = setup(args.device, args.precision, args.threads)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['protocol'] != PROTOCOL:
        raise ValueError('Checkpoint belongs to a different course protocol.')
    model, _ = make_model(checkpoint['implementation'], checkpoint['config'], device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    mixes = parse_floats(args.lambdas)
    thetas = parse_floats(args.thetas) if args.mode == 'cache' else [None]
    ngram = SparseTrigram(args.ngram_path, device=device) if args.mode == 'trigram' else None
    tokens, byte_count = load_data()['validation']
    started = time.perf_counter()
    totals = {(theta, mix): 0.0 for theta in thetas for mix in mixes}
    with torch.no_grad():
        for x, y in windows(tokens, 32):
            x, y = x.to(device), y.to(device)
            with autocast(device, precision):
                hidden = model.features(x)
                model_probs = F.softmax(model.head(hidden).float(), dim=-1)
            if args.mode == 'trigram':
                aux = ngram.probabilities(x)
                for mix in mixes:
                    totals[(None, mix)] += mix_nll(model_probs, aux, mix, y)
            else:
                saved_theta = model.cache_theta
                for theta in thetas:
                    model.cache_theta = theta
                    cache = model._window_neural_cache(hidden, x)
                    for mix in mixes:
                        totals[(theta, mix)] += mix_nll(model_probs, cache, mix, y)
                model.cache_theta = saved_theta
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    rows = []
    for (theta, mix), nll in totals.items():
        row = {
            'mode': args.mode,
            'lambda': mix,
            'bpb': nll / math.log(2) / byte_count,
            'nll_nats': nll,
            'targets': 376599,
            'utf8_bytes': byte_count,
        }
        if theta is not None:
            row['theta'] = theta
        rows.append(row)
        print(json.dumps(row), flush=True)
    best = min(rows, key=lambda row: row['bpb'])
    payload = {
        'checkpoint': str(args.checkpoint),
        'mode': args.mode,
        'seconds': seconds,
        'best': best,
        'rows': rows,
    }
    output = args.output or Path('runs') / f'sweep-{args.mode}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({'best': best, 'seconds': seconds, 'output': str(output)}, indent=2))


if __name__ == '__main__':
    main()
