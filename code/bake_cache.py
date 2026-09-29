"""Write a frozen checkpoint with neural-cache mix settings. Run from code/."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from common import PROTOCOL


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-mix', type=float, default=0.04)
    parser.add_argument('--cache-theta', type=float, default=12.0)
    args = parser.parse_args()
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if ckpt['protocol'] != PROTOCOL:
        raise ValueError('Checkpoint belongs to a different course protocol.')
    config = dict(ckpt['config'])
    config['cache_mix'] = float(args.cache_mix)
    config['cache_theta'] = float(args.cache_theta)
    payload = {**ckpt, 'config': config}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    source = {
        'source_checkpoint': str(args.checkpoint),
        'source_sha256': sha256(args.checkpoint),
        'cache_mix': config['cache_mix'],
        'cache_theta': config['cache_theta'],
        'config': config,
        'checkpoint_sha256': sha256(args.output),
    }
    (args.output.parent / 'source.json').write_text(json.dumps(source, indent=2) + '\n')
    print(json.dumps(source, indent=2))


if __name__ == '__main__':
    main()
