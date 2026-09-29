"""Build a pruned training trigram table. Run from code/."""
import argparse
from pathlib import Path
import torch
from common import load_data
from ngram import build_trigram_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('assets/trigram.pt'))
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--min-trigram-count', type=int, default=2)
    args = parser.parse_args()
    tokens, _ = load_data()['train']
    table = build_trigram_table(tokens, alpha=args.alpha, min_trigram_count=args.min_trigram_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(table, args.output)
    nbytes = args.output.stat().st_size
    print({
        'output': str(args.output),
        'bytes': nbytes,
        'mib': round(nbytes / 1024 / 1024, 3),
        'bigrams': int(table['bigram_token'].numel()),
        'trigrams': int(table['trigram_token'].numel()),
        'trigram_contexts': int(table['trigram_pair'].unique().numel()),
        'alpha': table['alpha'],
        'min_trigram_count': table['min_trigram_count'],
    })


if __name__ == '__main__':
    main()
