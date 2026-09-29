"""Measure checkpoint assets and peak process RAM during official scoring."""
import argparse
import json
from pathlib import Path
import ctypes
from ctypes import wintypes
import torch
from common import PROTOCOL, ROOT, load_data, make_model, setup, sha
from evaluate import score


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ('cb', wintypes.DWORD),
        ('PageFaultCount', wintypes.DWORD),
        ('PeakWorkingSetSize', ctypes.c_size_t),
        ('WorkingSetSize', ctypes.c_size_t),
        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
        ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
        ('PagefileUsage', ctypes.c_size_t),
        ('PeakPagefileUsage', ctypes.c_size_t),
        ('PrivateUsage', ctypes.c_size_t),
    ]


def peak_working_set_bytes():
    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    get_process = kernel32.GetCurrentProcess
    get_process.restype = wintypes.HANDLE
    query = getattr(kernel32, 'K32GetProcessMemoryInfo', None)
    if query is None:
        query = ctypes.WinDLL('psapi', use_last_error=True).GetProcessMemoryInfo
    query.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD]
    query.restype = wintypes.BOOL
    if not query(get_process(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(counters.PeakWorkingSetSize)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    device, precision = setup('cpu', 'fp32', args.threads)
    checkpoint_path = args.checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if ckpt['protocol'] != PROTOCOL:
        raise ValueError('Checkpoint belongs to a different course protocol.')
    model, implementation_sha = make_model(ckpt['implementation'], ckpt['config'], device)
    model.load_state_dict(ckpt['model'])
    data = load_data()
    peak_working_set_bytes()
    result = score(model, *data[args.split], device, precision)
    result.pop('window_nll_nats')
    peak = peak_working_set_bytes()
    asset_bytes = checkpoint_path.stat().st_size
    payload = {
        **result,
        'split': args.split,
        'precision': precision,
        'checkpoint': str(checkpoint_path),
        'checkpoint_config': ckpt['config'],
        'checkpoint_sha256': sha(checkpoint_path),
        'implementation_sha256': implementation_sha,
        'student_sha256': sha(ROOT / 'student.py'),
        'ngram_sha256': sha(ROOT / 'ngram.py'),
        'evaluator_sha256': sha(ROOT / 'evaluate.py'),
        'asset_bytes': asset_bytes,
        'asset_mib': asset_bytes / 1024 / 1024,
        'peak_working_set_bytes': peak,
        'peak_working_set_gib': peak / (1024 ** 3),
        'baseline_cpu_seconds': 5.92,
        'cpu_limit_seconds': 5.92 * 5,
        'within_time_5x': result['seconds'] <= 5.92 * 5,
        'within_ram_4gib': peak <= 4 * (1024 ** 3),
        'within_assets_64mib': asset_bytes <= 64 * 1024 * 1024,
    }
    output = args.output or checkpoint_path.parent / f'{args.split}_cpu_limits.json'
    output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
