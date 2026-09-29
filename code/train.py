"""Default recipe: 1,200 steps x 32 sequences x 256 targets = 9,830,400 tokens."""
import argparse
import json
import math
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from common import PROTOCOL, ROOT, autocast, device_metrics, load_data, make_model, setup, sha
from evaluate import score


def score_with_optional_ema(model, ema_state, eval_fn, *args, **kwargs):
    if ema_state is None:
        return eval_fn(model, *args, **kwargs)
    sync_shared_ema_entries(ema_state, parameter_aliases(model))
    raw_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    was_training = model.training
    try:
        model.load_state_dict(ema_state)
        model.eval()
        with torch.no_grad():
            return eval_fn(model, *args, **kwargs)
    finally:
        model.load_state_dict(raw_state)
        model.train(was_training)


def checkpoint_payload(implementation, config, model_state, seed, train_tokens):
    return {'protocol': PROTOCOL, 'implementation': implementation, 'config': config,
            'model': model_state, 'seed': seed, 'train_tokens': train_tokens}


def cpu_state(state):
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def parameter_aliases(model):
    aliases = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    return list(aliases.values())


def unique_trainable_ema_pairs(model):
    seen = set()
    pairs = []
    for name, parameter in model.named_parameters(remove_duplicate=False):
        key = id(parameter)
        if key in seen:
            continue
        seen.add(key)
        pairs.append((name, parameter))
    return pairs


def sync_shared_ema_entries(ema_state, aliases):
    if ema_state is None:
        return ema_state
    for names in aliases:
        canonical = ema_state[names[0]]
        for name in names[1:]:
            ema_state[name].copy_(canonical)
    return ema_state


def moe_expert_usage(model):
    if not hasattr(model, 'expert_usage'):
        return None
    usage = model.expert_usage()
    if usage is None:
        return None
    return {'mean': [round(float(value), 4) for value in usage.detach().cpu().tolist()]}


def export_state_for_checkpoint(model, ema_state, aliases=None):
    if ema_state is None:
        return cpu_state(model.state_dict())
    sync_shared_ema_entries(ema_state, aliases or parameter_aliases(model))
    raw_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    try:
        model.load_state_dict(ema_state)
        if hasattr(model, 'export_int8_state'):
            return model.export_int8_state()
        return cpu_state(model.state_dict())
    finally:
        model.load_state_dict(raw_state)


def main():
    total_started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', default='student')
    p.add_argument('--config', type=Path, default=ROOT/'configs/baseline.json')
    p.add_argument('--run-dir', type=Path, default=ROOT/'runs/baseline-s17')
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=['auto','fp32','bf16'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--lr-horizon', type=int, default=0,
                   help='Cosine decay horizon; defaults to --steps.')
    p.add_argument('--lr', type=float, default=1e-3, help='Peak learning rate.')
    p.add_argument('--warmup-steps', type=int, default=100)
    p.add_argument('--min-lr-ratio', type=float, default=0.1,
                   help='Cosine floor as a fraction of peak learning rate.')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--ema-decay', type=float, default=0.0,
                   help='Optional EMA decay; 0 disables EMA.')
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--adam-beta1', type=float, default=0.9)
    p.add_argument('--adam-beta2', type=float, default=0.999,
                   help='AdamW second-moment coefficient; default 0.999.')
    p.add_argument('--moe-aux-weight', type=float, default=0.01,
                   help='Coefficient for the MoE routing balance loss.')
    p.add_argument('--eval-every', type=int, default=0,
                   help='Optional validation-curve interval; 0 evaluates only after training.')
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        p.error('Batch size and step count must be positive.')
    if args.lr_horizon < 0:
        p.error('Learning-rate horizon must be nonnegative.')
    if args.lr <= 0 or args.warmup_steps < 1:
        p.error('Peak learning rate must be positive and warmup steps must be at least 1.')
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        p.error('Minimum learning-rate ratio must be in [0, 1].')
    lr_horizon = args.lr_horizon or args.steps
    if not 0.0 <= args.ema_decay < 1.0:
        p.error('EMA decay must be in [0, 1).')
    if not 0.0 <= args.adam_beta1 < 1.0 or not 0.0 <= args.adam_beta2 < 1.0:
        p.error('AdamW betas must be in [0, 1).')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Run directory already contains results. Use a new --run-dir.')
    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)
    prepared = time.perf_counter()
    data = load_data()
    config = json.loads(args.config.read_text())
    model, implementation_sha = make_model(args.implementation, config, device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    no_decay_terms = ('norm', 'rope', 'router', 'bias', 'smear_gate')
    decay_parameters = []
    no_decay_parameters = []
    for name, parameter in model.named_parameters():
        (no_decay_parameters if any(term in name.lower() for term in no_decay_terms)
         else decay_parameters).append(parameter)
    optimizer = torch.optim.AdamW([
        {'params': decay_parameters, 'weight_decay': args.weight_decay},
        {'params': no_decay_parameters, 'weight_decay': 0.0},
    ], lr=args.lr, betas=(args.adam_beta1, args.adam_beta2))
    ema_state = None
    ema_parameters = []
    shared_parameter_names = parameter_aliases(model)
    if args.ema_decay > 0:
        ema_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        ema_parameters = unique_trainable_ema_pairs(model)
    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter()-prepared
    started = time.perf_counter()
    history = []
    validation_history = []
    best_val_bpb = float('inf')
    best_step = 0
    best_model_state = None
    intermediate_validation_seconds = 0.
    tokens_per_step = args.batch_size * 256
    train_tokens = args.steps * tokens_per_step
    for step in range(args.steps):
        starts = torch.randint(len(tokens)-257, (args.batch_size,), generator=rng).to(device)
        batch = tokens[starts[:,None]+torch.arange(257,device=device)]
        progress = min(1.0, step / lr_horizon)
        learning_rate = (args.lr * min(1., (step+1)/args.warmup_steps)
                         * (args.min_lr_ratio + (1.0-args.min_lr_ratio)*.5*(1+math.cos(math.pi*progress))))
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            logits, moe_loss = (model.forward_with_aux(batch[:,:-1])
                                if hasattr(model, 'forward_with_aux')
                                else (model(batch[:,:-1]), None))
            loss = F.cross_entropy(logits.flatten(0,1).float(),batch[:,1:].flatten())
            if moe_loss is not None:
                loss = loss + args.moe_aux_weight * moe_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if ema_state is not None:
            with torch.no_grad():
                for name, parameter in ema_parameters:
                    ema_state[name].mul_(args.ema_decay).add_(parameter.detach(), alpha=1.0 - args.ema_decay)
                sync_shared_ema_entries(ema_state, shared_parameter_names)
        if (step+1)%100 == 0 or step+1 == args.steps:
            row = {'step':step+1,'loss':loss.item(),'seconds':time.perf_counter()-started-intermediate_validation_seconds}
            if (step+1)%500 == 0 or step+1 == args.steps:
                expert_usage = moe_expert_usage(model)
                if expert_usage is not None:
                    row['expert_usage'] = expert_usage['mean']
            history.append(row)
            print(json.dumps(row),flush=True)
        if args.eval_every > 0 and (step+1)%args.eval_every == 0:
            expert_usage = moe_expert_usage(model)
            intermediate = score_with_optional_ema(
                model, ema_state, score, *data['validation'], device, 'fp32'
            )
            intermediate.pop('window_nll_nats')
            intermediate_validation_seconds += intermediate['seconds']
            if expert_usage is not None:
                intermediate['expert_usage'] = expert_usage['mean']
            validation_history.append({'step':step+1,**intermediate})
            if intermediate['bpb'] < best_val_bpb:
                best_val_bpb = intermediate['bpb']
                best_step = step + 1
                best_model_state = export_state_for_checkpoint(model, ema_state, shared_parameter_names)
                torch.save(
                    checkpoint_payload(
                        args.implementation, config, best_model_state, args.seed, (step + 1) * tokens_per_step
                    ),
                    args.run_dir / 'best_checkpoint.pt'
                )
            print(json.dumps({'validation':validation_history[-1]}),flush=True)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter()-started-intermediate_validation_seconds
    if ema_state is not None:
        sync_shared_ema_entries(ema_state, shared_parameter_names)
        model.load_state_dict(ema_state)
    model.eval()
    last_validation = score(model,*data['validation'],device,'fp32')
    last_validation.pop('window_nll_nats')
    if last_validation['bpb'] < best_val_bpb:
        best_val_bpb = last_validation['bpb']
        best_step = args.steps
        best_model_state = export_state_for_checkpoint(model, ema_state, shared_parameter_names)
        torch.save(
            checkpoint_payload(
                args.implementation, config, best_model_state, args.seed, args.steps * tokens_per_step
            ),
            args.run_dir / 'best_checkpoint.pt'
        )
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    # `validation` always scores the weights written to checkpoint.pt.
    if best_model_state is None or best_step == args.steps:
        validation = last_validation
    else:
        validation = score(model, *data['validation'], device, 'fp32')
        validation.pop('window_nll_nats')
        best_val_bpb = validation['bpb']
    checkpoint = args.run_dir/'checkpoint.pt'
    model_state = model.export_int8_state() if hasattr(model, 'export_int8_state') else model.cpu().state_dict()
    torch.save(
        checkpoint_payload(
            args.implementation, config, model_state, args.seed, best_step * tokens_per_step
        ),
        checkpoint
    )
    result = {'protocol':PROTOCOL,'implementation':args.implementation,'config':config,'seed':args.seed,
              'ema_decay':args.ema_decay,'lr':args.lr,'warmup_steps':args.warmup_steps,
              'min_lr_ratio':args.min_lr_ratio,'lr_horizon':lr_horizon,
              'weight_decay':args.weight_decay,'adam_beta1':args.adam_beta1,
              'adam_beta2':args.adam_beta2,'moe_aux_weight':args.moe_aux_weight,
              'parameters':sum(p.numel() for p in model.parameters()),'precision':precision,
              'train_tokens':train_tokens,'best_train_tokens':best_step * tokens_per_step,
              'best_val_bpb':best_val_bpb,'best_step':best_step,
              'train_seconds':train_seconds,'validation':validation,
              'last_validation':last_validation,'history':history,
              'validation_history':validation_history,
              'intermediate_validation_seconds':intermediate_validation_seconds,
              'process_seconds':time.perf_counter()-total_started,
              'torch_version':str(torch.__version__),'threads':args.threads,
              'checkpoint_sha256':sha(checkpoint),'implementation_sha256':implementation_sha,
              **device_metrics(device)}
    (args.run_dir/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result|{'history':[]},indent=2),flush=True)


if __name__ == '__main__':
    main()
