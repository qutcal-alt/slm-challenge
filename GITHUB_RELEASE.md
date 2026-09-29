# GitHub upload notes

Push `MP1_submission/` as the immutable code repository.

Recommended layout:
- Git: source, configs, data, report, training/eval JSON logs, the small baseline checkpoint.
- GitHub Release asset: `h1rc2-checkpoint.pt` copied from
  `code/runs/expH1rC2-w320-d6-h4-moe4x320-drop01-4200-cache004-theta12/checkpoint.pt`
  SHA-256 `cc3b4bfc717e23ccb0322e37e1b4696631edb529b4928598f67849713e9cb96d`.

After publishing the Release, paste the asset URL into `code/README.md` section 2.

If Git LFS or the host allows ~43 MiB files, the checkpoint can stay in the
repository at the path above. Do not upload `.venv/`, `__pycache__/`, other
`runs/exp*` checkpoints, or window-nll numpy dumps.
