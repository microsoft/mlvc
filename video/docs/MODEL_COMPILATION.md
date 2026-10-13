# Model Compilation

P-frame training uses `torch.compile` on CUDA. Configure it under `train` in the training YAML,
or override it on the command line, for example:

```bash
uv run train_video.py ... --overrides="{ train: { compile_model: false } }"
```

The supported cases are:

| Configuration | Behavior |
| --- | --- |
| `compile_model` omitted or `compile_model: true` | Enables compilation. This is the default. |
| `compile_model: false` | Disables compilation. Use this for debugging, CPU runs, or simulated ANE. |
| Compilation enabled, `compile_backend` omitted | Uses the `inductor` backend. This is the default backend. |
| `compile_backend: <backend>` | Uses the specified `torch.compile` backend, such as `cudagraphs`. |
| `inductor` with `compile_mode` and `compile_options` omitted | Uses `reduce-overhead` mode (CUDA graphs) |
| `compile_mode: <mode>` | Passes the mode to `torch.compile`, for example `max-autotune`. `null` passes no mode. |
| `compile_options: { ... }` | Passes backend options to `torch.compile`, for example `{"triton.cudagraphs": false}`. |

`torch.compile` does not accept `mode` and `options` together, so setting both `compile_mode` and
`compile_options` raises an error. When `compile_options` is set, the default mode is not applied; if
your config sets `compile_mode` elsewhere, set `compile_mode: null` next to `compile_options`.
Changing `compile_backend`, `compile_mode`, `compile_options`, `compile_model` or `cascade` in a
training schedule resets the `torch.compile` caches, frees the CUDA cache and recompiles the model.
With CUDA graphs, the caches are also reset before DeQA testing (see
[Benchmark Testing During Training](#benchmark-testing-during-training)).

Compilation requires a CUDA device. ANE simulation inside the compiled forward with gradient
checkpointing raises an error, because the simulation would otherwise be silently skipped. To
select a backend explicitly, set both options in the YAML or pass an override such as
`--overrides="{ train: { compile_model: true, compile_backend: cudagraphs } }"`.

## Benchmark Testing During Training

Each `benchmark_test` entry is a testset evaluated during training:

- `test_interval`: when regular metrics (bpp, PSNR, LPIPS, VIF, ...) are computed.
- `deqa_test_interval`: when DeQA-Score is added (requires `calc_deqa_score_rgb: true`). Defaults to `test_interval`.

Both take `1` (every epoch, also the default), `0` (never) or `N` (epochs where `(epoch + 1) % N == 0`).
DeQA scores the decoded frames, so it never runs alone: a testset runs on the union of both schedules
and always computes the regular metrics. On epochs without DeQA, DeQA entries in `log_metrics` and
`distortion_metrics` are skipped. To change `deqa_test_interval` in a `schedule`, define it on the
testset first.

On DeQA epochs with CUDA graphs enabled, `torch.compiler.reset()` and `torch.cuda.empty_cache()` run
before testing to free the CUDA graph memory pools for the large DeQA model. Other epochs skip this,
since the model must be recompiled afterwards. GPU memory is logged before and after the reset, with
a per-pool breakdown: pool `(0, 0)` is the default allocator, other ids are CUDA graph pools.
