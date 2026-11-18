# SimpleVQA

[SimpleVQA](https://github.com/sunwei925/SimpleVQA) is a no-reference video-quality
model (Sun et al., *A Deep Learning based No-reference Quality Assessment Model for
UGC Videos*, ACM MM 2022). It combines spatial features from sampled frames with
SlowFast motion features and predicts one quality score per video. This integration
uses the official `UGC_BVQA_model.pth` weights and model code at revision
`017edda50e8cdbeb48113a001dbb9af0dbd2030b`, with the repository's existing RGB
preprocessing.

## License

The adapted implementation is in `src/metrics/simple_vqa/scorer.py`. Upstream code
is licensed under Apache-2.0; the [upstream license](../src/metrics/simple_vqa/LICENSE)
is included alongside the scorer. Repository-specific modifications remain MIT-licensed.
The fixed SlowFast-R50 architecture in `src/metrics/simple_vqa/slowfast.py` is
adapted from [PyTorchVideo 0.1.5](https://github.com/facebookresearch/pytorchvideo/tree/v0.1.5)
under its [Apache-2.0 license](../src/metrics/simple_vqa/LICENSE.pytorchvideo).

## Usage

Follow the [main README's installation instructions](../../README.md#installation):

```bash
uv sync --locked
```

SimpleVQA requires no additional dependency extra. Enable the metric on a
benchmark (`compute_benchmark_metrics.py`) or training-validation
dataset. Checkpoints are read from `SimpleVQA/` under `model.auxiliary.path`:

```yaml
model:
  auxiliary:
    path: pretrained/

benchmark_metrics:          # or benchmark_test for training validation
  my_dataset:
    calc_simple_vqa: true
    # Training only: score every Nth epoch (default: test_interval, 0 disables)
    simple_vqa_test_interval: 5
```

The SimpleVQA and SlowFast architectures are packaged in the repository. Only
`UGC_BVQA_model.pth` and `SLOWFAST_8x8_R50.pyth` are needed in storage.
The required `torchvision` transforms are already provided by the base
dependencies (`lpips` and `pyfacer`).

## Input and output

- **Input:** decoded frames per sequence and QP. Native validation feeds codec
  reconstructions directly; the benchmark adapter decodes existing encoded clips.
  SimpleVQA does not require `decoder_folder`, save/reload an MP4, or provide a
  separate file-path scoring API. Clips must be at least one second long.
  The scorer retains sparse spatial samples and bounded motion windows, not the
  full-resolution video.
- **Preprocessing:** encoded benchmark videos use the existing FFmpeg YUV reader.
  SimpleVQA consumes the same reconstructed RGB tensor as the other RGB metrics,
  converted to HWC uint8 with the same scaling/truncation as DeQA. YUV
  reconstructions use the repository's `ycbcr2rgb`; native RGB reconstructions
  are used directly. No metric-specific decoder, temporary video, or direct
  PyAV conversion is used.
  This intentionally replaces the legacy OpenCV-compatible preprocessing:
  scores can differ from historical file-based SimpleVQA results.
- **Output:** `ave_all_frame_simple_vqa` per sequence and QP in the metrics JSON.
  Higher is better; the score is the model's raw prediction, not normalized to MOS.
  If scoring fails, the value is `null` and `ave_all_frame_simple_vqa_error` explains
  why; other QPs and metrics are unaffected.

## Notes

- `decoder_metrics` does not inherit `calc_simple_vqa`, to avoid scoring the same
  video twice; enable it there explicitly if needed.
- With `frame_step > 1` the motion features see a lower frame rate, so scores are
  not comparable with full-frame-rate scores.
- Explicit decoder outputs remain available for other purposes; they are not
  inputs to SimpleVQA scoring.
- Use `--distortion_metrics simple_vqa` with `compare_rd_video.py` for RD plots
  and BD-rate. The `ave_all_frame_simple_vqa` field uses the existing
  frame-count-weighted aggregation and BD-rate path, as for DeQA.
  The field name follows the existing metric convention: SimpleVQA
  predicts one video score, not an average of independent frame predictions.
- Default frame selection uses `all` only; explicit `i`/`p` is rejected.
  Datasets and clips must match across models. Within each model/dataset,
  every clip must have the same named rate ladder for frame-weighted pooling.
  Anchor and candidate may use different QP labels, I/P QP pairs, bitrate targets
  and numbers of rate points; each curve requires at least three points.
  The anchor must also contain `ave_all_frame_simple_vqa` scores. Missing/nonfinite
  scores, invalid codec rates and nonoverlapping quality ranges fail explicitly,
  rather than becoming zero, partial-population means or extrapolated BD-rates.
  Finite negative scores are valid; negative BD-rate is better. Nonmonotone
  curves use the repository BD-rate helper's existing rate/quality envelope and
  warnings, just like other metrics.
- Bitrate-based QP matching does not use quality scores and is unaffected.
  Use `all_frame_simple_vqa` in `log_metrics`, like `all_frame_deqa_score_rgb`;
  clips that failed scoring are
  left out of the logged mean with a warning.
- Not supported on Windows ARM64.
- Real-weight test: set `SIMPLE_VQA_PRETRAINED_PATH` to the directory containing
  `SimpleVQA/` and run `python -m pytest tests\test_simple_vqa.py` from `video/`.

```powershell
python compare_rd_video.py --base_method anchor `
  --log_paths anchor anchor.json candidate candidate.json `
  --distortion_metrics simple_vqa --compare_frame_type all `
  --plot_path rd_plots --output_path simple_vqa_bdrate.json
```
