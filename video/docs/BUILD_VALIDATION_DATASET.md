# Creating a Validation Set

## 1. Download raw clips

Place source MP4 files into the `raw/` folder. For example, download a [subset](https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVidHD/OpenVidHD_part_29_part_aa?download=true) of OpenVidHD from HuggingFace:

## 2. Convert raw clips and generate json description

```bash
python create_validation_set.py
```

Main settings:

| Flag | Default | Description |
|---|---|---|
| `--resolutions` | `1920x1080 960x540` | Output resolutions (any number of `WxH`) |
| `--frames` | `300` | Exact frame count per clip (shorter videos are skipped) |
| `--max-sequences` | `50` | Required number of sequences (fails if not enough, trims excess) |
| `--workers` | `4` | Parallel FFmpeg processes |
| `--overwrite` | off | Re-convert existing files |
| `--dry-run` | off | Preview without converting |

See `python create_validation_set.py --help` for the full list of options.

## 3. Run evaluation

Validate models on the new dataset - see **Running Evaluation** in `README.md`.

## Optional: prepare restored reference frames

`run_basicvsrpp.py` can restore either image sequences described by
`dataset.description_path` or raw YUV420 sequences described by `dataset.config`
and `dataset.data_path`. Configure exactly one description mode and supply
BasicVSR++ checkpoint weights using `model.weights_path`.

Create a configuration for your dataset and checkpoint using the settings
above. From the `video` directory:

```powershell
python run_basicvsrpp.py --config=path/to/restoration_config.yaml
```

The YUV mode writes restored files and a dataset description preserving sequence
alignment under the configured save directory's `restored` folder. It writes
`restoration_manifest.json` in the save directory alongside that folder.
Keep the description and provenance with the restored files.

Description paths must be relative and contain no parent-directory (`..`)
components: image-sequence `path` values and frame names, and YUV `base_path`
values and sequence names. Paths escaping the input or output root, and
source/output collisions, are rejected before writing a sequence.

Both modes regenerate selected outputs on every run; existing files are not
reused.

`model.overlap` is the context margin on each side of a chunk. It must be
nonnegative and `model.chunk_size - 2 * model.overlap` must be positive.
