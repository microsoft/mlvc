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
