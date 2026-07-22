# Building image sequences dataset

In case you want to build your own training dataset from scratch, follow the instruction [Building training dataset from scratch](#building-training-dataset-from-scratch) below.
If you want to reproduce the OpenVidHD dataset used for model training, refer to the section [Reproduction of the OpenVidHD training dataset](#reproduction-of-the-openvidhd-training-dataset) at the bottom.

## Building training dataset from scratch
1. Download the dataset into `{DATASET_PATH}` and create a list of relative clip paths in `clip_paths.txt`.

**An example of the dataset directory structure:**
```
DATASET_PATH/
├── subfolder_0000/
│   ├── video_clip_00000.mp4
│   ├── video_clip_00001.mp4
│   ├── video_clip_00002.mp4
│   └── ...
├── subfolder_0001/
│   ├── video_clip_00000.mp4
│   ├── video_clip_00001.mp4
│   ├── video_clip_00002.mp4
│   └── ...
├── subfolder_0002/
│   └── ...
└── ...
```
**An example of `clip_paths.txt`:**
```
subfolder_0000/video_clip_00000.mp4
subfolder_0000/video_clip_00001.mp4
subfolder_0000/video_clip_00002.mp4
...
subfolder_0001/video_clip_00000.mp4
subfolder_0001/video_clip_00001.mp4
subfolder_0001/video_clip_00002.mp4
...
```
2. Modify the `source_dir` and `video_list` fields in the config used for the `run_ffprobe.py` script to match your data (see an example in `configs/dataset/run_ffprobe.yaml`) and run the `ffprobe` script to collect clip metadata:
```commandline
python run_ffprobe.py --config=configs/dataset/run_ffprobe.yaml
```
3. Download job results and build `clip_meta.json`:
```commandline
python build_clip_meta.py <path to run_ffprobe results> clip_meta.json
```
4. Modify `clip_folder` and verify that the path to `clip_meta.json` is correct in the config used for the `run_optic_flow.py` script (see an example in `configs/dataset/run_optic_flow.yaml`). Then, run optical flow calculation and collect per-clip statistics:
```commandline
python run_optic_flow.py --config=configs/dataset/run_optic_flow.yaml
```
5. Download the results of `run_optic_flow.py`, i.e., optical flow and other statistics grouped by clip. Modify the config used for the `build_frame_sequences.py` script (see an example in `configs/dataset/build_frame_sequences.yaml`). In particular, `sequence_length` and `n_sequences` can be changed to set the desired sequence length (number of frames per sequence) and the number of sequences to generate. Then, build `frame_sequences.csv`:
```commandline
python build_frame_sequences.py --config=configs/dataset/build_frame_sequences.yaml <path to collected statistics> frame_sequences.csv
```
6. Modify the `source_dir` and `sequence_range` fields in the config used for the `extract_frame_sequences.py` script (see an example in `configs/dataset/extract_frame_sequences.yaml`). Run the sequence extraction job:
```commandline
python extract_frame_sequences.py --config=configs/dataset/extract_frame_sequences.yaml
```
NB: Instead of running the script for the full sequence range, you may run several jobs in parallel, specifying a subset `sequence_range` in the config for each job, e.g., `[0, 1000]`, `[1000, 2000]` and `[2000, 3000]` instead of `[0, 3000]`.

7. Build `description.json`:
```commandline
python build_dataset_description.py frame_sequences.csv description.json
```
8. Copy `description.json` and `frame_sequences.csv` into the output dataset folder.

9. [Optional] Compute face segmentation masks, which are required for training with LPIPS-ROI loss. Set the `path` field in the config for the `compute_segmentation_masks.py` script. Run segmentation:
```commandline
python compute_segmentation_masks.py --config=configs/dataset/compute_segmentation_masks.yaml
```
Copy `masks` to the output dataset folder. Besides, segmentation might have failed for some sequences. In this case, the script will produce `failed_paths*.txt` files and an updated `description.json`. Substitute the old `description.json` with the new one.

## Reproduction of the OpenVidHD training dataset
1. Download a subset of OpenVidHD from [here](https://huggingface.co/datasets/nkp37/OpenVid-1M/tree/main/OpenVidHD). Only parts 10-28 are required. Besides, `*_part_ab` subsets can be skipped.
2. The file [openvidhd_parts_10-28_videos.txt](https://mlvideopub.blob.core.windows.net/mlvc/datasets/OpenVidHD_parts_10-28/openvidhd_parts_10-28_videos.txt) can be used to verify that all necessary videos were downloaded at the previous step.
3. Follow steps 6-9 from the instruction above. Make sure to specify [openvidhd_60k64_frame_sequences.csv](https://mlvideopub.blob.core.windows.net/mlvc/datasets/OpenVidHD_parts_10-28/openvidhd_60k64_frame_sequences.csv) in `configs/dataset/extract_frame_sequences.yaml`. The sub-index `60k64` refers to a dataset containing 60,000 sequences, each with 64 frames. In addition, you can generate a dataset using [openvidhd_11k150_frame_sequences.csv](https://mlvideopub.blob.core.windows.net/mlvc/datasets/OpenVidHD_parts_10-28/openvidhd_11k150_frame_sequences.csv) or [openvidhd_3k300_frame_sequences](https://mlvideopub.blob.core.windows.net/mlvc/datasets/OpenVidHD_parts_10-28/openvidhd_3k300_frame_sequences.csv) for training on longer sequences.
