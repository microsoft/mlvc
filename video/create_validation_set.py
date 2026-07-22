# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Convert raw MP4 videos to YUV 4:2:0 planar format at configurable resolutions.

Produces the folder structure expected by video codec test harnesses:
    yuv/{W}x{H}_{fps}fps/{test_class}/{name}_{W}x{H}_{fps}fps.yuv

Also generates JSON manifest files describing the sequences.

Requirements:
    - FFmpeg on PATH
    - Python 3.10+
    - No external packages (stdlib only)

Usage examples:
    # Default: convert all MP4s in raw/ to 1920x1080 and 960x540 YUV at 30fps
    python create_validation_set.py

    # Custom resolutions and workers
    python create_validation_set.py --resolutions 1920x1080 1280x720 --workers 8

    # Dry-run to see what would be converted
    python create_validation_set.py --dry-run

    # Overwrite existing files
    python create_validation_set.py --overwrite
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# helpers
def check_ffmpeg() -> str:
    """Return the ffmpeg path or exit with an error."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        first_line = result.stdout.split("\n")[0]
        log.info("Found %s", first_line)
        return "ffmpeg"
    except FileNotFoundError:
        log.error("ffmpeg not found on PATH. Please install FFmpeg first.")
        sys.exit(1)


def probe_frame_count(mp4_path: str, fps: int) -> int:
    """
    Return the number of video frames available at the given fps.

    Uses ffprobe to get duration/nb_frames, then converts to target fps.
    Returns -1 on any failure (caller should skip the video).
    """
    try:
        # fmt: off
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=duration,nb_frames,r_frame_rate",
                "-of", "json",
                mp4_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # fmt: on
        if proc.returncode != 0:
            return -1
        stream = json.loads(proc.stdout).get("streams", [{}])[0]

        # Try nb_frames + source fps to compute duration, then re-scale
        nb = stream.get("nb_frames")
        if nb and nb != "N/A":
            nb = int(nb)
            src_rate = stream.get("r_frame_rate", "")
            if src_rate and "/" in src_rate:
                num, den = src_rate.split("/")
                src_fps = int(num) / int(den) if int(den) else 0
                if src_fps > 0:
                    return int((nb / src_fps) * fps)
            return nb

        # Fallback: use duration field
        dur = stream.get("duration")
        if dur and dur != "N/A":
            return int(float(dur) * fps)
    except Exception:
        pass

    return -1


def parse_resolution(s: str) -> tuple:
    """Parse 'WxH' into (width, height) ints."""
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError(f"Invalid resolution '{s}'. Expected format: WxH (e.g. 1920x1080)")


def detect_dataset_name(raw_dir: Path) -> str:
    """Derive dataset name from folder structure (parent of raw/)."""
    parent = raw_dir.resolve().parent
    return parent.name


def output_yuv_name(stem: str, w: int, h: int, fps: int) -> str:
    """Build output YUV filename from the MP4 stem."""
    return f"{stem}_{w}x{h}_{fps}fps.yuv"


def resolution_folder(w: int, h: int, fps: int) -> str:
    """Build the resolution subfolder name."""
    return f"{w}x{h}_{fps}fps"


def json_filename(dataset_name: str, w: int, h: int, fps: int, seq_count: int, frames: int) -> str:
    """Build the JSON manifest filename."""
    return f"{dataset_name}_{w}x{h}_{fps}fps_{seq_count}s{frames}f.json"


def expected_yuv_size(width: int, height: int, frames: int) -> int:
    """Expected file size for YUV 4:2:0 planar: W * H * 1.5 * frames."""
    return int(width * height * 1.5 * frames)


# conversion
def convert_one(
    mp4_path: str,
    output_path: str,
    width: int,
    height: int,
    fps: int,
    frames: int,
    overwrite: bool,
) -> dict:
    """
    Convert a single MP4 to raw YUV 4:2:0.

    Returns a dict with keys: input, output, resolution, success, skipped, error.
    """
    result = {
        "input": mp4_path,
        "output": output_path,
        "resolution": f"{width}x{height}",
        "success": False,
        "skipped": False,
        "error": None,
    }

    want_size = expected_yuv_size(width, height, frames)

    # Skip if output exists and has the correct size
    if not overwrite and os.path.isfile(output_path) and os.path.getsize(output_path) == want_size:
        result["success"] = True
        result["skipped"] = True
        return result

    # fmt: off
    cmd = [
        "ffmpeg",
        "-y",                       # overwrite output
        "-i", mp4_path,
        "-vf", f"scale={width}:{height}",
        "-r", str(fps),
        "-frames:v", str(frames),
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    # fmt: on

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10-minute timeout per file
        )
        if proc.returncode != 0:
            result["error"] = proc.stderr[-500:] if proc.stderr else "unknown error"
        else:
            # Validate output size matches exactly
            actual_size = os.path.getsize(output_path) if os.path.isfile(output_path) else 0
            if actual_size != want_size:
                result["error"] = (
                    f"Size mismatch: got {actual_size} bytes, "
                    f"expected {want_size} ({frames} frames). "
                    f"Source video may be too short."
                )
                # Remove the bad file
                try:
                    os.remove(output_path)
                except OSError:
                    pass
            else:
                result["success"] = True
    except subprocess.TimeoutExpired:
        result["error"] = "FFmpeg timed out (600s)"
    except Exception as e:
        result["error"] = str(e)

    return result


# JSON manifest
def generate_manifest(
    output_dir: Path,
    dataset_name: str,
    test_class: str,
    width: int,
    height: int,
    fps: int,
    frames: int,
    gop: int,
) -> Path:
    """Generate the JSON manifest for one resolution folder."""
    res_folder = output_dir / resolution_folder(width, height, fps)
    tc_folder = res_folder / test_class
    want_size = expected_yuv_size(width, height, frames)

    # Collect only YUV files with the exact expected size
    sequences = {}
    if tc_folder.is_dir():
        for yuv_file in sorted(tc_folder.iterdir()):
            if yuv_file.suffix == ".yuv" and yuv_file.stat().st_size == want_size:
                sequences[yuv_file.name] = {
                    "width": width,
                    "height": height,
                    "frames": frames,
                    "gop": gop,
                }

    manifest = {
        "test_classes": {
            test_class: {
                "base_path": test_class,
                "src_type": "yuv420",
                "sequences": sequences,
            }
        }
    }

    seq_count = len(sequences)
    fname = json_filename(dataset_name, width, height, fps, seq_count, frames)
    manifest_path = res_folder / fname

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    log.info("Wrote manifest: %s (%d sequences)", manifest_path, seq_count)
    return manifest_path


# main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert raw MP4 videos to YUV 4:2:0 at multiple resolutions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("raw"),
        help="Directory containing source .mp4 files (default: raw/)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("yuv"),
        help="Root output directory (default: yuv/)",
    )
    p.add_argument(
        "--resolutions",
        nargs="+",
        default=["1920x1080", "960x540"],
        help="Output resolutions as WxH (default: 1920x1080 960x540)",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Output frame rate (default: 30)",
    )
    p.add_argument(
        "--frames",
        type=int,
        default=300,
        help="Exact number of frames required per video (default: 300). Source videos with fewer frames are skipped.",
    )
    p.add_argument(
        "--gop",
        type=int,
        default=32,
        help="GOP size for JSON metadata (default: 32)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel FFmpeg processes (default: 4)",
    )
    p.add_argument(
        "--test-class",
        default="s1",
        help="Subfolder / test class name (default: s1)",
    )
    p.add_argument(
        "--dataset-name",
        default=None,
        help="Dataset name prefix for JSON filename (default: auto-detect from parent folder)",
    )
    p.add_argument(
        "--max-sequences",
        type=int,
        default=50,
        help="Maximum number of sequences to include (default: 50). Excess videos are skipped.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-convert even if output .yuv already exists",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without converting",
    )
    return p


def main():
    args = build_parser().parse_args()

    # --- Validate inputs ---
    raw_dir: Path = args.raw_dir.resolve()
    output_dir: Path = args.output_dir.resolve()

    if not raw_dir.is_dir():
        log.error("Raw directory does not exist: %s", raw_dir)
        sys.exit(1)

    resolutions = [parse_resolution(r) for r in args.resolutions]
    dataset_name = args.dataset_name or detect_dataset_name(raw_dir)

    log.info("Dataset name : %s", dataset_name)
    log.info("Raw directory: %s", raw_dir)
    log.info("Output dir   : %s", output_dir)
    log.info("Resolutions  : %s", [f"{w}x{h}" for w, h in resolutions])
    log.info("FPS          : %d", args.fps)
    log.info("Frames       : %d (exact)", args.frames)
    log.info("Max sequences: %d", args.max_sequences)
    log.info("Workers      : %d", args.workers)
    log.info("Overwrite    : %s", args.overwrite)

    # --- Check FFmpeg ---
    check_ffmpeg()

    # --- Discover source MP4 files ---
    all_mp4_files = sorted([f for f in raw_dir.iterdir() if f.suffix.lower() == ".mp4"])
    if not all_mp4_files:
        log.error("No .mp4 files found in %s", raw_dir)
        sys.exit(1)

    log.info("Found %d MP4 files in %s", len(all_mp4_files), raw_dir)

    # --- Pre-filter: skip videos shorter than required frame count ---
    log.info("Probing source videos for frame count (need >= %d at %dfps)...", args.frames, args.fps)
    mp4_files = []
    too_short = []
    for mp4 in all_mp4_files:
        fc = probe_frame_count(str(mp4), args.fps)
        if fc < 0:
            log.warning("  Could not probe %s — including anyway", mp4.name)
            mp4_files.append(mp4)
        elif fc < args.frames:
            too_short.append((mp4, fc))
            log.info("  SKIP %s (%d frames < %d required)", mp4.name, fc, args.frames)
        else:
            mp4_files.append(mp4)

    if too_short:
        log.info("Skipped %d video(s) that are too short", len(too_short))

    if len(mp4_files) < args.max_sequences:
        log.error("Not enough raw videos: found %d eligible but need %d", len(mp4_files), args.max_sequences)
        sys.exit(1)

    if len(mp4_files) > args.max_sequences:
        log.info("Limiting to first %d of %d eligible videos", args.max_sequences, len(mp4_files))
        mp4_files = mp4_files[: args.max_sequences]

    log.info("%d video(s) selected for conversion", len(mp4_files))

    # --- Create output directories ---
    for w, h in resolutions:
        tc_dir = output_dir / resolution_folder(w, h, args.fps) / args.test_class
        tc_dir.mkdir(parents=True, exist_ok=True)

    # --- Build task list ---
    tasks = []
    for mp4 in mp4_files:
        stem = mp4.stem  # e.g. "024jYGFpByA_6_829to1172"
        for w, h in resolutions:
            yuv_name = output_yuv_name(stem, w, h, args.fps)
            out_path = output_dir / resolution_folder(w, h, args.fps) / args.test_class / yuv_name
            tasks.append((str(mp4), str(out_path), w, h))

    total = len(tasks)
    log.info("Total conversion tasks: %d (%d files x %d resolutions)", total, len(mp4_files), len(resolutions))

    # --- Dry run ---
    if args.dry_run:
        for mp4_path, out_path, w, h in tasks:
            exists = os.path.isfile(out_path) and os.path.getsize(out_path) > 0
            status = "EXISTS" if exists else "PENDING"
            log.info("[%s] %s -> %s", status, Path(mp4_path).name, Path(out_path).name)
        log.info("Dry run complete. No files were converted.")
        return

    # --- Convert in parallel ---
    t0 = time.time()
    converted = 0
    skipped = 0
    failed = 0
    failures = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        future_map = {}
        for mp4_path, out_path, w, h in tasks:
            future = pool.submit(
                convert_one,
                mp4_path,
                out_path,
                w,
                h,
                args.fps,
                args.frames,
                args.overwrite,
            )
            future_map[future] = (mp4_path, out_path)

        for i, future in enumerate(as_completed(future_map), 1):
            res = future.result()
            tag = f"[{i}/{total}]"
            if res["skipped"]:
                skipped += 1
                log.info("%s SKIP  %s (%s)", tag, Path(res["output"]).name, res["resolution"])
            elif res["success"]:
                converted += 1
                log.info("%s OK    %s (%s)", tag, Path(res["output"]).name, res["resolution"])
            else:
                failed += 1
                failures.append(res)
                log.error("%s FAIL  %s (%s): %s", tag, Path(res["output"]).name, res["resolution"], res["error"])

    elapsed = time.time() - t0

    # --- Generate JSON manifests ---
    log.info("Generating JSON manifests...")
    for w, h in resolutions:
        generate_manifest(
            output_dir,
            dataset_name,
            args.test_class,
            w,
            h,
            args.fps,
            args.frames,
            args.gop,
        )

    # --- Summary ---
    log.info("=" * 60)
    log.info("DONE in %.1fs", elapsed)
    log.info("  Converted : %d", converted)
    log.info("  Skipped   : %d (already existed)", skipped)
    log.info("  Failed    : %d", failed)
    log.info("=" * 60)

    if failures:
        log.warning("Failed conversions:")
        for f in failures:
            log.warning("  %s -> %s : %s", Path(f["input"]).name, Path(f["output"]).name, f["error"])
        sys.exit(1)


if __name__ == "__main__":
    main()
