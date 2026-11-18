# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import csv
import json
import os
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image


def write_panning_clip(path: Path, image: Image.Image, *, fps=10, seconds=8, width=640, height=360):
    import cv2

    frames = np.asarray(image)[..., ::-1]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"FFV1"), fps, (width, height))
    assert writer.isOpened(), "OpenCV FFV1 writer is unavailable"
    try:
        for index in range(fps * seconds):
            top = index % 40
            writer.write(np.ascontiguousarray(frames[top : top + height, index : index + width]))
    finally:
        writer.release()
    return str(path)


def test_simple_vqa_inference_matches_expected_score(tmp_path):
    pretrained_value = os.getenv("SIMPLE_VQA_PRETRAINED_PATH")
    if pretrained_value is None:
        pytest.skip("Set SIMPLE_VQA_PRETRAINED_PATH to run the real-weight SimpleVQA test")

    pretrained_path = Path(pretrained_value)
    assert (pretrained_path / "SimpleVQA").is_dir(), f"SimpleVQA assets not found: {pretrained_path}"

    from src.metrics.simple_vqa.scorer import SimpleVQA

    repository_root = Path(__file__).resolve().parents[2]
    image = Image.open(repository_root / "assets" / "frame_comparison_full-hevc_qsv.png").convert("RGB")
    # Eight seconds avoid upstream's minimum-clip padding, so these scores match its test_demo.py.
    h265_clip = write_panning_clip(tmp_path / "h265.avi", image.crop((3, 37, 722, 441)))
    mlvc_clip = write_panning_clip(tmp_path / "mlvc.avi", image.crop((743, 37, 1462, 441)))
    metric = SimpleVQA(str(pretrained_path), "cpu", {})

    import cv2

    scores = []
    for path in (h265_clip, mlvc_clip):
        capture = cv2.VideoCapture(path)
        stream = metric.start_video(
            total_frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            frame_rate=capture.get(cv2.CAP_PROP_FPS),
        )
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                stream.add_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            scores.append(stream.finish())
        finally:
            capture.release()
    scores = torch.tensor(scores)

    torch.testing.assert_close(
        scores,
        torch.tensor([2.275536060333252, 3.3751094341278076], dtype=scores.dtype),
        rtol=0,
        atol=1e-3,
    )


def stream_metric_stub(monkeypatch):
    from torchvision import transforms

    from src.metrics.simple_vqa.scorer import SimpleVQA

    metric = SimpleVQA.__new__(SimpleVQA)
    metric.device = "cpu"
    metric.spatial_transform = transforms.Compose(
        [transforms.Lambda(lambda image: torch.from_numpy(np.array(image)).permute(2, 0, 1).float())]
    )
    metric.motion_transform = metric.spatial_transform
    metric._extract_motion_feature = lambda frames: torch.stack([frame[0, 0, 0] for frame in frames])
    monkeypatch.setattr(metric, "model", lambda spatial, motion: spatial.mean() + motion.mean(), raising=False)
    return metric


def test_simple_vqa_loads_checkpoints_without_external_source(tmp_path, monkeypatch):
    from src.metrics.simple_vqa import scorer

    checkpoints = tmp_path / "SimpleVQA"
    checkpoints.mkdir()
    (checkpoints / "UGC_BVQA_model.pth").touch()
    (checkpoints / "SLOWFAST_8x8_R50.pyth").touch()
    monkeypatch.setattr(scorer, "resnet50", lambda: torch.nn.Linear(1, 1, bias=False))
    monkeypatch.setattr(scorer.torch, "load", lambda path, **kwargs: {"module.weight": torch.tensor([[0.25]])})
    monkeypatch.setattr(scorer, "_SlowFastFeatureExtractor", lambda path, device: torch.nn.Identity())

    metric = scorer.SimpleVQA(str(tmp_path), "cpu", {})

    torch.testing.assert_close(metric.model.state_dict()["weight"], torch.tensor([[0.25]]))
    assert not (checkpoints / "source").exists()


@pytest.mark.parametrize("fps,frames", [(10, 20), (15, 47), (30, 65)])
def test_simple_vqa_stream_matches_sampling_and_padding(fps, frames, monkeypatch):
    from src.metrics.simple_vqa.scorer import SimpleVQAStream

    metric = stream_metric_stub(monkeypatch)
    observed = []
    monkeypatch.setattr(
        metric, "model", lambda spatial, motion: observed.append((spatial.clone(), motion.clone())) or spatial.mean()
    )
    for offset in (0, 80):
        stream = SimpleVQAStream(metric, total_frames=frames, frame_rate=fps)
        for index in range(frames):
            stream.add_frame(np.full((4, 6, 3), index + offset, dtype=np.uint8))
        assert stream.finish() == pytest.approx(np.mean(np.arange(frames // fps) * fps) + offset)
        spatial, motion = observed[-1]
        torch.testing.assert_close(spatial[0, :, 0, 0, 0], torch.arange(frames // fps).float() * fps + offset)
        expected = np.minimum(np.arange(frames // fps)[:, None] * fps + np.arange(32), frames - 1) + offset
        torch.testing.assert_close(motion[0], torch.from_numpy(expected).float())
        assert not stream.active_motion and not stream.spatial_frames and not stream.completed_motion


def test_simple_vqa_stream_rejects_short_and_incomplete_videos(monkeypatch):
    from src.metrics.simple_vqa.scorer import SimpleVQAStream

    with pytest.raises(ValueError, match="at least one second"):
        SimpleVQAStream(stream_metric_stub(monkeypatch), total_frames=9, frame_rate=10)
    stream = SimpleVQAStream(stream_metric_stub(monkeypatch), total_frames=20, frame_rate=10)
    with pytest.raises(ValueError, match="uint8 RGB"):
        stream.add_frame(np.zeros((4, 6, 3), dtype=np.float32))
    stream.add_frame(np.zeros((4, 6, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="received only 1 frames"):
        stream.finish()
    assert not stream.active_motion and not stream.spatial_frames


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("is_yuv420", [False, True])
def test_simple_vqa_receives_each_reconstruction_without_decoder_folder(deterministic, is_yuv420, monkeypatch):
    from src.metrics.simple_vqa.scorer import SimpleVQAStream
    from src.transforms.functional import yuv_444_to_420
    from src.utils.encoder_tester import (
        EncoderTestParams,
        EncoderTestTask,
        EncoderTestTaskProcessor,
        EncoderTestTaskQPoint,
    )

    metric = stream_metric_stub(monkeypatch)

    tester = EncoderTestTaskProcessor.__new__(EncoderTestTaskProcessor)
    tester.task = EncoderTestTask(
        params=EncoderTestParams(is_yuv420=is_yuv420, calc_psnr=False, calc_simple_vqa=True),
        dataset_name="test",
        seq_name="clip",
        src_type="png",
        src_path="",
        src_width=6,
        src_height=4,
        n_frames=20,
        frame_rate=10,
        intra_period=1024,
        q_points=[EncoderTestTaskQPoint("a"), EncoderTestTaskQPoint("b")],
        decoder_format="mp4",
    )
    tester.rec_frame_writer_list = None
    tester._simple_vqa = metric
    tester._simple_vqa_streams = [
        SimpleVQAStream(metric, total_frames=20, frame_rate=10),
        SimpleVQAStream(metric, total_frames=20, frame_rate=10),
    ]
    tester._simple_vqa_errors = [None, None]
    before = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
        for index in range(20):
            frames = torch.stack([torch.full((3, 4, 6), (index + offset + 0.5) / 255) for offset in (0, 80)])
            if is_yuv420:
                frames[:, 1:] = 0.5
                image = yuv_444_to_420(frames[:1])
            else:
                image = frames[:1]
            tester._process_rec(frames, image)
            assert torch.are_deterministic_algorithms_enabled() == deterministic
            assert torch.is_deterministic_algorithms_warn_only_enabled()
        scores = []
        for stream in tester._simple_vqa_streams:
            assert stream is not None
            scores.append(stream.finish())
        assert scores == pytest.approx([20.328125, 180.328125])
    finally:
        torch.use_deterministic_algorithms(before, warn_only=warn_only)


def test_simple_vqa_error_is_logged_with_traceback(caplog):
    from src.utils.encoder_tester import (
        EncoderTestParams,
        EncoderTestTask,
        EncoderTestTaskProcessor,
        EncoderTestTaskQPoint,
    )

    tester = EncoderTestTaskProcessor.__new__(EncoderTestTaskProcessor)
    tester.task = EncoderTestTask(
        params=EncoderTestParams(is_yuv420=False, calc_psnr=False, calc_simple_vqa=True),
        dataset_name="test",
        seq_name="clip",
        src_type="png",
        src_path="",
        src_width=6,
        src_height=4,
        n_frames=20,
        frame_rate=10,
        intra_period=1024,
        q_points=[EncoderTestTaskQPoint("a")],
        decoder_format="mp4",
    )
    tester._simple_vqa_streams = [None]
    tester._simple_vqa_errors = [None]

    error = RuntimeError("scoring failed")
    try:
        raise error
    except RuntimeError as caught:
        tester._record_simple_vqa_error(0, caught)

    assert caplog.records[-1].exc_info == (RuntimeError, error, error.__traceback__)


def test_simple_vqa_benchmark_uses_existing_ffmpeg_reader(tmp_path, monkeypatch):
    from src.transforms.functional import yuv_420_to_444
    from src.utils import frame_reader_model_adapter as adapter
    from src.utils.encoder_tester import EncoderTestParams, EncoderTestTask, EncoderTestTaskQPoint

    y = np.full((1, 4, 6), 0.4, dtype=np.float32)
    uv = np.stack([np.full((2, 3), 0.25, dtype=np.float32), np.full((2, 3), 0.75, dtype=np.float32)])
    calls = []

    def reader(path, **kwargs):
        calls.append((path, kwargs))
        return iter([(y, uv)])

    monkeypatch.setattr(adapter, "FfmpegIterator", reader)
    monkeypatch.setattr(
        adapter.FrameReaderModelAdapter,
        "_get_stream_info",
        staticmethod(lambda path, return_bits: (6, 4, 1, torch.tensor([123.0]), 15.0)),
    )
    task = EncoderTestTask(
        params=EncoderTestParams(is_yuv420=True, calc_psnr=False, calc_simple_vqa=True),
        dataset_name="test",
        seq_name="clip",
        src_type="yuv",
        src_path="",
        src_width=6,
        src_height=4,
        n_frames=1,
        frame_rate=15,
        intra_period=1024,
        q_points=[EncoderTestTaskQPoint("0")],
        decoder_format="mp4",
    )
    factory = adapter.FrameReaderModelFactory(
        clip_folder=str(tmp_path), clip_format="mp4", metrics=None, device=torch.device("cpu")
    )
    model = factory(task)
    result = model(torch.zeros(1, 3, 4, 6))
    expected = yuv_420_to_444(
        (torch.from_numpy(y)[None], torch.from_numpy(uv[:1])[None], torch.from_numpy(uv[1:])[None]),
        mode="nearest",
    )
    torch.testing.assert_close(result["x_hat"], expected)
    torch.testing.assert_close(result["bits"], torch.tensor([123.0]))
    assert model.frame_rate == 15
    assert calls == [(str(tmp_path / "test" / "clip_qp0.mp4"), {"width": 6, "height": 4, "dst_format": "420"})]


@pytest.mark.parametrize(
    "frame_count,container_fps,max_n_frames,n_frames,expected_fps",
    [
        (60, 15, None, 60, 15),
        (60, 15, 1000, 60, 15),
        (60, 15, 30, 30, 15),
        (60, 15, None, 30, 15),
        (60, None, 30, 30, 30),
        (None, None, 30, 30, 30),
    ],
)
def test_simple_vqa_preserves_container_fps_for_partial_clips(
    monkeypatch, frame_count, container_fps, max_n_frames, n_frames, expected_fps
):
    from src.utils.encoder_tester import (
        EncoderTestParams,
        EncoderTestTask,
        EncoderTestTaskProcessor,
        EncoderTestTaskQPoint,
    )

    model = torch.nn.Linear(1, 1)
    monkeypatch.setattr(model, "frame_count", frame_count, raising=False)
    monkeypatch.setattr(model, "frame_rate", container_fps, raising=False)
    metric = Mock()
    metric.start_video.return_value.finish.return_value = 2.5
    task = EncoderTestTask(
        params=EncoderTestParams(is_yuv420=True, calc_psnr=False, calc_simple_vqa=True, max_n_frames=max_n_frames),
        dataset_name="test",
        seq_name="clip",
        src_type="yuv",
        src_path="",
        src_width=6,
        src_height=4,
        n_frames=n_frames,
        frame_rate=30,
        intra_period=1,
        q_points=[EncoderTestTaskQPoint("0")],
        decoder_format="mp4",
    )
    with EncoderTestTaskProcessor(
        task,
        i_frame_model=model,
        p_frame_model=None,
        auxiliary_models=Mock(simple_vqa_model=metric),
        read_one_frame=lambda: None,
    ) as processor:
        monkeypatch.setattr(processor, "_process_frame", Mock(return_value=None))
        monkeypatch.setattr(processor, "_generate_result", Mock(return_value={}))
        assert processor.run() == [{"ave_all_frame_simple_vqa": 2.5}]

    metric.start_video.assert_called_once_with(total_frames=n_frames, frame_rate=expected_fps)


@pytest.mark.parametrize("frame_rate", [None, 15])
def test_simple_vqa_benchmark_forwards_fallback_fps(monkeypatch, frame_rate):
    from compute_benchmark_metrics import ComputeBenchmarkMetricApp

    app = ComputeBenchmarkMetricApp()
    app._device = torch.device("cpu")
    monkeypatch.setattr(app, "resolve_path", lambda path, **kwargs: path)
    run = Mock(return_value={})
    monkeypatch.setattr(app, "_run_encoder_test", run)

    app.collect_metrics_on_testset(
        "set",
        {
            "calc_simple_vqa": True,
            "frame_rate": frame_rate,
            "max_n_frames": 30,
            "i_frame_q_index_list": [0],
            "clip_folder": "clips",
        },
    )

    run.assert_called_once()
    assert run.call_args.kwargs["params"].frame_rate == frame_rate


@pytest.fixture
def simple_vqa_rd(tmp_path):
    from compare_rd_video import RDTester

    paths = []
    for model, rate_scale in (("anchor", 1.0), ("candidate", 0.8)):
        data = {
            "set": {
                clip: {
                    str(rate): {
                        "i_frame_num": 1,
                        "p_frame_num": frames - 1,
                        **{f"ave_{scope}_frame_bpp": rate * rate_scale for scope in ("i", "p", "all")},
                        "ave_all_frame_simple_vqa": rate - 3 + offset,
                    }
                    for rate in (1, 2, 3)
                }
                for clip, frames, offset in (("a", 10, 0), ("b", 30, 1))
            }
        }
        path = tmp_path / f"{model}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        paths.extend((model, str(path)))
    return RDTester(
        Namespace(
            base_method="anchor",
            log_paths=paths,
            distortion_metrics=["simple_vqa"],
            compare_frame_type="default",
            compare_between="class",
            auto_test=0,
            rate_exclude=None,
            plot_rd_curve=0,
            plot_scheme=None,
            plot_path=None,
            output_path=str(tmp_path / "result.json"),
            output_plot_data=str(tmp_path / "points.csv"),
        )
    )


def test_simple_vqa_rd_exports_weighted_scores_and_bdrate(simple_vqa_rd, tmp_path):
    simple_vqa_rd.main()
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["simple_vqa"] == {"all": {"candidate": {"set": pytest.approx(-20)}}}
    with (tmp_path / "points.csv").open() as stream:
        points = list(csv.DictReader(stream))
    assert [float(point["candidate_simple_vqa"]) for point in points] == [-1.25, -0.25, 0.75]


@pytest.mark.parametrize(
    "labels",
    [
        ("4", "20", "36"),
        ("0i-4p", "16i-20p", "32i-36p"),
        ("bitrate_kbps=100.0", "bitrate_kbps=250.0", "bitrate_kbps=500.0"),
    ],
    ids=["different-qps", "different-ip-qps", "different-bitrate-targets"],
)
def test_simple_vqa_rd_accepts_independent_rate_labels(simple_vqa_rd, tmp_path, labels):
    path = tmp_path / "candidate.json"
    data = json.loads(path.read_text())
    for clip, points in data["set"].items():
        data["set"][clip] = dict(zip(labels, points.values()))
    path.write_text(json.dumps(data))

    simple_vqa_rd.main()

    result = json.loads((tmp_path / "result.json").read_text())
    assert result["simple_vqa"] == {"all": {"candidate": {"set": pytest.approx(-20)}}}


@pytest.mark.parametrize("longer_model", ["anchor", "candidate"])
def test_simple_vqa_rd_accepts_independent_rate_counts(simple_vqa_rd, tmp_path, longer_model):
    path = tmp_path / f"{longer_model}.json"
    data = json.loads(path.read_text())
    for points in data["set"].values():
        points["4"] = dict(points["3"])
        for scope in ("i", "p", "all"):
            points["4"][f"ave_{scope}_frame_bpp"] *= 4 / 3
        points["4"]["ave_all_frame_simple_vqa"] += 1
    path.write_text(json.dumps(data))

    simple_vqa_rd.main()

    result = json.loads((tmp_path / "result.json").read_text())
    assert np.isfinite(result["simple_vqa"]["all"]["candidate"]["set"])
    with (tmp_path / "points.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    for model in ("anchor", "candidate"):
        assert sum(row[f"{model}_bpp"] != "" for row in rows) == (4 if model == longer_model else 3)


@pytest.mark.parametrize("model", ["anchor", "candidate"])
def test_simple_vqa_rd_rejects_inconsistent_per_run_ladders(simple_vqa_rd, tmp_path, model):
    path = tmp_path / f"{model}.json"
    data = json.loads(path.read_text())
    points = data["set"]["b"]
    points["different-label"] = points.pop("2")
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match=f"same named rate ladder for every clip: {model}/set"):
        simple_vqa_rd.main()


@pytest.mark.parametrize("missing", ["dataset", "clip"])
def test_simple_vqa_rd_rejects_mismatched_populations(simple_vqa_rd, tmp_path, missing):
    path = tmp_path / "candidate.json"
    data = json.loads(path.read_text())
    if missing == "dataset":
        data["different-dataset"] = data.pop("set")
    else:
        del data["set"]["b"]
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="matching datasets and clips"):
        simple_vqa_rd.main()


@pytest.mark.parametrize("model", ["anchor", "candidate"])
def test_simple_vqa_rd_requires_three_points_per_model(simple_vqa_rd, tmp_path, model):
    path = tmp_path / f"{model}.json"
    data = json.loads(path.read_text())
    for points in data["set"].values():
        del points["3"]
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match=f"at least 3 rate points: {model}/set"):
        simple_vqa_rd.main()


def test_simple_vqa_rd_rejects_missing_scores(simple_vqa_rd, tmp_path):
    path = tmp_path / "candidate.json"
    data = json.loads(path.read_text())
    del data["set"]["a"]["2"]["ave_all_frame_simple_vqa"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Invalid ave_all_frame_simple_vqa"):
        simple_vqa_rd.main()
