import os
import cv2
import pytest
import numpy as np
from app.services.pipeline.video_validator import VideoValidator
from app.services.pipeline.frame_sampler import FrameSampler
from app.services.pipeline.motion_filter import MotionFilter
from app.services.pipeline.object_detector import ObjectDetector
from app.services.pipeline.multi_object_tracker import MultiObjectTracker
from app.services.pipeline.keyframe_extractor import KeyframeExtractor
from app.services.pipeline.interaction_detector import InteractionDetector
from app.services.pipeline.manifest_generator import ManifestGenerator
from datetime import datetime

@pytest.fixture
def sample_video_path(tmp_path):
    """Creates a temporary 30-frame synthetic MP4 video file for testing."""
    video_file = os.path.join(tmp_path, "test_cctv.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_file, fourcc, 10.0, (320, 240))

    for i in range(30):
        # Create black frame with moving white square (simulating moving person/object)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        x_pos = int(20 + i * 8)
        cv2.rectangle(frame, (x_pos, 80), (x_pos + 40, 160), (255, 255, 255), -1)
        out.write(frame)

    out.release()
    return video_file

def test_video_validator(sample_video_path):
    info = VideoValidator.validate_video(sample_video_path)
    assert info["valid"] is True
    assert info["width"] == 320
    assert info["height"] == 240
    assert info["frame_count"] == 30

def test_frame_sampler(sample_video_path):
    frames = list(FrameSampler.sample_frames(sample_video_path, target_fps=2.0))
    assert len(frames) > 0
    assert frames[0][0] == 0
    assert isinstance(frames[0][2], np.ndarray)

def test_motion_filter(sample_video_path):
    frames = list(FrameSampler.sample_frames(sample_video_path, target_fps=5.0))
    intervals = MotionFilter.analyze_motion(frames)
    assert len(intervals) > 0
    assert "activity_level" in intervals[0]

def test_object_detector_and_tracker(sample_video_path):
    frames = list(FrameSampler.sample_frames(sample_video_path, target_fps=5.0))
    detector = ObjectDetector(confidence_threshold=0.3)
    tracker = MultiObjectTracker(iou_threshold=0.2)

    all_dets = []
    for fn, ts, frame in frames:
        dets = detector.detect_frame(frame, fn, ts)
        tracked = tracker.process_frame_detections(dets)
        all_dets.extend(tracked)

    summaries = MultiObjectTracker.generate_track_summaries(all_dets)
    assert isinstance(all_dets, list)
    assert isinstance(summaries, list)

def test_manifest_generator(tmp_path):
    now = datetime.utcnow()
    manifest_dict, manifest_hash = ManifestGenerator.generate_manifest(
        evidence_id=1,
        analysis_job_id=10,
        source_sha256="abc123def456",
        started_at=now,
        completed_at=now,
        model_name="YOLOv8n",
        model_version="8.2.0",
        tracker_algorithm="ByteTrack",
        sampling_fps=2.0,
        confidence_threshold=0.4,
        stats={"test": 1},
        keyframes=[]
    )
    assert len(manifest_hash) == 64
    assert manifest_dict["evidence_id"] == 1
