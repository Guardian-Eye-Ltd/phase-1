import logging
import asyncio
import time
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from app.database.session import SessionLocal
from app.models.analysis import (
    AnalysisJob, JobStatus, JobStage, ActivityInterval,
    FrameObservation, Detection, Track, Keyframe, PossibleInteraction
)
from app.models.evidence import Evidence, EvidenceStatus
from app.models.audit import AuditLog
from app.core.config import settings
from app.services.pipeline.video_validator import VideoValidator
from app.services.pipeline.frame_sampler import FrameSampler
from app.services.pipeline.motion_filter import MotionFilter
from app.services.pipeline.object_detector import ObjectDetector
from app.services.pipeline.multi_object_tracker import MultiObjectTracker
from app.services.pipeline.keyframe_extractor import KeyframeExtractor
from app.services.pipeline.interaction_detector import InteractionDetector
from app.services.pipeline.event_engine import EventEngine
from app.services.pipeline.manifest_generator import ManifestGenerator
from app.services.semantic.indexer import SemanticIndexer

logger = logging.getLogger(__name__)

# Active background jobs dict for cancellation support
active_job_cancellations = set()

async def run_analysis_job_async(job_id: int):
    """
    Background worker orchestrating the full Phase 1B computer vision pipeline
    and Phase 1C automatic evidence-grounded semantic indexing.
    """
    start_perf_time = time.time()

    async with SessionLocal() as db:
        result = await db.execute(select(AnalysisJob).where(AnalysisJob.id == job_id))
        job = result.scalar_one_or_none()
        if not job:
            logger.error(f"[VIDEO] Analysis job {job_id} not found.")
            return

        evidence_res = await db.execute(select(Evidence).where(Evidence.id == job.evidence_id))
        evidence = evidence_res.scalar_one_or_none()
        if not evidence:
            logger.error(f"[VIDEO] Evidence {job.evidence_id} not found for job {job_id}.")
            job.status = JobStatus.FAILED
            job.error_message = "Evidence record not found."
            await db.commit()
            return

        job.status = JobStatus.PROCESSING
        job.started_at = datetime.utcnow()
        job.progress = 5.0
        job.current_stage = JobStage.VALIDATING
        evidence.status = EvidenceStatus.PROCESSING

        # Clean up any existing records for this job ID to guarantee zero duplicates
        await db.execute(delete(Detection).where(Detection.analysis_job_id == job.id))
        await db.execute(delete(Track).where(Track.analysis_job_id == job.id))
        await db.execute(delete(Keyframe).where(Keyframe.analysis_job_id == job.id))
        await db.execute(delete(ActivityInterval).where(ActivityInterval.analysis_job_id == job.id))
        await db.execute(delete(PossibleInteraction).where(PossibleInteraction.analysis_job_id == job.id))
        await db.execute(delete(FrameObservation).where(FrameObservation.analysis_job_id == job.id))
        await db.commit()

        try:
            # Stage 1: Video Validation
            if job_id in active_job_cancellations:
                raise asyncio.CancelledError("Job cancelled by user.")

            logger.info(f"[VIDEO] Validating video evidence file: {evidence.file_path}")
            video_info = VideoValidator.validate_video(evidence.file_path)
            
            # Stage 2: Extracting Metadata
            job.current_stage = JobStage.EXTRACTING_METADATA
            job.progress = 15.0
            evidence.duration = video_info["duration"]
            evidence.resolution = f"{video_info['width']}x{video_info['height']}"
            evidence.fps = video_info["fps"]
            await db.commit()

            logger.info(
                f"[VIDEO] Metadata Extracted — Duration: {video_info['duration']}s, "
                f"FPS: {video_info['fps']}, Resolution: {video_info['width']}x{video_info['height']}, "
                f"Frames: {video_info['frame_count']}, Codec: {video_info.get('codec', 'N/A')}"
            )

            # Stage 3: Frame Sampling
            if job_id in active_job_cancellations:
                raise asyncio.CancelledError("Job cancelled by user.")

            job.current_stage = JobStage.SAMPLING_FRAMES
            job.progress = 25.0
            await db.commit()

            logger.info(f"[FRAME] Sampling video at {job.sampling_fps} FPS...")
            sampled_frames = list(FrameSampler.sample_frames(
                evidence.file_path, 
                target_fps=job.sampling_fps
            ))

            for fn, ts, frame in sampled_frames:
                frame_obs = FrameObservation(
                    analysis_job_id=job.id,
                    evidence_id=evidence.id,
                    frame_number=fn,
                    timestamp=ts
                )
                db.add(frame_obs)
            await db.commit()

            logger.info(f"[FRAME] Sampled {len(sampled_frames)} frames out of {video_info['frame_count']} total video frames.")

            # Stage 4: Motion Analysis
            if job_id in active_job_cancellations:
                raise asyncio.CancelledError("Job cancelled by user.")

            job.current_stage = JobStage.MOTION_ANALYSIS
            job.progress = 40.0
            await db.commit()

            motion_intervals = MotionFilter.analyze_motion(sampled_frames)
            for mi in motion_intervals:
                interval_obj = ActivityInterval(
                    analysis_job_id=job.id,
                    evidence_id=evidence.id,
                    start_time=mi["start_time"],
                    end_time=mi["end_time"],
                    start_frame=mi["start_frame"],
                    end_frame=mi["end_frame"],
                    activity_level=mi["activity_level"],
                    motion_score=mi["motion_score"]
                )
                db.add(interval_obj)
            await db.commit()

            # Stage 5: Object Detection
            if job_id in active_job_cancellations:
                raise asyncio.CancelledError("Job cancelled by user.")

            job.current_stage = JobStage.OBJECT_DETECTION
            job.progress = 55.0
            await db.commit()

            detector = ObjectDetector(
                model_name=job.model_name,
                confidence_threshold=job.confidence_threshold,
                iou_threshold=settings.DETECTION_IOU_THRESHOLD,
                max_processing_dim=settings.MAX_PROCESSING_RESOLUTION
            )

            raw_detections = detector.detect_batch(sampled_frames, batch_size=16)

            # Stage 6: Multi-Object Tracking
            if job_id in active_job_cancellations:
                raise asyncio.CancelledError("Job cancelled by user.")

            job.current_stage = JobStage.TRACKING
            job.progress = 70.0
            await db.commit()

            final_detections = raw_detections
            track_summaries = []

            if job.tracking_enabled and raw_detections:
                tracker = MultiObjectTracker(
                    iou_threshold=settings.DETECTION_IOU_THRESHOLD,
                    max_time_lost=3.0
                )
                
                # Group by frame and process chronologically
                by_frame = {}
                for d in raw_detections:
                    fn = d["frame_number"]
                    by_frame.setdefault(fn, []).append(d)

                final_detections = []
                for fn, frame_ts, _ in sampled_frames:
                    if fn in by_frame:
                        tracked_dets = tracker.process_frame_detections(by_frame[fn])
                        final_detections.extend(tracked_dets)

                track_summaries = MultiObjectTracker.generate_track_summaries(final_detections)

            # Save Detections & Tracks
            for d in final_detections:
                det_obj = Detection(
                    analysis_job_id=job.id,
                    evidence_id=evidence.id,
                    frame_number=d["frame_number"],
                    timestamp=d["timestamp"],
                    class_name=d["class_name"],
                    confidence=d["confidence"],
                    bbox_x1=d["bbox_x1"],
                    bbox_y1=d["bbox_y1"],
                    bbox_x2=d["bbox_x2"],
                    bbox_y2=d["bbox_y2"],
                    track_id=d.get("track_id")
                )
                db.add(det_obj)

            for trk in track_summaries:
                trk_obj = Track(
                    analysis_job_id=job.id,
                    evidence_id=evidence.id,
                    track_number=trk["track_number"],
                    class_name=trk["class_name"],
                    first_seen_timestamp=trk["first_seen_timestamp"],
                    last_seen_timestamp=trk["last_seen_timestamp"],
                    first_seen_frame=trk["first_seen_frame"],
                    last_seen_frame=trk["last_seen_frame"],
                    duration=trk["duration"],
                    observation_count=trk["observation_count"],
                    keyframe_count=0
                )
                db.add(trk_obj)
            await db.commit()

            # Stage 7: Keyframe Extraction & Spatial Interaction Detection
            if job_id in active_job_cancellations:
                raise asyncio.CancelledError("Job cancelled by user.")

            job.current_stage = JobStage.KEYFRAME_EXTRACTION
            job.progress = 85.0
            await db.commit()

            keyframes_data = KeyframeExtractor.extract_keyframes(
                sampled_frames,
                final_detections,
                track_summaries,
                evidence.id,
                settings.DERIVED_STORAGE_DIR
            )

            for k in keyframes_data:
                kf_obj = Keyframe(
                    analysis_job_id=job.id,
                    evidence_id=evidence.id,
                    frame_number=k["frame_number"],
                    timestamp=k["timestamp"],
                    selection_reason=k["selection_reason"],
                    image_path=k["image_path"],
                    sha256_hash=k["sha256_hash"],
                    track_ids=k["track_ids"],
                    detection_ids=k["detection_ids"]
                )
                db.add(kf_obj)

            interactions = InteractionDetector.detect_interactions(final_detections)
            for inter in interactions:
                inter_obj = PossibleInteraction(
                    analysis_job_id=job.id,
                    evidence_id=evidence.id,
                    entity_a_track_id=inter["entity_a_track_id"],
                    entity_b_track_id=inter["entity_b_track_id"],
                    start_time=inter["start_time"],
                    end_time=inter["end_time"],
                    min_distance_or_overlap=inter["min_distance_or_overlap"],
                    confidence_score=inter["confidence_score"],
                    label=inter["label"]
                )
                db.add(inter_obj)
            await db.commit()

            # Stage 8: Event Engine & Manifest Generation
            job.current_stage = JobStage.FINALIZING
            job.progress = 95.0
            await db.commit()

            # Run deterministic event engine for loitering, proximity, and carried objects
            detected_events = EventEngine.detect_events(track_summaries, final_detections)

            stats = {
                "total_frames_sampled": len(sampled_frames),
                "total_detections": len(final_detections),
                "total_tracks": len(track_summaries),
                "total_keyframes": len(keyframes_data),
                "total_events": len(detected_events),
                "total_interactions": len(interactions),
                "total_activity_intervals": len(motion_intervals)
            }

            manifest_dict, manifest_hash = ManifestGenerator.generate_manifest(
                evidence_id=evidence.id,
                analysis_job_id=job.id,
                source_sha256=evidence.sha256_hash,
                started_at=job.started_at,
                completed_at=datetime.utcnow(),
                model_name=job.model_name,
                model_version=job.model_version,
                tracker_algorithm=job.tracker_algorithm,
                sampling_fps=job.sampling_fps,
                confidence_threshold=job.confidence_threshold,
                stats=stats,
                keyframes=keyframes_data,
                detections=final_detections,
                tracks=track_summaries,
                events=detected_events
            )

            job.manifest_hash = manifest_hash
            job.status = JobStatus.COMPLETED
            job.progress = 100.0
            job.completed_at = datetime.utcnow()
            evidence.status = EvidenceStatus.COMPLETED

            # Log audit record
            audit_log = AuditLog(
                user_id=evidence.uploaded_by,
                action="ANALYSIS_COMPLETED",
                resource_type="ANALYSIS_JOB",
                resource_id=str(job.id),
                metadata_json=f"Analysis completed for evidence '{evidence.original_filename}'. Hash: {manifest_hash[:16]}..."
            )
            db.add(audit_log)
            await db.commit()

            total_proc_time = round(time.time() - start_perf_time, 2)
            avg_proc_fps = round(video_info['frame_count'] / max(total_proc_time, 0.01), 2)

            logger.info(
                f"[VIDEO] Processing Performance Summary — Video: {evidence.original_filename} (ID: {evidence.id}) | "
                f"Duration: {video_info['duration']}s | FPS: {video_info['fps']} | Total Frames: {video_info['frame_count']} | "
                f"Sampled Frames: {len(sampled_frames)} | Detections: {len(final_detections)} | Tracks: {len(track_summaries)} | "
                f"Keyframes: {len(keyframes_data)} | Processing Time: {total_proc_time}s | Processing Speed: {avg_proc_fps} FPS"
            )

            # Automatic Evidence-Grounded Semantic Indexing
            logger.info(f"[EMBEDDING] Auto-triggering semantic indexing for evidence {evidence.id}, job {job.id}...")
            await SemanticIndexer.index_evidence_async(evidence.id, job.id)

        except asyncio.CancelledError:
            logger.warning(f"[VIDEO] Analysis job {job_id} was cancelled.")
            job.status = JobStatus.CANCELLED
            job.error_message = "Analysis cancelled by investigator."
            evidence.status = EvidenceStatus.FAILED
            await db.commit()

        except Exception as e:
            logger.exception(f"[VIDEO] Error executing analysis job {job_id}: {e}")
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            evidence.status = EvidenceStatus.FAILED
            
            audit_log = AuditLog(
                user_id=evidence.uploaded_by,
                action="ANALYSIS_FAILED",
                resource_type="ANALYSIS_JOB",
                resource_id=str(job.id),
                metadata_json=f"Analysis failed for evidence '{evidence.original_filename}': {str(e)}"
            )
            db.add(audit_log)
            await db.commit()

        finally:
            active_job_cancellations.discard(job_id)
