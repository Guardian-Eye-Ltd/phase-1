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
from app.models.observation import (
    VisualAttributeObservation, TrackAttributeAggregate,
    ObservationStatus, ObservationSource, EntityType
)
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
from app.services.pipeline.attribute_aggregator import AttributeAggregator
from app.services.pipeline.face_extractor import extract_faces
from app.services.pipeline.alpr_service import read_vehicle_plates
from app.services.face.engine import FaceEngine
from app.models.face import FaceObservation
from app.services.analysis_statistics import STATS_SCHEMA_VERSION, compute_analysis_statistics
from app.services.pipeline.capability_registry import capability_report, get_capability
from app.services.semantic.indexer import SemanticIndexer

logger = logging.getLogger(__name__)

# Active background jobs dict for cancellation support
active_job_cancellations = set()

# Jobs actually executing in this process. Tracked in memory rather than via the
# DB status column, because a crashed run leaves a row stuck at PROCESSING forever.
running_analysis_jobs: set = set()

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
        await db.execute(delete(VisualAttributeObservation).where(VisualAttributeObservation.analysis_job_id == job.id))
        await db.execute(delete(TrackAttributeAggregate).where(TrackAttributeAggregate.analysis_job_id == job.id))
        await db.execute(delete(FaceObservation).where(FaceObservation.analysis_job_id == job.id))
        await db.commit()

        running_analysis_jobs.add(job_id)
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

            # Batch insert frame observations
            db.add_all([
                FrameObservation(
                    analysis_job_id=job.id,
                    evidence_id=evidence.id,
                    frame_number=fn,
                    timestamp=ts
                )
                for fn, ts, _ in sampled_frames
            ])
            await db.flush()

            logger.info(f"[FRAME] Sampled {len(sampled_frames)} frames out of {video_info['frame_count']} total video frames.")

            # Stage 4: Motion Analysis
            if job_id in active_job_cancellations:
                raise asyncio.CancelledError("Job cancelled by user.")

            job.current_stage = JobStage.MOTION_ANALYSIS
            job.progress = 40.0
            await db.commit()

            motion_intervals = MotionFilter.analyze_motion(sampled_frames)
            db.add_all([
                ActivityInterval(
                    analysis_job_id=job.id,
                    evidence_id=evidence.id,
                    start_time=mi["start_time"],
                    end_time=mi["end_time"],
                    start_frame=mi["start_frame"],
                    end_frame=mi["end_frame"],
                    activity_level=mi["activity_level"],
                    motion_score=mi["motion_score"]
                )
                for mi in motion_intervals
            ])
            await db.flush()

            # Stage 5: Object Detection
            if job_id in active_job_cancellations:
                raise asyncio.CancelledError("Job cancelled by user.")

            job.current_stage = JobStage.OBJECT_DETECTION
            job.progress = 55.0
            await db.commit()

            detector = ObjectDetector(
                model_name=settings.YOLO_MODEL_NAME,
                confidence_threshold=job.confidence_threshold,
                iou_threshold=settings.DETECTION_IOU_THRESHOLD,
                max_processing_dim=settings.MAX_PROCESSING_RESOLUTION
            )

            raw_detections = detector.detect_batch(sampled_frames, batch_size=16)
            # Count how many distinct frames had at least one detection
            frames_with_dets = len(set(d["frame_number"] for d in raw_detections))
            raw_detection_count = len(raw_detections)
            logger.info(
                f"[DETECTION] job={job_id} raw_detections={raw_detection_count} "
                f"frames_with_detections={frames_with_dets}"
            )

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

            # Batch Save Detections & Tracks
            db.add_all([
                Detection(
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
                for d in final_detections
            ])

            db.add_all([
                Track(
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
                for trk in track_summaries
            ])
            await db.flush()

            # Stage 6b: Visual Attribute Observations & Temporal Aggregation
            # The detector already computed garment/vehicle colors, headwear,
            # carried items and plate reads. Persist them with provenance so
            # structured attribute search has something to query.
            attr_observations = AttributeAggregator.extract_observations(final_detections)

            caps = capability_report()
            capability_states = {name: c["state"] for name, c in caps.items()}

            # Licence plates: read once per vehicle track on its best frames.
            if caps.get("license_plate_text", {}).get("state") != "NOT_AVAILABLE":
                attr_observations += await asyncio.to_thread(
                    read_vehicle_plates, sampled_frames, final_detections
                )

            _MODEL_BY_SOURCE = {
                "CLIP_ZERO_SHOT": settings.CLIP_MODEL_NAME,
                "POSE_CROP_COLOR_MODEL": f"{settings.YOLO_POSE_MODEL_NAME}+hsv-colour-naming",
                "COLOR_HEURISTIC": "hsv-colour-naming",
                "COMBINED": f"{settings.CLIP_MODEL_NAME}+hsv-colour-naming",
                "ALPR_OCR": "easyocr",
            }

            def _source(obs):
                try:
                    return ObservationSource(obs.get("source", "CLIP_ZERO_SHOT"))
                except ValueError:
                    return ObservationSource.CLIP_ZERO_SHOT

            db.add_all([
                VisualAttributeObservation(
                    evidence_id=evidence.id,
                    analysis_job_id=job.id,
                    track_number=obs["track_number"],
                    frame_number=obs["frame_number"],
                    timestamp=obs["timestamp"],
                    entity_type=EntityType(obs["entity_type"]),
                    attribute=obs["attribute"],
                    value=obs["value"],
                    confidence=obs["confidence"],
                    source=_source(obs),
                    model_name=_MODEL_BY_SOURCE.get(_source(obs).value, "unknown"),
                    model_version="2.0",
                    status=ObservationStatus.OBSERVED,
                )
                for obs in attr_observations
            ])

            track_obs_counts = {t["track_number"]: t["observation_count"] for t in track_summaries}
            aggregates = AttributeAggregator.aggregate(
                attr_observations,
                track_observation_counts=track_obs_counts,
                capability_states=capability_states,
            )

            db.add_all([
                TrackAttributeAggregate(
                    evidence_id=evidence.id,
                    analysis_job_id=job.id,
                    track_number=agg["track_number"],
                    entity_type=EntityType(agg["entity_type"]),
                    attribute=agg["attribute"],
                    value=agg["value"],
                    confidence=agg["confidence"],
                    observation_count=agg["observation_count"],
                    supporting_count=agg["supporting_count"],
                    dissenting_count=agg["dissenting_count"],
                    first_observed_at=agg["first_observed_at"],
                    last_observed_at=agg["last_observed_at"],
                    source=ObservationSource.COMBINED,
                    status=ObservationStatus(agg["status"]),
                    confidence_breakdown=agg["confidence_breakdown"],
                )
                for agg in aggregates
            ])
            await db.flush()

            vehicle_attr_count = sum(1 for o in attr_observations if o["entity_type"] == "VEHICLE")
            plate_obs_count = sum(1 for o in attr_observations if o["attribute"] == "license_plate_text")
            withheld_count = sum(1 for a in aggregates if a["status"] == "WITHHELD")

            logger.info(
                "[ATTRIBUTES] job=%s observations=%d (vehicle=%d plate=%d) "
                "aggregates=%d withheld=%d",
                job_id, len(attr_observations), vehicle_attr_count,
                plate_obs_count, len(aggregates), withheld_count,
            )

            # Stage 6c: Face extraction from person tracks. A failure here is
            # recorded on the job (face_stage) rather than failing the analysis,
            # so face search can say exactly why it has nothing to compare.
            face_cap = get_capability("face_recognition")
            if face_cap["state"] == "NOT_AVAILABLE":
                job.face_stage, job.face_stage_detail = "UNAVAILABLE", face_cap["reason"]
            elif not await asyncio.to_thread(FaceEngine.load):
                job.face_stage, job.face_stage_detail = "UNAVAILABLE", FaceEngine.load_error()
            else:
                try:
                    face_obs = await asyncio.to_thread(
                        extract_faces, sampled_frames, final_detections,
                        evidence.id, job.id, settings.DERIVED_STORAGE_DIR,
                    )
                    db.add_all([
                        FaceObservation(
                            evidence_id=evidence.id,
                            analysis_job_id=job.id,
                            track_number=o["track_number"],
                            frame_number=o["frame_number"],
                            timestamp=o["timestamp"],
                            bbox_x1=o["bbox"][0], bbox_y1=o["bbox"][1],
                            bbox_x2=o["bbox"][2], bbox_y2=o["bbox"][3],
                            face_width_px=o["face_width_px"],
                            face_height_px=o["face_height_px"],
                            det_score=o["det_score"],
                            sharpness=o["sharpness"],
                            quality_status=o["quality_status"],
                            quality_reasons=o["quality_reasons"],
                            embedding_encrypted=o["embedding_encrypted"],
                            embedding_model=o["embedding_model"],
                            embedding_dim=o["embedding_dim"],
                            crop_filename=o["crop_filename"],
                            crop_sha256=o["crop_sha256"],
                        )
                        for o in face_obs
                    ])
                    await db.flush()
                    usable = sum(1 for o in face_obs if o["quality_status"] == "USABLE")
                    job.face_stage = "COMPLETED"
                    job.face_stage_detail = f"{len(face_obs)} face(s) extracted, {usable} usable for comparison."
                except Exception as e:
                    logger.exception(f"[FACE] Face extraction failed for job {job_id}: {e}")
                    job.face_stage, job.face_stage_detail = "FAILED", f"{type(e).__name__}: {e}"

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
                settings.DERIVED_STORAGE_DIR,
                analysis_job_id=job.id,
            )

            db.add_all([
                Keyframe(
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
                for k in keyframes_data
            ])

            interactions = InteractionDetector.detect_interactions(final_detections)
            db.add_all([
                PossibleInteraction(
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
                for inter in interactions
            ])
            await db.commit()

            # Stage 8: Event Engine & Manifest Generation
            job.current_stage = JobStage.FINALIZING
            job.progress = 95.0
            await db.commit()

            # Run deterministic event engine for loitering, proximity, and carried objects
            detected_events = EventEngine.detect_events(track_summaries, final_detections)

            # Run-time counters (only knowable now). Stored-entity counts are
            # recomputed from the DB by compute_analysis_statistics.
            diag = detector.last_diagnostics
            job.source_frames = video_info["frame_count"]
            job.processed_frames = diag.get("processed_frames", 0)
            job.raw_detections = diag.get("raw_model_detections", 0)
            job.confidence_filtered_detections = diag.get("confidence_filtered_detections", 0)
            job.class_filtered_detections = diag.get("class_filtered_detections", 0)
            job.event_candidates = len(detected_events)
            job.stats_schema_version = STATS_SCHEMA_VERSION
            job.tracker_algorithm = "IoU-Centroid-Custom"
            job.model_name = settings.YOLO_MODEL_NAME

            # Legacy summary columns, kept in sync for existing readers.
            job.frames_sampled = len(sampled_frames)
            job.frames_with_detections = frames_with_dets
            job.stored_detections = len(final_detections)
            job.unique_tracks = len(track_summaries)
            job.unique_keyframes = len(keyframes_data)
            job.activity_intervals_count = len(motion_intervals)
            job.visual_attribute_observations = len(attr_observations)
            job.vehicle_attribute_observations = vehicle_attr_count
            job.plate_observations = plate_obs_count
            job.track_attribute_aggregates = len(aggregates)
            job.withheld_attributes = withheld_count
            await db.commit()

            canonical_stats = (await compute_analysis_statistics(db, job)).to_dict()
            canonical_stats.pop("definitions", None)
            logger.info(f"[PIPELINE] job={job_id} statistics={canonical_stats}")

            # One timestamp for both the manifest and the job row, so the stored
            # manifest and the job record describe the same moment.
            finished_at = datetime.utcnow()
            manifest_dict, manifest_hash = ManifestGenerator.generate_manifest(
                evidence_id=evidence.id,
                analysis_job_id=job.id,
                source_sha256=evidence.sha256_hash,
                started_at=job.started_at,
                completed_at=finished_at,
                model_name=job.model_name,
                model_version=job.model_version,
                tracker_algorithm=job.tracker_algorithm,
                sampling_fps=job.sampling_fps,
                confidence_threshold=job.confidence_threshold,
                stats=canonical_stats,
                keyframes=keyframes_data,
                detections=final_detections,
                tracks=track_summaries,
                events=detected_events,
                capabilities=caps,
            )
            ManifestGenerator.save_manifest(
                ManifestGenerator.manifest_path(evidence.id, job.id), manifest_dict
            )

            job.manifest_hash = manifest_hash
            job.status = JobStatus.COMPLETED
            job.progress = 100.0
            job.completed_at = finished_at
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
            running_analysis_jobs.discard(job_id)
