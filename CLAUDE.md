# CLAUDE.md — GuardianEye Codebase Reference

> This file is the authoritative architecture reference for AI coding assistants (Claude Code) working in this repo. It is auto-loaded into every session. Keep it accurate.
>
> **Standing instruction for Claude:** After **every** code edit you make in this repository (any `Edit`, `Write`, or multi-file change), you MUST append a one-line entry to the **AI Change Log** at the bottom of this file in the same turn. Entry format:
> `- YYYY-MM-DD HH:MM — <files touched (repo-relative)> — <one-sentence summary of what and why>`
> Also update the relevant architecture section above if the edit changes module boundaries, routes, models, config keys, or data flow. If the change is purely cosmetic (whitespace, comments), still log it. Do not batch multiple edits into a single entry unless they share one atomic purpose.

---

## 1. Project Overview

**GuardianEye** is an AI-powered video surveillance, investigation, and digital-forensics platform. It ingests CCTV/video evidence, runs a computer-vision pipeline (detection, tracking, keyframes), produces vision-language descriptions, indexes them in a vector DB, and lets investigators run natural-language semantic search and agentic investigations over the evidence.

- Repo root: [phase-1/](./)
- Default branch: `main`
- Phase 1A = evidence ingestion + auth; Phase 1B = CV pipeline; Phase 1C = semantic search + VLM; latest additions = agentic investigation layer.

## 2. Technology Stack

**Backend** (Python, FastAPI)
- FastAPI + Uvicorn, async SQLAlchemy 2.x
- DB: SQLite (`aiosqlite`) locally (`backend/guardianeye.db`), Postgres (`asyncpg`) in Docker
- Auth: JWT (PyJWT), bcrypt via passlib
- CV: `ultralytics` (YOLO11n, YOLO11n-pose), OpenCV, PyAV
- VLM: HuggingFace `transformers` (BLIP, CLIP)
- Vector DB: ChromaDB (`backend/storage/chroma_db`)
- Embeddings: `sentence-transformers` (`all-MiniLM-L6-v2`)
- Agentic LLM: pluggable provider — Ollama (default), OpenAI, Groq, or pure heuristics

**Frontend** (Next.js 14, TypeScript)
- App Router, TailwindCSS, Lucide icons
- Axios API client, React Hook Form, Recharts

**Infra**
- `docker-compose.yml` boots Postgres + backend + frontend
- Local demo: `backend/scripts/mediamtx.exe` + `start_demo_cameras.ps1` serve demo RTSP streams

## 3. Repository Layout

```
phase-1/
├── backend/
│   ├── main.py                    # FastAPI entrypoint + lifespan (init_db) + CORS + /health
│   ├── requirements.txt
│   ├── alembic/                   # DB migrations (also auto-migrate in init_db)
│   ├── guardianeye.db             # Local SQLite DB (gitignored in intent)
│   ├── yolo11n.pt, yolo11n-pose.pt, yolov8n.pt   # Model weights
│   ├── scripts/                   # mediamtx RTSP server + demo camera launcher
│   ├── storage/
│   │   ├── evidence/              # Uploaded raw videos (SHA-256 keyed)
│   │   ├── derived/               # Keyframes, manifests, pipeline outputs
│   │   └── chroma_db/             # ChromaDB persistent vector store
│   ├── tests/                     # pytest suites (auth, pipeline, semantic, streams)
│   └── app/
│       ├── core/
│       │   ├── config.py          # Pydantic settings (env + defaults)
│       │   ├── logging_config.py
│       │   └── permissions.py     # Role-based access rules
│       ├── database/
│       │   ├── base.py            # SQLAlchemy DeclarativeBase
│       │   ├── session.py         # Async engine + SessionLocal + get_db
│       │   └── init_db.py         # create_all + auto schema migration + seed roles/users/demo
│       ├── authentication/
│       │   ├── jwt.py             # Token encode/decode
│       │   └── password.py        # bcrypt hashing
│       ├── models/                # SQLAlchemy ORM models
│       │   ├── user.py, role.py
│       │   ├── camera.py, alert.py, incident.py
│       │   ├── evidence.py        # Evidence + hashes + chain-of-custody
│       │   ├── analysis.py        # AnalysisJob, Detection, Track, Keyframe, ActivityInterval
│       │   ├── face.py            # FaceObservation (encrypted embeddings, job-scoped)
│       │   ├── observation.py     # VisualAttributeObservation, TrackAttributeAggregate,
│       │   │                      #   ObservationStatus/Source, EntityType
│       │   ├── semantic.py        # Semantic documents / index metadata
│       │   └── audit.py           # Audit log
│       ├── schemas/               # Pydantic request/response schemas (mirror models/)
│       ├── api/
│       │   ├── dependencies/
│       │   │   ├── auth.py        # get_current_user, require_role
│       │   │   └── ws_auth.py     # WebSocket auth
│       │   └── routes/
│       │       ├── __init__.py    # api_router aggregator
│       │       ├── auth.py        # /auth/login, /auth/me, token refresh
│       │       ├── evidence.py    # upload, list, download, delete, duplicate detection
│       │       ├── analysis.py    # trigger & inspect CV pipeline jobs
│       │       ├── semantic.py    # VLM indexing + hybrid semantic search
│       │       ├── investigation.py  # Agentic multi-step investigations
│       │       ├── admin.py       # Admin-only: system reset preview + execute
│       │       ├── faces.py       # Face search by photo, face list, authenticated face crops
│       │       ├── audit.py, camera.py, streams.py, alerts.py, incidents.py, health.py
│       ├── services/
│       │   ├── auth_service.py
│       │   ├── evidence_service.py        # SHA-256, duplicate check, storage layout
│       │   ├── storage_service.py         # Filesystem I/O for evidence/derived
│       │   ├── metadata_service.py        # Video metadata via PyAV/OpenCV
│       │   ├── audit_service.py
│       │   ├── camera_service.py
│       │   ├── analysis_jobs.py           # get_active_analysis_job(): THE job resolver — use it everywhere
│       │   ├── analysis_statistics.py     # Canonical AnalysisStatistics + DEFINITIONS (one meaning per counter)
│       │   ├── analysis_runner.py         # Orchestrates the CV pipeline per evidence; one run at a time, stages on worker threads
│       │   ├── system_reset_service.py    # Wipes all evidence + derived data, files, vector index
│       │   ├── face/                      # Face recognition
│       │   │   ├── engine.py              # InsightFace wrapper (detection+recognition only) + assess_quality
│       │   │   ├── crypto.py              # Fernet encryption of embeddings at rest
│       │   │   └── search_service.py      # embed_query_photo, search_faces, audit entry
│       │   ├── ai_interfaces/base_services.py   # ABCs for detectors/VLM/vector/LLM
│       │   ├── pipeline/                  # Phase 1B CV pipeline modules
│       │   │   ├── video_validator.py
│       │   │   ├── frame_sampler.py       # FPS-based sampling
│       │   │   ├── motion_filter.py       # Skip static frames
│       │   │   ├── object_detector.py     # YOLO + pose; IoU pose matching; calls attribute_extractor
│       │   │   ├── attribute_extractor.py # Per-detection person/vehicle attributes; carried-item association
│       │   │   ├── body_regions.py        # Pose-guided torso/leg/face regions; vehicle body band; pose↔box IoU match
│       │   │   ├── color_naming.py        # Deterministic HSV colour naming + person-specific skin exclusion
│       │   │   ├── clip_attributes.py     # CLIP zero-shot with cached text embeddings + shared image encode
│       │   │   ├── open_vocab_detector.py # CLIP-based open-vocab classification
│       │   │   ├── multi_object_tracker.py # IoU + centroid hybrid tracker
│       │   │   ├── interaction_detector.py # Spatial proximity / interactions
│       │   │   ├── keyframe_extractor.py  # Track entry/exit + anomaly frames
│       │   │   ├── alpr_service.py        # Per-track plate OCR (EasyOCR), format correction, char voting
│       │   │   ├── event_engine.py        # Rule-based event synthesis
│       │   │   ├── attribute_aggregator.py # Temporal attribute consensus + confidence decomposition
│       │   │   ├── face_extractor.py      # Per-person-track face extraction (stage 6c)
│       │   │   ├── capability_registry.py # Declares AVAILABLE / DEGRADED / NOT_AVAILABLE
│       │   │   └── manifest_generator.py  # Writes per-evidence manifest.json
│       │   ├── semantic/                  # Phase 1C semantic layer
│       │   │   ├── vlm_service.py         # BLIP captioning
│       │   │   ├── document_generator.py  # Builds searchable text docs per keyframe/track
│       │   │   ├── indexer.py             # Chunk + embed + upsert to Chroma
│       │   │   ├── vector_service.py      # ChromaDB client wrapper
│       │   │   ├── hybrid_search_engine.py # Structured answers first, then vector + lexical/metadata filtering
│       │   │   ├── structured_answers.py  # Plate / attribute questions answered from TrackAttributeAggregate
│       │   │   └── query_parser.py        # NL → structured query hints
│       │   └── agents/                    # Agentic investigation layer (intent-first)
│       │       ├── taxonomy.py            # Canonical entity map + normalize_entity_class
│       │       ├── analysis_job_resolver.py # thin wrapper over services/analysis_jobs.py
│       │       ├── llm_provider.py        # Ollama/OpenAI/Groq/Heuristics abstraction
│       │       ├── investigation_tools.py # count_entities, find_vehicles/persons, timelines (all job-scoped)
│       │       ├── investigation_orchestrator.py # Parses intent, resolves job, branches by intent
│       │       ├── evidence_verifier.py   # verify_finding + verify_count_result (deterministic)
│       │       └── report_generator.py    # Generic + specialized COUNT / FRAME_COUNT reports
│       ├── streaming/
│       │   ├── stream_manager.py          # Per-camera workers lifecycle
│       │   └── camera_worker.py           # RTSP read + fan-out frames
│       ├── websocket/                     # WS endpoints for live feeds / events
│       └── utils/
├── frontend/
│   ├── app/                       # Next.js App Router pages
│   │   ├── layout.tsx, page.tsx, globals.css
│   │   ├── login/, dashboard/, users/
│   │   ├── settings/              # Admin Danger Zone: reset all evidence
│   │   ├── evidence/              # list, [id] detail, upload
│   │   ├── forensic/              # semantic search workspace, search/
│   │   ├── realtime-monitoring/   # Live camera grid
│   │   ├── system-status/, audit/
│   │   └── api/video/[id]/        # Next route proxy for authenticated video playback
│   ├── components/
│   │   ├── DashboardLayout.tsx, ProtectedRoute.tsx
│   │   ├── CameraFeedCard.tsx
│   │   ├── AnalysisModal.tsx, AnalysisProgress.tsx
│   │   ├── InvestigationWorkspace.tsx     # Agentic investigation UI
│   │   ├── FaceSearchPanel.tsx            # "Face Search" tab on the evidence page
│   │   ├── AuthImage.tsx                  # Image fetched with the bearer token (protected crops)
│   │   ├── SemanticSearchWorkspace.tsx    # NL search UI + results
│   │   └── VideoOverlayPlayer.tsx         # Playback with detection/track overlays
│   ├── contexts/AuthContext.tsx
│   ├── services/                  # Axios clients (api, auth, evidence, analysis, semantic, audit, admin)
│   ├── types/                     # analysis.ts, audit.ts, evidence.ts
│   ├── next.config.mjs, tailwind.config.ts, tsconfig.json
│   └── package.json
├── docker-compose.yml             # Postgres + backend + frontend
├── README.md
└── CLAUDE.md                      # (this file)
```

## 4. Data Flow (End-to-End)

1. **Upload** — `POST /api/v1/evidence` → `evidence_service` computes SHA-256, dedups, writes to `storage/evidence/`, records `Evidence` row. Audit log entry.
2. **Trigger analysis** — `POST /api/v1/evidence/{evidence_id}/analysis` (or `/reprocess`, or automatically on upload) creates an `AnalysisJob` row and runs `analysis_runner` as an in-process background task.

   **Runner scheduling.** `run_analysis_job_async` registers the job in `running_analysis_jobs` immediately (so a reset is refused even while it is queued), then waits on `_analysis_slot` (`asyncio.Semaphore(1)`) — analyses run **one at a time**, because each already saturates every core and all sampled frames are held in memory. Every blocking stage goes through `run_stage(name, fn, ...)`, which runs it in `asyncio.to_thread` and adds its wall time to `job.stage_timings` (`validate`, `sample_frames`, `motion`, `detection` (incl. CLIP/pose attributes), `tracking`, `plates`, `faces`, `keyframes`, `interactions`, `events`, `manifest`, `semantic_index`). **Never call model inference, video decoding or file hashing directly on the event loop** — before this, the API froze for up to 65 s per analysis. VLM loading/captioning and the sentence-transformer load are likewise threaded. A job found `CANCELLED` when its turn comes is left untouched.

   **Analysis-job isolation — the core invariant.** An evidence file can be analysed many times; every derived row belongs to exactly one `(evidence_id, analysis_job_id)`, and track numbers are only unique *within* a job ("Track #5" in job 41 and job 42 are unrelated entities). Every reader resolves its job through `services/analysis_jobs.get_active_analysis_job(db, evidence_id, requested_job_id=None, strict=False)` — latest **COMPLETED** run (ties broken by id) unless a job is requested; with `strict=True` a wrong/unfinished job raises instead of silently falling back. Never write a query on Track/Detection/Keyframe/etc. that filters by `evidence_id` alone. `investigation_tools._scope_filter` deliberately matches *nothing* when the job id is None. Derived files are job-scoped too: keyframes are `keyframes/{evidence_id}/keyframe_ev{e}_job{j}_fn{f}.jpg`, sealed manifests are `manifests/ev{e}_job{j}.json`, Chroma collections are `ev{e}_job{j}`.

   **Statistics** come only from `analysis_statistics.compute_analysis_statistics(db, job)` (exposed at `GET /evidence/{id}/statistics`). Stored-entity counts are recomputed from the DB; run-time-only counters (`source_frames`, `processed_frames`, `raw_model_detections`, `confidence_filtered_detections`, `class_filtered_detections`, `events`) come from the job row and are **`None`, not 0**, for jobs with `stats_schema_version < STATS_SCHEMA_VERSION` (their counters were absent or meant something else). The detector runs YOLO at `DETECTOR_RAW_CONFIDENCE_FLOOR` and applies the job threshold in Python so every filtering stage is countable (`ObjectDetector.last_diagnostics`).

   **Manifests are sealed.** At completion the runner writes the exact hashed bytes to `manifests/ev{e}_job{j}.json`; `GET /manifest` loads that file and re-verifies its SHA-256 against `job.manifest_hash` (`source: "SEALED"`, `integrity_verified`). Older jobs get a rebuilt manifest flagged `source: "RECONSTRUCTED", integrity_verified: false` — a rebuild can never match the recorded hash.
3. **Phase 1B pipeline** (per `services/pipeline/`): validate video → sample frames → motion filter → YOLO detect → open-vocab classify → multi-object track → **extract visual attribute observations + temporally aggregate** → detect interactions → extract keyframes → ALPR → synthesize events → write manifest + persist Detections/Tracks/Keyframes/ActivityIntervals/Observations/Aggregates.

   **Evidence tiers — never conflate these.** `observation.py` defines a four-tier ladder enforced by `ObservationStatus`: `OBSERVED` (raw model output) → `CANDIDATE` (event engine proposed) → `SUPPORTED` (correlated with independent evidence) → `VERIFIED` (verification agent confirmed), plus `WITHHELD` (insufficient) and `CONTRADICTED`. There is deliberately **no `CONFIRMED` member** — the system reports observable patterns, never criminal responsibility.

   **How attributes are extracted (rewritten 2026-10-08 after real-footage failures).** The old path ran CLIP over fixed box slices with bare-word prompts ("a photo of a black") and no "none/not visible" option, so it *always* named a colour: 15 shirtless swimmers were all recorded as "brown shirt, white trousers" (skin and water). Now, in `attribute_extractor.py`:
   - **People:** YOLO-pose keypoints are matched to person boxes **by IoU** (`body_regions.match_pose_to_boxes`; the old index pairing attached one person's keypoints to another). Regions come from visible joints only — torso = shoulders→hips, lower = **knee→ankle** (long tops cover thighs), head from nose/eyes; a joint below 0.5 keypoint confidence means that region is *not visible* and produces **no claim** (reason recorded in `upper_garment_not_visible` / `lower_garment_not_visible`). Colour is named deterministically per pixel in HSV (`color_naming.read_color`, 12 names incl. `beige`) after removing skin pixels, where skin = generic YCrCb range **and** within Lab-chroma distance 9 of the person's own face tone (measured: shirtless torsos 3.6–7.6 from their face, a cream coat 12.4, dark jackets 23–28; lightness is ignored because wet/sunlit skin is much brighter). Torso ≥45% skin → `upper_garment_presence = absent` (no shirt colour); 25–45% → no claim either way. Garment type and headwear use CLIP with person-level prompts. People under 60 px tall get no attributes.
   - **Carried items** come only from YOLO-detected backpack/handbag/suitcase boxes whose centre lies inside the person box (`associate_carried_items`). CLIP was tried and reported handbags on people holding phones. No detected bag ⇒ no claim (never `carries_bag=false`).
   - **Vehicles:** colour = 0.6 × CLIP("a photo of a {colour} {class}") + 0.4 × pixel colour of the body band below the windscreen (`vehicle_body_region`); CLIP handles glass/road, pixels correct cases where a large graphic misleads CLIP (blue bus with a green logo). Body style via CLIP. Vehicles under 40 px wide get no attributes.
   - **Plates** (`alpr_service.read_vehicle_plates`, runner stage 6b) run **per vehicle track**, not per detection: up to 4 largest frames ≥0.5 s apart, vehicles ≥120 px wide, lower 65 % of the crop, CLAHE, EasyOCR with an A–Z0–9 allowlist (its text detector replaces the old contour plate locator), line-fragment merging, reads below OCR confidence 0.35 dropped, **format-guided correction** to the Indian pattern swapping only look-alike glyphs (O↔0, I↔1, B↔8, …, never inserting/deleting; valid reads are never rewritten), and early stop if the largest frame shows no text. `AttributeAggregator._aggregate_plate` votes **character by character**: asserted only with ≥2 same-length reads, a ≥0.6 confidence-weighted majority at every position, **and every character read the same way by ≥2 reads** (before 2026-10-08 one confident read could outvote one weaker read: `EA82545` vs `CA82545` was asserted as `CA82545`). The breakdown records `format_complete` / `format_issue`: a consistent read can still be partial or carry an impossible state prefix (`JA3K961` — `JA` is not an Indian state code); `INDIAN_STATE_CODES` only labels, never filters or rewrites OCR output. Measured on rendered plates: exact at ≥100 px plate width; 58–75 px plates previously produced wrong text at 0.08–0.25 confidence and now produce nothing.
   - **Make/model: NOT_AVAILABLE by measurement.** CLIP zero-shot make recognition was tested on the road footage and was wrong for 5/5 vehicles (Ford F-150 → "GMC" at 0.41, Toyota → "Kia"). It is not enabled; a fine-grained classifier plus close-range footage would be needed.
   - `clip_attributes.classify` caches text embeddings and shares one image encode across questions; verified identical to a full CLIP forward pass (max diff 1.8e-6).
   - **Cross-attribute consistency:** `AttributeAggregator._apply_cross_attribute_consistency` marks `upper_garment_color` / `upper_garment_type` as `CONTRADICTED` when the same track's `upper_garment_presence` consensus is `absent` (real case: presence 39/42 absent, yet a 3/3 "white" colour from misread frames). Search (`investigation_tools._ASSERTED_STATUSES`) only accepts `OBSERVED` / `SUPPORTED` / `VERIFIED` — `WITHHELD` and `CONTRADICTED` never match.
   - **Class-specific searches** ("find white cars") filter by detector class via the same taxonomy as `COUNT`, so "car" excludes a bus/truck-classed track; "vehicle" includes all.
   - **Body style** (`vehicle_body_style`) is CLIP zero-shot and *unvalidated* against labelled data; overhead angles bias it toward "SUV". Treat the detector class as the reliable vehicle type.

   **Attribute flow.** The extractor writes each value with `<attr>_confidence` and `<attr>_source` (`POSE_CROP_COLOR_MODEL`, `COLOR_HEURISTIC`, `COMBINED`, `CLIP_ZERO_SHOT`, `YOLO_DETECTOR`, `ALPR_OCR`), which become the observation's recorded source and model name. `AttributeAggregator.extract_observations` flattens these into per-frame `VisualAttributeObservation` rows with full provenance (evidence_id, analysis_job_id, track_number, frame_number, timestamp, source model + version, confidence). `AttributeAggregator.aggregate` then runs a confidence-weighted vote per (track, attribute) producing one `TrackAttributeAggregate` — **this is the table structured attribute search should query; a single frame is never authoritative.** A contested (<50% consensus) or weak (<0.35 confidence) aggregate is marked `WITHHELD` rather than asserted.

   **Batched CLIP.** CLIP is ~90% of the detection stage. `ObjectDetector.detect_batch` enriches each YOLO batch twice: first on throw-away dict copies inside `clip_attributes.recording()` (classify() only records the crops it is asked about and returns `{}`), then for real inside `clip_attributes.serving(clip_attributes.encode_batch(crops))`, where classify() reads features from one batched forward pass (content-keyed; a miss falls back to per-crop). The attribute logic in `attribute_extractor.py` is unaware of this — **keep enrichment deterministic and side-effect-free on the det dict it is given**, or the dry run and the real run will ask for different crops.

   **Confidence decomposition** (`compute_confidence_breakdown` / `overall_confidence`) is deterministic and documented — never LLM-assigned. Axes: `model`, `temporal`, `coverage`, `consensus_ratio`, `support`, `capability`. The `support` axis exists because one observation trivially has perfect consensus and coverage with *itself*; agreement axes earn full credit only at 3+ observations.

   **Capability honesty.** `capability_registry.py` declares each capability `AVAILABLE` / `DEGRADED` / `NOT_AVAILABLE` with a specific reason, and the registry snapshot is embedded in every analysis manifest. `face_recognition` is `AVAILABLE` when insightface + onnxruntime are installed and `FACE_RECOGNITION_ENABLED`. Currently `NOT_AVAILABLE`: `vehicle_make`, `vehicle_model`, `calibrated_speed_kmh`, `fire_smoke_detection`. `DEGRADED`: garment colour, vehicle colour, plate OCR, `vlm_captioning` (useful context, never authoritative). Call `unavailable_result()` rather than returning a guess or silently omitting the field — a caller must distinguish "we looked and could not determine" from "we never looked".
   **Face recognition (stage 6c, after tracking).** For each *person* track, `face_extractor.select_frames_for_track` picks the `FACE_MAX_FRAMES_PER_TRACK` largest person boxes ≥0.5 s apart; InsightFace (`buffalo_l`, loaded with `allowed_modules=["detection","recognition"]` — the bundled gender/age model is deliberately excluded) runs on the head region only. `assess_quality` gates on face size *in source pixels* (upscaling never makes a face identifiable), detection score and sharpness: `USABLE` faces get an ArcFace embedding encrypted with Fernet (`face/crypto.py`); `LOW_QUALITY` faces keep a crop but **no embedding** and are never compared. `FaceEngine.detect` runs only the detector; recognition is deferred to `DetectedFace.vector()`, so only the face actually kept (and only if `USABLE`) pays the ~0.26 s ArcFace cost — head crops often contain a neighbour's face too. The job's `face_stage` (`NOT_RUN` / `COMPLETED` / `UNAVAILABLE` / `FAILED`) is recorded so a search can say "faces were never extracted" instead of "no match"; a face-stage failure does not fail the analysis. **Search** (`POST /evidence/{id}/faces/search`, multipart `photo`) embeds the photo in memory (rejecting no face / unclear face / more than one clear face — silently picking one could search for the wrong person), compares by cosine similarity against the run's usable faces, and returns per-track `POSSIBLE_MATCH` candidates above `FACE_MATCH_THRESHOLD` with timestamps and crops, plus a coverage block (person tracks with usable / low-quality / no visible face). Overall statuses: `POSSIBLE_MATCH_FOUND`, `NO_MATCH_AMONG_USABLE_FACES` (message states how many people could not be checked), `NO_USABLE_FACES` ("neither confirmed nor ruled out"), `FACE_EXTRACTION_NOT_AVAILABLE`. The query photo is never stored — only its SHA-256 goes into the `FACE_SEARCH` / `FACE_SEARCH_REJECTED` audit entry. Probe on sample images: same person ≈0.92–0.97 cosine, different people ≈ −0.1…0.1.

4. **Phase 1C semantic indexing** (per `services/semantic/`): for each keyframe → BLIP caption → `document_generator` composes searchable text (caption + detected objects + track attributes + timestamp). TRACK documents take attributes from **asserted** `TrackAttributeAggregate` rows only (colour, garments, headwear, carried item, plate) — never WITHHELD/CONTRADICTED values, in text *or* metadata (the search engine's keyword check reads metadata too). Until 2026-10-08 the generator read `getattr(det, 'vehicle_color', ...)` etc. off `Detection` rows, which have no such columns, so no document ever contained a colour or plate → `indexer` embeds via sentence-transformers → upsert into a job-scoped ChromaDB collection.

   **VLM provenance.** `vlm_service` loads BLIP via `BlipProcessor`/`BlipForConditionalGeneration` directly (the `pipeline("image-to-text")` task was removed in transformers 5.x and had silently disabled the VLM). A `VLMObservation` + `VLM_OBSERVATION` document is written **only** for real model output. When the VLM is unavailable the keyframe gets a `SYSTEM_GENERATED` document titled "Detection Summary", prefixed `[Detection Summary — not a VLM description]`, `is_vlm: false` — never a VLM record. VLM output is scene context only; never authoritative for counts, identity, colour, plates or timing (BLIP has been observed calling buses "a train"). Captions are generated in batches of `CAPTION_BATCH_SIZE` (8) via `_caption_many`; a failing batch is retried image by image.
5. **Semantic search** — `POST /api/v1/semantic/search` → `query_parser` → `hybrid_search_engine` (vector similarity + metadata/time filters) → ranked keyframes. **Plate and attribute questions are answered structurally first** (`structured_answers.structured_search`): `PLATE_LOOKUP` → `list_vehicle_plates`; a given plate → exact, then fuzzy (`PARTIALLY_SUPPORTED`), then matching *unconfirmed* reads (`UNVERIFIED`, never an identification); colour/garment/carried-item → `find_tracks_by_attributes`. Embeddings cannot answer "what is the vehicle number". If an attribute lookup finds nothing, vector search still runs but drops TRACK documents, keeps only scene descriptions that mention every requested attribute, badges them `UNVERIFIED`, and the answer says no tracked entity has that consensus. A `track N` question returns that track's own TRACK document, not a caption that mentions it.
6. **Agentic investigation** — `POST /api/v1/investigation/query` is **intent-first** (not search-first). Flow:
   1. `QueryIntentParser.parse_query` emits a `structured_intent` (`COUNT` / `FRAME_COUNT` / `SEARCH` / `ATTRIBUTE_SEARCH` / `PLATE_LOOKUP` / `TEMPORAL_SEARCH` / `RELATION_SEARCH` / `EVENT_SEARCH` / `SUMMARY`) using deterministic rules; no LLM in the parse path. `PLATE_LOOKUP` = a plate is *asked for* ("vehicle number", "number plate", "licence plate", "registration"), optionally with a colour or `track N`; these phrases are stripped before count detection so "vehicle number of track 14" is not "number of" (COUNT). Plate text after a cue word is built only from plate-like fragments, so "plate of track 14" never becomes `OFTRACK14`.
   2. `analysis_job_resolver.resolve_analysis_job` locks the investigation to exactly ONE `analysis_job_id` (latest completed by default; caller may supply an explicit id). Every downstream query is scoped by `(evidence_id, analysis_job_id)` so prior analysis runs can't contaminate results.
   3. The orchestrator branches on intent:
      - **COUNT / FRAME_COUNT** → `InvestigationToolSystem.count_entities[_at_time]` runs deterministic `SELECT COUNT(DISTINCT track_number)` with canonical class filters from `services/agents/taxonomy.py`. **No** LLM, **no** semantic search, **no** EventEngine. Results verified by `EvidenceVerifier.verify_count_result` (checks job exists, no duplicate ids, aggregation is well-formed).
      - **EVENT_SEARCH** → job-scoped `EventEngine.detect_events` + `_dedup_events` to remove the "Sudden Acceleration ×3" style duplicates ((event_type, primary_track, secondary_track, rounded_timestamp) key, 0.5s tolerance).
      - **ATTRIBUTE_SEARCH** → `InvestigationToolSystem.find_tracks_by_attributes` filters `TrackAttributeAggregate` structurally (not by vector similarity). A track must satisfy **all** requested attributes; `WITHHELD` aggregates are excluded by default. Plate queries route to `find_vehicle_by_plate`, which is **exact by default** — fuzzy matching is opt-in and its hits are labelled `POSSIBLE_PLATE_MATCH`, never an identification. Plate queries and `PLATE_LOOKUP` go to `_handle_plate` (same `structured_search` as semantic search + `generate_plate_report`); every vehicle in scope is `READ` (all characters agreed by ≥2 reads), `UNCONFIRMED` (best guess + exact disputed characters) or `NOT_READ` (reason from `alpr_service.plate_unread_reason`, shared with `/entities`). Attribute-search finding cards come from `structured_search` too (raw matches stay in `result.matches`).
      - **SEARCH / TEMPORAL / RELATION / SUMMARY** → `HybridSearchEngine` + per-finding `EvidenceVerifier.verify_finding`.
   4. `ForensicReportGenerator` has specialized templates (`generate_count_report`, `generate_frame_count_report`) in addition to the generic forensic report. VLM per-frame counts are shown as supporting observations and are **never summed** — the authoritative total always comes from distinct tracks.
   **No LLM is called anywhere in the investigation path** — `LLMProvider` is imported by the orchestrator but unused since the intent-first rewrite; every report is a deterministic template filled from database results. (`MAX_AGENT_STEPS` / `INVESTIGATION_TIMEOUT_SECONDS` are currently unused.) The canonical entity taxonomy (`vehicle` ⊇ car/truck/bus/motorcycle/van) lives in `services/agents/taxonomy.py`; any new vehicle class is added there once.
7. **Live streaming** — `stream_manager` spawns a `camera_worker` per `Camera` row, pulling RTSP via OpenCV/PyAV. Frames are fanned out over WebSocket to the frontend `CameraFeedCard`.

## 5. Key Domain Models (`backend/app/models/`)

- `User` ↔ `Role` (Admin / Operator / Investigator / Viewer)
- `Evidence` — original file metadata, SHA-256, chain-of-custody timestamps
- `Camera` — code, name, location, `stream_url`, `status`
- `Alert`, `Incident` — operational events
- `AnalysisJob` — status + diagnostic counters (`frames_sampled`, `frames_with_detections`, `raw_detections`, `stored_detections`, `unique_tracks`, `unique_keyframes`, `activity_intervals_count`) + `stage_timings` (JSON seconds per stage, NULL for older runs; returned by `GET /evidence/analysis/{job_id}` and `/analysis-jobs`). These columns are auto-added by `init_db.auto_migrate_schema` on startup.
- `Detection`, `Track`, `Keyframe`, `ActivityInterval` (defined in `models/analysis.py`)
- Semantic metadata in `models/semantic.py`
- `AuditLog` in `models/audit.py`

## 6. Configuration (`backend/app/core/config.py`)

Pydantic `Settings` reads `backend/.env`. Notable keys:

- **Auth:** `SECRET_KEY`, `ALGORITHM`, `ACCESS_TOKEN_EXPIRE_MINUTES`
- **Storage:** `STORAGE_DIR`, `DERIVED_STORAGE_DIR`, `MAX_UPLOAD_SIZE_MB`, `ALLOWED_EXTENSIONS`
- **Pipeline:** `FRAME_SAMPLE_FPS`, `DETECTION_INTERVAL`, `KEYFRAME_INTERVAL`, `MAX_PROCESSING_RESOLUTION`, `YOLO_MODEL_NAME`, `YOLO_POSE_MODEL_NAME`, `DETECTION_CONFIDENCE_THRESHOLD`, `DETECTION_IOU_THRESHOLD`, `DETECTOR_RAW_CONFIDENCE_FLOOR` (model runs at this floor so pre-threshold counts are measurable; the job threshold is applied afterwards)
- **Attribute/ALPR:** `CLIP_MODEL_NAME`, `ENABLE_ATTRIBUTE_CLASSIFICATION`, `ENABLE_ALPR`
- **Face recognition:** `FACE_RECOGNITION_ENABLED`, `FACE_MODEL_PACK` (`buffalo_l`), `FACE_DET_SIZE`, `FACE_MAX_FRAMES_PER_TRACK`, `FACE_MIN_SIZE_PX`, `FACE_MIN_DET_SCORE`, `FACE_MIN_SHARPNESS`, `FACE_MATCH_THRESHOLD` (0.45), `FACE_QUERY_MAX_BYTES`, `FACE_EMBEDDING_KEY` (Fernet key; if empty it is derived from `SECRET_KEY` with a warning — set it explicitly outside development)
- **Semantic:** `VLM_ENABLED`, `VLM_MODEL_NAME`, `MAX_KEYFRAMES_FOR_VLM`, `EMBEDDING_MODEL_NAME`, `VECTOR_DB_TYPE`, `VECTOR_DB_PATH`, `MIN_SEARCH_RELEVANCE`, `LOITERING_THRESHOLD_SECONDS`, `CONCEALMENT_DISTANCE_THRESHOLD`
- **Agent/LLM:** `LLM_PROVIDER` (`ollama` default), `OLLAMA_BASE_URL`, `OLLAMA_MODEL_NAME`, `OPENAI_API_KEY`, `OPENAI_MODEL_NAME`, `GROQ_API_KEY`, `GROQ_MODEL_NAME`, `MAX_AGENT_STEPS`, `INVESTIGATION_TIMEOUT_SECONDS`
- **Streams:** `STREAM_TARGET_FPS`, `STREAM_IDLE_TIMEOUT`, `STREAM_MAX_BUFFER_SIZE`, `DEMO_STREAMS_ENABLED`, `DEMO_CAMERA_00X_SOURCE`
- **DB:** `DATABASE_URL` (preferred) or `POSTGRES_*` fields; falls back to SQLite only via `.env` override. Docker uses `postgresql+asyncpg://...`.
- **CORS:** `BACKEND_CORS_ORIGINS` (defaults to localhost:3000)

## 7. API Surface (prefix: `/api/v1`)

Mounted in [backend/app/api/routes/__init__.py](backend/app/api/routes/__init__.py):
- `auth` — login / me / refresh
- `evidence` — CRUD + upload + download
- `GET /evidence/{id}/entities?job_id=` — every tracked person/vehicle with aggregated attributes (value, confidence, status, frames agreeing/observed) and a `not_determined` map giving the reason for anything missing (region not visible, vehicle too small for OCR with its pixel width, no bag detected, make/model capability reason). Feeds the "People & Vehicles" tab (`components/EntitiesPanel.tsx`).
- `analysis` — kick off / inspect jobs. Every evidence-scoped read (`/detections`, `/tracks`, `/keyframes`, `/activity`, `/timeline`, `/manifest`, `/statistics`) accepts an optional `?job_id=`; without it the active (latest completed) run is used, and a job_id belonging to other evidence or not COMPLETED is a 404. `GET /evidence/{id}/analysis-jobs` lists every run and marks the active one. `POST /evidence/{id}/reprocess` enqueues a real new run (it used to be a fake that slept and flipped status to COMPLETED). `POST /evidence/{id}/reset-analysis` is **Admin only**, requires `{"confirmation": "RESET EVIDENCE <id>"}`, 409s while that evidence has a running job, and wipes all runs of that evidence (rows, keyframes, sealed manifests, Chroma collections) while keeping the evidence record and video.
- `semantic` — index / search
- `investigation` — agent runs
- `faces` — **Investigator/Operator/Admin only (`RequireInvestigator`); Viewers get 403.** `POST /evidence/{id}/faces/search` (multipart `photo`, optional form `threshold` 0.30–0.90, optional `?job_id=`), `GET /evidence/{id}/faces` (per person track: `USABLE_FACE` / `LOW_QUALITY` / `NO_FACE_VISIBLE` + crops; audited as `FACE_LIST_VIEWED`), `GET /evidence/{id}/faces/crops/{filename}` (authenticated, filename must match `face_ev{id}_job*_trk*_fn*.jpg`, `Cache-Control: private, no-store`). No response ever contains an embedding.
- `admin` — **Admin role only.** `GET /admin/reset/preview` (read-only counts of what would be deleted), `POST /admin/reset` with body `{"confirmation": "DELETE ALL EVIDENCE"}`. The reset wipes every evidence-derived table child-first, then the ChromaDB collections, then the *contents* of `STORAGE_DIR` and `DERIVED_STORAGE_DIR`. It **preserves** users, roles, cameras, alerts, incidents and the audit log, and writes a `SYSTEM_RESET` audit entry in the same transaction as the DB wipe. Returns 409 while any analysis job is running in-process.
- `audit`, `camera`, `streams`, `alerts`, `incidents`, `health`

OpenAPI docs: `http://localhost:8000/docs`.

## 8. Default Accounts (seeded by `init_db`)

- `admin / Admin123!` (role: Admin)
- `investigator / Investigator123!` (role: Investigator)

Mock cameras CAM-001..CAM-006 are seeded only when `DEMO_STREAMS_ENABLED=true`.

## 9. Developer Commands

**Backend**
```bash
cd backend
python -m venv venv && venv\Scripts\activate   # Windows
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pytest                                         # Runs backend/tests
```

**Frontend**
```bash
cd frontend
npm install
npm run dev        # http://localhost:3000
npm run build && npm run start
```

**Full stack via Docker**
```bash
docker compose up --build
```

**Demo RTSP cameras (Windows)**
```powershell
cd backend/scripts
./mediamtx.exe
./start_demo_cameras.ps1
```

## 10. Conventions

- Async everywhere on the backend — DB sessions come from `get_db` (`app/database/session.py`); never block the event loop (use `asyncio.to_thread` for heavy CPU/ML work; `analysis_runner` already handles this).
- Add new ORM models → import them in `app/database/init_db.py` so `Base.metadata.create_all` picks them up; add a Pydantic schema in `app/schemas/` mirroring the shape.
- New routes → create `app/api/routes/<name>.py` exposing `router`, then register it in `app/api/routes/__init__.py`.
- Pipeline modules should implement the abstract interfaces in `app/services/ai_interfaces/base_services.py` where applicable.
- Frontend API calls go through `services/api.ts` (Axios instance with JWT interceptor); never hit `fetch` directly in components.
- Protected pages wrap content in `components/ProtectedRoute.tsx`; auth state lives in `contexts/AuthContext.tsx`.
- Secrets (`SECRET_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY`, DB passwords) must come from env vars — never commit `.env`.

## 11. Known Gotchas

- The `analysis_jobs` table has diagnostic counter columns that are added at startup by `auto_migrate_schema`. If you add more counters, follow the same pattern in `init_db.py`.
- `init_db` is called in the FastAPI lifespan but wrapped in `try/except`; a failure logs a warning and the app still boots. Check logs when things seem seeded-but-empty.
- ChromaDB persists under `backend/storage/chroma_db/`. Deleting that directory wipes the vector index (DB rows survive) — reindex via the semantic routes.
- YOLO model weights (`yolo11n.pt`, `yolo11n-pose.pt`, `yolov8n.pt`) live at `backend/` root and are downloaded by Ultralytics on first run if missing.
- `LLM_PROVIDER=ollama` requires a local Ollama server at `OLLAMA_BASE_URL`; set to `heuristics` for offline / CI runs.
- **OCR engine:** `easyocr` is installed (declared in `requirements.txt`); `paddleocr` deliberately is not — see the change log for why. `capability_registry` resolves `license_plate_text` at runtime: `DEGRADED` with easyocr present, `NOT_AVAILABLE` if neither engine is importable. First ALPR use after a fresh install downloads ~100 MB of weights (~50 s).
- **Legacy VLM rows:** before the provenance fix, every `vlm_observations` row was template text (`model_name = 'VLM_Heuristic_Fallback'`). A startup migration relabels their search documents to `SYSTEM_GENERATED`, and statistics exclude them; re-run analysis to get real captions. First BLIP load downloads ~1 GB.
- **Tests are isolated by `tests/conftest.py` setting `DATABASE_URL`, `STORAGE_DIR`, `DERIVED_STORAGE_DIR` and `VECTOR_DB_PATH` to a temp dir *before* importing the app, and asserting it.** Do not remove or reorder this. Before it existed, `test_evidence_and_auth.py` (own client + autouse `init_db()`), `test_streams.py` (`TestClient` runs the lifespan) and the analysis runner (opens its own `SessionLocal`) all wrote to the developer's real `guardianeye.db` and `storage/` — creating junk `cctv_test_feed.mp4` evidence rows on every run. Evidence #3–#9 in the local DB are leftovers of that from September.
- **Startup marks QUEUED/PROCESSING jobs as FAILED** (`init_db.recover_interrupted_jobs`): analysis runs in-process, so nothing can be running when the process starts. This assumes a **single worker**; with multiple uvicorn/gunicorn workers it would fail other workers' live jobs (and `_analysis_slot` only serialises runs within one process).
- **Startup integrity migrations** (`init_db._INTEGRITY_MIGRATIONS`) run one statement per transaction (Postgres aborts the whole transaction on the first error, which previously meant later `ALTER TABLE`s were silently skipped there). They dedupe VLM rows and add unique indexes `uq_vlm_job_keyframe`, `uq_taa_job_track_attr`.
- **Open security finding:** `GET /evidence/{id}/derived/keyframes/{filename}` is unauthenticated (the UI loads keyframes via `<img src>`, which cannot send a bearer token). Needs signed URLs or cookie auth — not yet fixed.
- **Face recognition specifics:** `buffalo_l` pretrained weights are licensed for **non-commercial research only** (fine for this academic project, not for commercial deployment). First use downloads ~280 MB to `~/.insightface/models` (took ~7 min here). Changing `SECRET_KEY` without setting `FACE_EMBEDDING_KEY` makes every stored embedding undecryptable — search reports them as `undecryptable_embeddings` rather than crashing. Face *crops* on disk (`derived/faces/{evidence_id}/`) are not encrypted (keyframes already contain the same pixels) but are only served through the authenticated crops route. Runs analysed before this feature have `face_stage = NOT_RUN`; re-run analysis to enable face search on them. CPU cost: full-frame detection is several seconds, so only head regions of a few frames per person are processed.
- **Dead duplicate:** `services/semantic/vector_service.py` (`VectorService`) is a second, unused semantic system; only the 12 long-failing legacy tests in `test_audit_scenarios.py` / `test_phase1c_semantic.py` reference it. Do not build on it.
- Attribute observations only populate on a **new** analysis job. Jobs created before this feature landed have zero rows in `visual_attribute_observations` — re-run analysis to backfill.
- **SQLite FK cascades do not fire.** `session.py` never sets `PRAGMA foreign_keys=ON`, so every `ondelete="CASCADE"` in the models is inert locally. Deleting a parent row leaves its children orphaned. Any bulk delete must remove children explicitly — see `_WIPE_ORDER` in `system_reset_service.py`.
- **Role guards:** use `Depends(RequireAdmin)` (etc.) from `app/core/permissions.py`. `PermissionChecker.__call__` previously took `current_user` without `Depends(get_current_user)`, so it could never have worked as a route dependency; it was unused until the admin routes. Fixed.
- **IDs restart after a reset.** Tables use plain `INTEGER PRIMARY KEY` (no SQLite `AUTOINCREMENT`), so after a full reset new evidence starts at #1 again. Audit entries *before* the `SYSTEM_RESET` entry refer to pre-reset objects — the reset entry is the boundary.
- Tests that exercise the reset must monkeypatch `_storage_dirs`, `_list_vector_collections`, `_wipe_vector_store` and `_running_job_ids` and use an isolated engine — see `tests/test_system_reset.py`. Never let a test reach the real `storage/` or ChromaDB.
- **Tracker was evaluated, not replaced (2026-10-08).** On the two real videos at the default 2 fps, Ultralytics ByteTrack (incl. low-score boxes, per-class) stored only 22–35% of detections and produced *fewer* car tracks (10) than cars visible in one frame (18): objects move too far between samples for IoU-only matching, so new tracks never confirm. A Hungarian + constant-velocity + size-gated candidate cut suspected wrong merges on the road clip (18 → 3) but fragmented more; a 10 fps ByteTrack pseudo-ground-truth turned out to swap identities itself (6–8 box-widths per 0.1 s). With no trustworthy reference, `MultiObjectTracker` was kept. Known weaknesses: greedy matching, no motion model, centroid fallback gated at 0.35 of the *frame* (not object size), and it is passed `DETECTION_IOU_THRESHOLD` (the NMS setting) as its IoU gate — measured to make no difference. Track counts are therefore upper-bound-ish (road: 24 car tracks for a peak of 12 visible). Improving this needs hand-labelled identities first.
- Windows: shell is PowerShell by default; the Bash tool uses Git Bash (POSIX). Match syntax to the tool you invoke.

---

## AI Change Log

- 2026-10-08 19:06 — backend/app/services/semantic/structured_answers.py, tests/test_search_plates_and_attributes.py (new); services/semantic/{document_generator,hybrid_search_engine,query_parser}.py, services/agents/{investigation_tools,investigation_orchestrator,report_generator}.py, services/pipeline/{alpr_service,attribute_aggregator}.py, api/routes/analysis.py; CLAUDE.md — Plates and visual attributes now reach semantic search and the Investigation tab. Root cause: `document_generator` read colour/plate fields via getattr() on Detection rows, which have no such columns, so no search document ever contained a colour or plate; and semantic search never consulted TrackAttributeAggregate. TRACK documents now carry asserted consensus attributes only; new `PLATE_LOOKUP` intent ("vehicle number", "number plate", "registration", with colour/track filters) and `structured_answers` answer plate/attribute questions from the per-track consensus before vector search (READ / UNCONFIRMED with disputed characters / NOT_READ with reason), with a plate report in investigations and real finding cards for attribute search. Parser fixes: "vehicle number of track 14" was a COUNT; "plate of track 14" was read as plate OFTRACK14. Plate vote fix: every asserted character now needs ≥2 agreeing reads — on the user's job #2, CA82545 and 968B6 had been asserted from one-vs-one disagreements; plus a format/state-code label (JA3K961 is consistent but 'JA' is not a state). Search polish found on the real video: fuzzy plate hits no longer badged VERIFIED; attribute-miss fallback shows only scene descriptions mentioning the attribute (UNVERIFIED); 'track N' returns the track document. Verified by re-analysing the user's traffic video in an isolated DB (user data untouched) and running the queries; every fix mutation-tested. Existing runs need re-analysis to pick up the new vote and documents. 29 new tests; suite 228 → 257 passing (11 legacy VectorService failures unchanged).
- 2026-10-08 10:08 — backend/app/services/analysis_runner.py, services/pipeline/{frame_sampler,clip_attributes,object_detector,face_extractor}.py, services/face/{engine,search_service}.py, services/semantic/{vlm_service,indexer,hybrid_search_engine}.py, models/analysis.py, database/init_db.py, api/routes/analysis.py, tests/test_analysis_runner.py + tests/test_batched_inference.py (new), tests/test_face_recognition.py, tests/test_analysis_pipeline.py; frontend/components/AnalysisProgress.tsx, frontend/types/analysis.ts, frontend/app/evidence/[id]/page.tsx; CLAUDE.md — Efficiency pass, measured on the two real videos (road 10 s 1080p, river 23.5 s 1080p60) with an isolated harness. (1) Every pipeline stage ran synchronously on the event loop, so the API froze for up to 37.7 s / 65.6 s per analysis (94 s / 127 s frozen in total); stages now run via `run_stage` → `asyncio.to_thread`, as do BLIP loading/captioning and the sentence-transformer load: longest freeze now 0.5 s. Runs are serialised by `_analysis_slot`; queued runs count as running for reset; a job cancelled while queued stays cancelled. (2) New `analysis_jobs.stage_timings` JSON column (auto-migrated), returned by the job status/list routes and shown in AnalysisProgress. (3) Exact speedups, each verified against the old path on real footage: CLIP attribute crops batched via record/serve dry run (0 value differences over 10,950 fields, detection stage −35%); face recognition only on the kept USABLE face (all 45 faces field-identical, embedding diff 0.0, face stage −28%); BLIP captions batched (24/24 identical, ~1.6×); frame sampler grab()/retrieve() (byte-identical frames, −25–30%). Warm analysis 99.9 → 76.7 s (road), 137.0 → 103.4 s (river); track counts unchanged. (4) Tracker evaluated against ByteTrack and a Hungarian/motion candidate and deliberately NOT replaced — see the Known Gotchas entry. Plate OCR left unchanged (its cost is the validated 3× upscale). Repaired a test that called a non-existent `ObjectDetector.detect_frame` (failing at HEAD). Removed `lap`, which ultralytics auto-installed into the venv during the tracker evaluation. 12 new tests + 1 repaired; suite 215 → 228 passing (11 legacy VectorService failures unchanged).
- 2026-10-08 — backend/app/services/pipeline/attribute_aggregator.py, services/agents/investigation_tools.py, services/pipeline/capability_registry.py, api/routes/analysis.py, tests/test_visual_attributes.py, tests/test_attribute_search.py; frontend/components/EntitiesPanel.tsx; CLAUDE.md — Fixes found by re-analysing the real river/road videos: shirt colours are now CONTRADICTED when a track's consensus is "no upper garment" (and search only accepts asserted statuses), class-specific searches filter by detector class so a train labelled "bus" no longer answers "find the silver car", and body style is registered as an unvalidated DEGRADED capability. Verified on new runs #53/#55: "find people wearing white" on the river video now returns no supported evidence; "find the shirtless man" returns 14 tracks. Suite 211 → 215 passing.
- 2026-10-08 — backend/app/services/pipeline/{attribute_extractor,body_regions,color_naming,clip_attributes}.py, tests/test_visual_attributes.py (new); services/pipeline/object_detector.py, services/pipeline/alpr_service.py (rewritten), services/pipeline/attribute_aggregator.py, services/pipeline/capability_registry.py, services/analysis_runner.py, services/semantic/query_parser.py, api/routes/analysis.py, tests/test_attribute_observations.py (modified); frontend/components/EntitiesPanel.tsx (new), services/analysisService.ts, types/analysis.ts, app/evidence/[id]/page.tsx (modified); CLAUDE.md — Rebuilt person/vehicle attribute extraction after finding it fabricated attributes on real footage (15 shirtless swimmers recorded as "brown shirt, white trousers"; handbags reported on people holding phones; plates never read). Pose-guided regions with IoU pose matching, "not visible ⇒ no claim", person-specific skin exclusion and deterministic colour naming for garments; detector-grounded carried items; CLIP-with-object-prompts + body-pixel ensemble for vehicle colour; per-track plate OCR with format-guided correction and character voting; CLIP text-embedding cache. Vehicle make/model kept NOT_AVAILABLE after measuring CLIP at 0/5 on real vehicles. New `/entities` endpoint and People & Vehicles UI that shows every attribute with confidence and the reason for anything undetermined. Validated on real frames (river, road, two sample photos) and rendered plates; 50 new tests; suite 162 → 211 passing.
- 2026-10-08 — backend/app/models/face.py, services/face/{__init__,engine,crypto,search_service}.py, services/pipeline/face_extractor.py, api/routes/faces.py, tests/test_face_recognition.py (new); core/config.py, models/analysis.py, database/init_db.py, services/analysis_runner.py, services/analysis_statistics.py, services/system_reset_service.py, services/pipeline/capability_registry.py, api/routes/__init__.py, requirements.txt (modified); frontend/services/faceService.ts, components/AuthImage.tsx, components/FaceSearchPanel.tsx (new), app/evidence/[id]/page.tsx (modified); CLAUDE.md — Face recognition (search by photo). InsightFace buffalo_l, detection+recognition only (gender/age model excluded). Faces extracted from the best few frames per person track during analysis, quality-gated in source pixels, embeddings Fernet-encrypted at rest; low-quality faces are never embedded or compared. Search returns POSSIBLE_MATCH candidates with timestamps, crops and a coverage block so "no match" is never confused with "not visible"; query photos are never stored, only hashed into the audit log. Investigator+ only, crops served through an authenticated route, embeddings never in any response, faces included in both reset paths. Verified end-to-end with the real model on Ultralytics sample images in an isolated environment: each person matched the correct track (0.915 / 0.961), a stranger matched nothing, a two-face photo was rejected. 26 new tests; suite 136 → 162 passing.
- 2026-10-07 — CLAUDE.md — Corrected an inaccurate claim that the LLM provider is used for narrative polishing in the semantic branch; it is imported but never called anywhere in the investigation path.
- 2026-10-07 — backend/app/services/analysis_jobs.py, services/analysis_statistics.py, tests/test_phase1_isolation.py (new); services/agents/analysis_job_resolver.py, services/semantic/indexer.py, api/routes/semantic.py, api/routes/analysis.py, api/routes/evidence.py, services/evidence_service.py, services/agents/evidence_verifier.py, services/agents/investigation_tools.py, services/agents/investigation_orchestrator.py, services/semantic/hybrid_search_engine.py, services/semantic/vlm_service.py, services/system_reset_service.py, services/analysis_runner.py, services/pipeline/object_detector.py, services/pipeline/keyframe_extractor.py, services/pipeline/manifest_generator.py, services/pipeline/capability_registry.py, services/pipeline/event_engine.py, models/analysis.py, database/init_db.py, core/config.py, tests/conftest.py (modified); frontend/types/analysis.ts, frontend/app/evidence/[id]/page.tsx; CLAUDE.md — Phase 0 audit + Phase 1 (analysis-job isolation). Collapsed four competing "latest job" resolvers into `get_active_analysis_job` with `?job_id=` on every read route; fixed unscoped queries in EvidenceVerifier, HybridSearchEngine (incl. its legacy count path) and job-optional tools; job-scoped keyframe filenames; canonical AnalysisStatistics with real detector stage counts (None, not 0, for legacy runs); sealed + re-verified manifests (the UI hash could never match before); replaced the fake `/reprocess` pipeline with a real run; made `reset-analysis` admin-only, confirmed, running-job-aware and complete; stopped storing template text as VLM output and fixed BLIP loading for transformers 5.x; stopped fabricating `carries_bag=False`; startup recovery for stuck jobs and idempotent integrity migrations (per-statement transactions for Postgres); fixed EventEngine's missing cv2 import. Found and fixed test isolation: three test paths had been writing to the real DB and storage; conftest now redirects everything to a temp dir and asserts it, and the 4 junk evidence rows my test runs created today (#14–#17) were verified as fake payloads and removed with a TEST_ARTIFACT_REMOVED audit entry. 28 new tests; suite 108 → 136 passing.
- 2026-10-07 — backend/app/services/system_reset_service.py, app/api/routes/admin.py, tests/test_system_reset.py (new); app/api/routes/__init__.py, app/core/permissions.py, app/services/analysis_runner.py (modified); frontend/services/adminService.ts (new), frontend/app/settings/page.tsx, frontend/components/DashboardLayout.tsx (modified); CLAUDE.md — Added an admin-only "reset all evidence" feature for a fresh start. Wipes all evidence-derived tables child-first (SQLite FK cascades are inert here), the ChromaDB collections, and the contents of the evidence/derived storage dirs; preserves users, roles, cameras, alerts, incidents and the audit log, writing a SYSTEM_RESET audit entry atomically with the DB wipe. Requires the typed phrase "DELETE ALL EVIDENCE", returns 409 while an analysis job is running (new in-process `running_analysis_jobs` set, since crashed runs leave DB rows stuck at PROCESSING), and refuses to rmtree a filesystem root / home / cwd if STORAGE_DIR is misconfigured. Also fixed `PermissionChecker.__call__`, which lacked `Depends(get_current_user)` and so could never work as a route dependency. Settings placeholder replaced with a Danger Zone panel showing a live preview of what will be deleted; added Settings to the sidebar. 15 new tests, all sandboxed to temp dirs and an isolated DB; suite 93 → 108 passing. Corrected the stale "no OCR engine installed" gotcha.
<!--
Append newest entries at the TOP of this list. Format:
- YYYY-MM-DD HH:MM — <files touched> — <summary>
-->

- 2026-10-07 — backend/requirements.txt, backend venv — Added `easyocr>=1.7.0` as the ALPR engine and installed it. Chose easyocr over paddleocr deliberately: paddleocr's resolver would downgrade numpy 2.5.2 → 2.3.5 and opencv 5.0 → 4.10 plus add a third cv2 provider, which risks breaking torch 2.14 + ultralytics. `alpr_service` already tries Paddle then falls through to EasyOCR, so no code change was needed. Note `alpr_service.py` passes `show_log=` to PaddleOCR, which was removed in Paddle 3.x — another reason that path stays dead.
- 2026-10-07 — backend/app/services/semantic/query_parser.py, services/agents/investigation_tools.py, services/agents/investigation_orchestrator.py, services/agents/report_generator.py (modified); backend/tests/test_attribute_search.py (new), tests/test_investigation_intent.py (extended); CLAUDE.md — Slice B: structured attribute search. Parser now emits canonical TrackAttributeAggregate column names (`vehicle_color`, `upper_garment_color`) so tools need no translation, and extracts licence plates conservatively (letter+digit mix, 5–10 chars, stopword-filtered) so ordinary words are never read as plates. New tools `find_tracks_by_attributes`, `find_vehicle_by_plate`, `get_visual_attributes`. New ATTRIBUTE_SEARCH orchestrator branch + attribute report with a "Why this result?" section showing supporting frames and confidence decomposition. Verified on real job #42: "find people wearing black" → track #1 at 13/13 consensus; "find white cars" → 4 tracks ranked by confidence. 21 new tests; suite 72 → 93 passing.
- 2026-10-07 — backend/app/services/pipeline/capability_registry.py, backend/tests/test_attribute_observations.py, CLAUDE.md — Made capability states runtime-resolved rather than hardcoded: license_plate_text now downgrades to NOT_AVAILABLE when no OCR engine is importable or ENABLE_ALPR is off, and garment/vehicle colour downgrade when ENABLE_ATTRIBUTE_CLASSIFICATION is off. Found during real-video verification that this machine has neither paddleocr nor easyocr installed, so the previously declared DEGRADED state was overclaiming.
- 2026-10-07 — backend/app/models/observation.py, services/pipeline/attribute_aggregator.py, services/pipeline/capability_registry.py (new); services/pipeline/object_detector.py, services/analysis_runner.py, models/analysis.py, database/init_db.py (modified); backend/tests/test_attribute_observations.py (new); CLAUDE.md — Slice A: canonical evidence/observation model. Root cause fixed: the detector computed every visual attribute (garment + vehicle colours, headwear, carried items, plate text) and analysis_runner discarded all of them when building Detection rows, so structured attribute search had no data to query. Now persisted as VisualAttributeObservation with full provenance, temporally aggregated into TrackAttributeAggregate via confidence-weighted voting, with deterministic confidence decomposition and an ObservationStatus ladder that has no CONFIRMED member. Capability registry marks vehicle make/model, calibrated speed and fire/smoke as NOT_AVAILABLE rather than fabricating them. 26 new tests; suite went 46 → 72 passing with no new failures.
- 2026-10-07 — frontend/services/semanticService.ts, frontend/components/InvestigationWorkspace.tsx, CLAUDE.md — Rendered intent-first fields in the Investigation UI. Added InvestigationIntent/CountResult/InvestigationVerification types and a "Count Result" tab (headline total, breakdown-by-class table, counted track id chips, VERIFIED badge, methodology note). COUNT/FRAME_COUNT responses now auto-switch to this tab instead of showing "NO SUPPORTED EVIDENCE FOUND" when findings is empty.
- 2026-10-07 — backend/app/services/agents/taxonomy.py, analysis_job_resolver.py (new); investigation_tools.py, investigation_orchestrator.py, evidence_verifier.py, report_generator.py, semantic/query_parser.py, api/routes/investigation.py (modified); backend/tests/test_investigation_intent.py (new); backend/app/services/agents/__init__.py (emptied to break circular import); CLAUDE.md — Implemented intent-first investigation architecture: deterministic COUNT / FRAME_COUNT routing with analysis_job_id isolation, canonical entity taxonomy, specialized count reports, event deduplication; 16 new tests pass. Prior investigation route sent every query through semantic search + LLM + EventEngine, producing wrong aggregation and merging multiple analysis jobs.
- 2026-10-07 — CLAUDE.md — Initial creation: codebase architecture reference + standing instruction to append to this log after every AI edit.
