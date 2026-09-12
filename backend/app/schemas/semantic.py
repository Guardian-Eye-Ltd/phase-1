from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class SemanticIndexRequest(BaseModel):
    analysis_job_id: Optional[int] = None
    force_reindex: bool = False

class SemanticStatusResponse(BaseModel):
    evidence_id: int
    status: str
    progress: float
    stage: str
    total_documents: Optional[int] = 0
    total_vlm_observations: Optional[int] = 0
    total_embeddings: Optional[int] = 0
    error: Optional[str] = None

class SearchRequest(BaseModel):
    query: str = Field(..., example="Find a person carrying a backpack near a vehicle")
    top_k: int = Field(5, ge=1, le=20)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    track_id: Optional[int] = None

class ExplanationSchema(BaseModel):
    semantic_match_score: float
    supporting_track: Optional[int] = None
    supporting_keyframe_id: Optional[int] = None
    document_type: str
    source_type: str
    temporal_range: str

class SearchResultItemSchema(BaseModel):
    doc_id: int
    title: str
    summary: str
    score: float
    confidence_level: str
    confidence_reason: str
    query_relevance: Optional[float] = None
    evidence_support: Optional[str] = "HIGH"
    model_confidence: Optional[float] = None
    track_id: Optional[int] = None
    keyframe_id: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    why_explanation: Dict[str, Any]
    verification_status: str

class SearchResponse(BaseModel):
    search_query_id: int
    evidence_id: int
    query: str
    extracted_intent: Dict[str, Any]
    answer: str
    execution_time_ms: float
    total_results: int
    suggestions: Optional[List[str]] = []
    results: List[SearchResultItemSchema]

class ForensicDocumentResponse(BaseModel):
    id: int
    evidence_id: int
    analysis_job_id: int
    track_id: Optional[int] = None
    keyframe_id: Optional[int] = None
    document_type: str
    source_type: str
    title: str
    content: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    metadata_json: Dict[str, Any]
    created_at: datetime

class VLMObservationResponse(BaseModel):
    id: int
    evidence_id: int
    analysis_job_id: int
    keyframe_id: int
    track_id: Optional[int] = None
    model_name: str
    model_version: str
    prompt_version: str
    description: str
    confidence: float
    is_safe: bool
    created_at: datetime

class EnhancedTimelineEvent(BaseModel):
    id: str
    timestamp: float
    timestamp_str: str
    event_type: str
    title: str
    description: str
    track_id: Optional[int] = None
    keyframe_id: Optional[int] = None
    image_path: Optional[str] = None
    confidence_score: float
    source_type: str
    is_clickable: bool = True

class EnhancedTimelineResponse(BaseModel):
    evidence_id: int
    total_events: int
    timeline: List[EnhancedTimelineEvent]
