from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseVideoAnalysisService(ABC):
    @abstractmethod
    async def analyze_video(self, evidence_id: int, file_path: str) -> Dict[str, Any]:
        pass

class BaseDetectionService(ABC):
    @abstractmethod
    async def detect_objects(self, evidence_id: int, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        pass

class BaseTrackingService(ABC):
    @abstractmethod
    async def track_persons(self, evidence_id: int) -> List[Dict[str, Any]]:
        pass

class BaseFaceRecognitionService(ABC):
    @abstractmethod
    async def recognize_faces(self, evidence_id: int) -> List[Dict[str, Any]]:
        pass

class BaseVLMService(ABC):
    @abstractmethod
    async def analyze_frame_with_prompt(self, evidence_id: int, frame_time: float, prompt: str) -> Dict[str, Any]:
        pass

class BaseEmbeddingService(ABC):
    @abstractmethod
    async def generate_evidence_embeddings(self, evidence_id: int) -> Dict[str, Any]:
        pass

class BaseSearchService(ABC):
    @abstractmethod
    async def semantic_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        pass

class BaseTimelineService(ABC):
    @abstractmethod
    async def generate_timeline(self, evidence_id: int) -> List[Dict[str, Any]]:
        pass

class BaseForensicReportService(ABC):
    @abstractmethod
    async def generate_report(self, evidence_id: int, investigator_id: int) -> Dict[str, Any]:
        pass


# =====================================================================
# PHASE 1 PLACEHOLDER IMPLEMENTATIONS (Plug-and-Play ready for Phase 2)
# =====================================================================

class PlaceholderVideoAnalysisService(BaseVideoAnalysisService):
    async def analyze_video(self, evidence_id: int, file_path: str) -> Dict[str, Any]:
        return {
            "status": "not_implemented",
            "message": "AI Video Analysis module scheduled for Phase 1B.",
            "evidence_id": evidence_id
        }

class PlaceholderDetectionService(BaseDetectionService):
    async def detect_objects(self, evidence_id: int, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        return [{
            "status": "not_implemented",
            "message": "YOLO Object & Person Detection engine not yet active."
        }]

class PlaceholderTrackingService(BaseTrackingService):
    async def track_persons(self, evidence_id: int) -> List[Dict[str, Any]]:
        return [{
            "status": "not_implemented",
            "message": "Multi-Camera Person Re-ID and tracking scheduled for Phase 1B."
        }]

class PlaceholderFaceRecognitionService(BaseFaceRecognitionService):
    async def recognize_faces(self, evidence_id: int) -> List[Dict[str, Any]]:
        return [{
            "status": "not_implemented",
            "message": "Forensic Face Recognition module scheduled for Phase 1B."
        }]

class PlaceholderVLMService(BaseVLMService):
    async def analyze_frame_with_prompt(self, evidence_id: int, frame_time: float, prompt: str) -> Dict[str, Any]:
        return {
            "status": "not_implemented",
            "message": "Qwen-VL / LLaVA Vision-Language Model integration scheduled for Phase 1C."
        }

class PlaceholderEmbeddingService(BaseEmbeddingService):
    async def generate_evidence_embeddings(self, evidence_id: int) -> Dict[str, Any]:
        return {
            "status": "not_implemented",
            "message": "ChromaDB Vector Embeddings generation scheduled for Phase 1C."
        }

class PlaceholderSearchService(BaseSearchService):
    async def semantic_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        return [{
            "status": "not_implemented",
            "message": "Semantic Evidence Search scheduled for Phase 1C."
        }]

class PlaceholderTimelineService(BaseTimelineService):
    async def generate_timeline(self, evidence_id: int) -> List[Dict[str, Any]]:
        return [{
            "status": "not_implemented",
            "message": "Interactive Forensic Timeline generation scheduled for Phase 1B."
        }]

class PlaceholderForensicReportService(BaseForensicReportService):
    async def generate_report(self, evidence_id: int, investigator_id: int) -> Dict[str, Any]:
        return {
            "status": "not_implemented",
            "message": "Automated PDF Forensic Report Generation scheduled for Phase 1C."
        }
