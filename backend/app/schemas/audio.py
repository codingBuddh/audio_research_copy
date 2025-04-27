from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from enum import Enum

class AudioFeatureType(Enum):
    ACOUSTIC = "acoustic"
    SPEAKER = "speaker"
    PARALINGUISTIC = "paralinguistic"
    COGNITIVE = "cognitive"
    TRANSCRIPTION = "transcription"

class ChunkStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class SpectralFeatures(BaseModel):
    centroid: float = Field(description="Spectral centroid - brightness of the sound")
    bandwidth: float = Field(description="Spectral bandwidth - width of the spectrum")
    flux: float = Field(description="Spectral flux - rate of change of the spectrum")
    rolloff: float = Field(description="Spectral rolloff - frequency below which 85% of the spectrum is concentrated")

class AcousticFeatures(BaseModel):
    mfcc: List[float] = Field(description="Mel-frequency cepstral coefficients")
    pitch: float = Field(description="Fundamental frequency (F0)")
    formants: List[float] = Field(description="Formant frequencies (F1, F2, F3)")
    energy: float = Field(description="Root mean square energy")
    zcr: float = Field(description="Zero-crossing rate")
    spectral: SpectralFeatures = Field(description="Spectral features")
    vot: Optional[float] = Field(None, description="Voice onset time")

class ParalinguisticFeatures(BaseModel):
    pitch_variability: float = Field(description="Variability in fundamental frequency")
    speech_rate: float = Field(description="Speech rate in syllables per second")
    jitter: float = Field(description="Cycle-to-cycle variations in fundamental frequency")
    shimmer: float = Field(description="Cycle-to-cycle variations in amplitude")
    hnr: float = Field(description="Harmonics-to-Noise Ratio")

class AudioFeatures(BaseModel):
    acoustic: Optional[AcousticFeatures] = None
    paralinguistic: Optional[ParalinguisticFeatures] = None
    transcription: Optional[str] = Field(
        None,
        description="Text transcription of audio using Whisper"
    )

class AudioChunk(BaseModel):
    chunk_id: int = Field(description="Unique identifier for the chunk")
    start_time: float = Field(description="Start time of the chunk in seconds")
    end_time: float = Field(description="End time of the chunk in seconds")
    status: ChunkStatus = Field(description="Current status of chunk processing")
    features: Optional[AudioFeatures] = Field(
        None,
        description="Extracted features for the chunk"
    )
    error: Optional[str] = Field(
        None,
        description="Error message if processing failed"
    )

class AudioAnalysisRequest(BaseModel):
    feature_types: List[AudioFeatureType] = Field(
        description="List of audio features to extract",
        min_items=1
    )
    chunk_duration: float = Field(
        default=60.0,
        description="Duration of each audio chunk in seconds",
        gt=0
    )

class AudioAnalysisResponse(BaseModel):
    task_id: str = Field(description="Unique identifier for the analysis task")
    total_chunks: int = Field(description="Total number of chunks to process")
    chunks: List[AudioChunk] = Field(description="List of audio chunks and their analysis results")
    file_path: Optional[str] = Field(None, description="Path to the audio file being analyzed")
    original_filename: Optional[str] = Field(None, description="Original filename of the uploaded audio") 