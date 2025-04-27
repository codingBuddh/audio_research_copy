export enum AudioFeatureType {
  ACOUSTIC = "acoustic",
  SPEAKER = "speaker",
  PARALINGUISTIC = "paralinguistic",
  COGNITIVE = "cognitive",
  TRANSCRIPTION = "transcription"
}

export enum ChunkStatus {
  PENDING = "PENDING",
  PROCESSING = "PROCESSING",
  COMPLETED = "COMPLETED",
  FAILED = "FAILED",
}

export interface AcousticFeatures {
  mfcc: number[];
  pitch: number;
  formants: number[];
  energy: number;
  zcr: number;
  spectral: {
    centroid: number;
    bandwidth: number;
    flux: number;
    rolloff: number;
  };
  vot: number;
}

export interface ParalinguisticFeatures {
  pitch_variability: number;
  speech_rate: number;
  jitter: number;
  shimmer: number;
  hnr: number;
}

export interface AudioChunk {
  chunk_id: number;
  start_time: number;
  end_time: number;
  status: ChunkStatus;
  features?: {
    acoustic?: AcousticFeatures;
    paralinguistic?: ParalinguisticFeatures;
    transcription?: string;
  };
  error?: string;
}

export interface AudioAnalysisResponse {
  task_id: string;
  total_chunks: number;
  chunks: AudioChunk[];
  file_path?: string;
  original_filename?: string;
} 