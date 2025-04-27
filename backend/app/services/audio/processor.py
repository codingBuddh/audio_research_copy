import librosa
import numpy as np
from typing import Dict, List, Optional, Any
import soundfile as sf
from pydub import AudioSegment
import os
from ...schemas.audio import AudioFeatureType, AudioFeatures
import logging
import whisper
import torch
from pathlib import Path
import asyncio
import concurrent.futures
import uuid

# Define model directory
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "models")

class AudioProcessor:
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(__name__)

        # Ensure model directory exists
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Initialize Whisper model with silent warnings and explicit device
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.device = "cpu"  # Explicitly use CPU to avoid CUDA warnings
            
            # Set environment variable to disable FP16 warnings
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
            
            # Initialize Whisper model with explicit fp16=False
            self.logger.info(f"Loading Whisper model from {MODEL_DIR}")
            self.whisper_model = whisper.load_model("tiny.en", device=self.device, download_root=MODEL_DIR)
            self.whisper_model.eval()

    def load_audio_chunk(self, file_path: str, start_time: float, end_time: float) -> np.ndarray:
        """Load a specific chunk of audio file"""
        try:
            y, sr = librosa.load(file_path, sr=self.sample_rate, offset=start_time, duration=end_time-start_time)
            return y
        except Exception as e:
            self.logger.error(f"Error loading audio chunk: {str(e)}")
            raise

    def extract_features(self, audio_chunk: np.ndarray, feature_types: List[AudioFeatureType]) -> AudioFeatures:
        """Extract requested features from audio chunk"""
        features = AudioFeatures()
        
        try:
            if AudioFeatureType.ACOUSTIC in feature_types:
                features.mfcc = self._extract_mfcc(audio_chunk)
                features.pitch = self._extract_pitch(audio_chunk)
                features.formants = self._extract_formants(audio_chunk)
                features.energy = self._extract_energy(audio_chunk)
                features.zcr = self._extract_zcr(audio_chunk)
                features.spectral_features = self._extract_spectral_features(audio_chunk)

            if AudioFeatureType.PARALINGUISTIC in feature_types:
                features.emotion_scores = self._extract_emotion_features(audio_chunk)

            if AudioFeatureType.SPEAKER in feature_types:
                features.speaking_rate = self._extract_speaking_rate(audio_chunk)
                features.voice_onset_time = self._extract_voice_onset_time(audio_chunk)

            return features
        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            raise

    def _extract_mfcc(self, y: np.ndarray, n_mfcc: int = 13) -> List[float]:
        """Extract MFCCs from audio chunk"""
        mfccs = librosa.feature.mfcc(y=y, sr=self.sample_rate, n_mfcc=n_mfcc)
        return mfccs.mean(axis=1).tolist()

    def _extract_pitch(self, y: np.ndarray) -> float:
        """Extract pitch (fundamental frequency) from audio chunk"""
        pitches, magnitudes = librosa.piptrack(y=y, sr=self.sample_rate)
        return float(np.mean(pitches[magnitudes > np.max(magnitudes)*0.7]))

    def _extract_formants(self, y: np.ndarray) -> List[float]:
        """Extract formant frequencies using LPC"""
        # Simplified formant extraction using LPC
        frame_length = 2048
        hop_length = 512
        pre_emphasis = 0.97
        
        # Pre-emphasis filter
        y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
        
        # Extract formants using LPC
        lpc_coeffs = librosa.lpc(y, order=8)
        formants = np.abs(np.roots(lpc_coeffs))
        formants = formants[formants.imag >= 0]
        formants = sorted(formants.real)
        
        return formants[:3].tolist()  # Return first 3 formants

    def _extract_energy(self, y: np.ndarray) -> float:
        """Extract energy from audio chunk"""
        return float(np.sum(y**2))

    def _extract_zcr(self, y: np.ndarray) -> float:
        """Extract zero-crossing rate from audio chunk"""
        zcr = librosa.feature.zero_crossing_rate(y)
        return float(np.mean(zcr))

    def _extract_spectral_features(self, y: np.ndarray) -> Dict[str, float]:
        """Extract various spectral features"""
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=self.sample_rate)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=self.sample_rate)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sample_rate)
        
        return {
            "centroid": float(np.mean(spectral_centroid)),
            "bandwidth": float(np.mean(spectral_bandwidth)),
            "rolloff": float(np.mean(spectral_rolloff))
        }

    def _extract_emotion_features(self, y: np.ndarray) -> Dict[str, float]:
        """Extract features related to emotional content"""
        # This is a simplified version. In practice, you'd want to use a trained model
        energy = np.mean(librosa.feature.rms(y=y))
        pitch_mean = np.mean(librosa.piptrack(y=y, sr=self.sample_rate)[0])
        
        # Simplified emotion scoring based on energy and pitch
        return {
            "arousal": float(energy),
            "valence": float(pitch_mean),
        }

    def _extract_speaking_rate(self, y: np.ndarray) -> float:
        """Estimate speaking rate"""
        # Simplified speaking rate estimation using energy peaks
        hop_length = 512
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sample_rate, hop_length=hop_length)
        peaks = librosa.util.peak_pick(onset_env, 3, 3, 3, 5, 0.5, 10)
        
        duration = len(y) / self.sample_rate
        speaking_rate = len(peaks) / duration  # syllables per second
        return float(speaking_rate)

    def _extract_voice_onset_time(self, y: np.ndarray) -> float:
        """Estimate voice onset time"""
        # Simplified VOT estimation
        energy = librosa.feature.rms(y=y)
        onset_frames = librosa.onset.onset_detect(y=y, sr=self.sample_rate)
        if len(onset_frames) > 0:
            return float(onset_frames[0] * 512 / self.sample_rate)  # Convert frames to seconds
        return 0.0

    # Helper to run a function with timeout
    async def run_with_timeout(self, func, timeout=30):
        """Run a function with timeout using asyncio.wait_for"""
        loop = asyncio.get_event_loop()
        try:
            # Run synchronous function in a thread pool
            return await asyncio.wait_for(
                loop.run_in_executor(None, func),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"Operation timed out after {timeout} seconds")

    def process_chunk(self, audio_path: str, start_time: float, end_time: float, feature_types: List[AudioFeatureType]) -> Dict[str, Any]:
        """Process a chunk of audio and extract requested features"""
        features = {}
        temp_chunk_path = None
        
        try:
            # Load the audio chunk
            self.logger.info(f"Loading chunk from {audio_path} (start={start_time:.2f}s, end={end_time:.2f}s)")
            y, sr = librosa.load(audio_path, offset=start_time, duration=end_time-start_time)
            
            if len(y) == 0:
                raise ValueError("Audio chunk is empty")
                
            # Transcribe audio chunk if needed
            if AudioFeatureType.TRANSCRIPTION in feature_types:
                # Create unique temp file name
                temp_chunk_id = f"{os.path.basename(audio_path)}_{start_time:.2f}_{end_time:.2f}"
                temp_chunk_path = os.path.join(MODEL_DIR, f"temp_chunk_{temp_chunk_id}.wav")
                
                try:
                    # Save temporary chunk for Whisper
                    self.logger.info(f"Saving temp chunk to {temp_chunk_path}")
                    sf.write(temp_chunk_path, y, sr)
                    
                    # Check file exists and is valid
                    if not os.path.exists(temp_chunk_path) or os.path.getsize(temp_chunk_path) < 100:
                        raise ValueError(f"Failed to create valid temp audio file (size: {os.path.getsize(temp_chunk_path) if os.path.exists(temp_chunk_path) else 'file missing'})")
                    
                    # Transcribe with Whisper, with 30 second timeout
                    self.logger.info(f"Transcribing chunk with Whisper")
                    
                    # Use a timeout with our helper function
                    try:
                        # Create a function to run with timeout
                        def run_transcription():
                            with torch.no_grad():
                                return self.whisper_model.transcribe(
                                    temp_chunk_path,
                                    fp16=False,  # Explicitly disable fp16
                                    language='en',
                                    task='transcribe'
                                )
                        
                        # Run with 30 second timeout
                        result = asyncio.run(self.run_with_timeout(run_transcription, 30))
                        features['transcription'] = result["text"].strip()
                        self.logger.info(f"Transcription successful: {features['transcription'][:50]}...")
                            
                    except TimeoutError:
                        features['transcription'] = "[Transcription timed out]"
                        self.logger.error(f"Transcription timed out after 30 seconds")
                        
                finally:
                    # Clean up temporary file
                    if temp_chunk_path and os.path.exists(temp_chunk_path):
                        try:
                            os.remove(temp_chunk_path)
                            self.logger.info(f"Removed temp file {temp_chunk_path}")
                        except Exception as e:
                            self.logger.warning(f"Failed to remove temp file {temp_chunk_path}: {str(e)}")
            
            # Process acoustic features if needed
            if AudioFeatureType.ACOUSTIC in feature_types:
                features['acoustic'] = self._extract_acoustic_features(y, sr)
            
            # Process paralinguistic features if needed
            if AudioFeatureType.PARALINGUISTIC in feature_types:
                features['paralinguistic'] = self._extract_paralinguistic_features(y, sr)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error processing chunk: {str(e)}")
            
            # Ensure temp file cleanup even after errors
            if temp_chunk_path and os.path.exists(temp_chunk_path):
                try:
                    os.remove(temp_chunk_path)
                except Exception:
                    pass
                    
            # Return error in transcription if that was what was requested
            if AudioFeatureType.TRANSCRIPTION in feature_types:
                return {'transcription': f"[Error: {str(e)}]"}
                
            raise

    def _extract_acoustic_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract acoustic features from audio chunk"""
        try:
            # MFCC
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = mfccs.mean(axis=1).tolist()
            
            # Pitch
            f0, voiced_flag, _ = librosa.pyin(
                y, 
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr
            )
            pitch = float(np.nanmean(f0[voiced_flag])) if np.any(voiced_flag) else 0.0
            
            # Formants (simplified using peak detection on spectrum)
            lpc_coeffs = librosa.lpc(y, order=8)
            formants = np.abs(np.roots(lpc_coeffs))
            formants = formants[formants.imag >= 0]
            formants = sorted(formants.real)
            formants = formants[:3].tolist() if len(formants) >= 3 else [0.0, 0.0, 0.0]
            
            # Energy
            energy = float(np.sum(y**2))
            
            # Zero-crossing rate
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
            
            # Spectral features
            # Extract spectral features in frequency domain
            D = librosa.stft(y)
            magnitude_spectrum = np.abs(D)
            
            # Spectral centroid
            centroid = float(np.mean(librosa.feature.spectral_centroid(S=magnitude_spectrum, sr=sr)))
            
            # Spectral bandwidth
            bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(S=magnitude_spectrum, sr=sr)))
            
            # Spectral rolloff
            rolloff = float(np.mean(librosa.feature.spectral_rolloff(S=magnitude_spectrum, sr=sr)))
            
            # Spectral flux - proper frame-by-frame calculation
            # Get magnitude spectra for consecutive frames
            hop_length = 512
            mag_frames = np.abs(librosa.stft(y, hop_length=hop_length))
            
            # Calculate flux between consecutive frames (L1 norm of differences)
            if mag_frames.shape[1] > 1:
                diffs = np.diff(mag_frames, axis=1)
                flux = float(np.mean(np.sum(np.abs(diffs), axis=0)))
            else:
                flux = 0.0
            
            return {
                "mfcc": mfcc_mean,
                "pitch": pitch,
                "formants": formants,
                "energy": energy,
                "zcr": zcr,
                "spectral": {
                    "centroid": centroid,
                    "bandwidth": bandwidth,
                    "rolloff": rolloff,
                    "flux": flux
                }
            }
        except Exception as e:
            self.logger.error(f"Error extracting acoustic features: {str(e)}")
            raise

    def _extract_paralinguistic_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract paralinguistic features from audio chunk"""
        try:
            # Pitch variability 
            f0, voiced_flag, _ = librosa.pyin(
                y, 
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr
            )
            f0_voiced = f0[voiced_flag]
            
            pitch_variability = float(np.std(f0_voiced) / np.mean(f0_voiced)) * 100 if len(f0_voiced) > 0 and np.mean(f0_voiced) > 0 else 0.0
            
            # Speech rate
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            peaks = librosa.util.peak_pick(onset_env, 3, 3, 3, 5, 0.5, 10)
            duration = len(y) / sr
            speech_rate = float(len(peaks) / duration if duration > 0 else 0)
            
            # Voice quality (jitter/shimmer)
            if len(f0_voiced) > 1:
                f0_diff = np.abs(np.diff(f0_voiced))
                jitter = float(np.mean(f0_diff) / np.mean(f0_voiced)) * 100
            else:
                jitter = 0.0
                
            # Calculate shimmer using RMS amplitude
            rms_frames = librosa.feature.rms(y=y)[0]
            if len(rms_frames) > 1:
                rms_diff = np.abs(np.diff(rms_frames))
                shimmer = float(np.mean(rms_diff) / np.mean(rms_frames)) * 100
            else:
                shimmer = 0.0
                
            # Harmonics-to-Noise Ratio
            # Use harmonic-percussive source separation to estimate HNR
            D = librosa.stft(y)
            H, P = librosa.decompose.hpss(D)
            harmonic_energy = np.sum(np.abs(H)**2)
            noise_energy = np.sum(np.abs(P)**2)
            hnr = 10 * np.log10(harmonic_energy / noise_energy) if noise_energy > 0 else 40.0
            
            return {
                "pitch_variability": pitch_variability,
                "speech_rate": speech_rate,
                "jitter": jitter,
                "shimmer": shimmer,
                "hnr": float(hnr)
            }
        except Exception as e:
            self.logger.error(f"Error extracting paralinguistic features: {str(e)}")
            raise

    async def process_audio_data(self, audio_data: np.ndarray, sr: int, feature_types: List[AudioFeatureType]) -> Dict[str, Any]:
        """Process audio data directly without loading from file"""
        features = {}
        temp_chunk_path = None
        
        try:
            # Validate audio data
            if len(audio_data) == 0:
                raise ValueError("Audio chunk is empty")
                
            # Process higher priority features first
            # Transcription is the most resource-intensive, so prioritize it
            if AudioFeatureType.TRANSCRIPTION in feature_types:
                # Create unique temp file name with UUID to avoid conflicts
                temp_chunk_id = str(uuid.uuid4())
                temp_chunk_path = os.path.join(MODEL_DIR, f"temp_chunk_{temp_chunk_id}.wav")
                
                try:
                    # Save temporary chunk for Whisper
                    self.logger.info(f"Saving temp chunk to {temp_chunk_path}")
                    sf.write(temp_chunk_path, audio_data, sr)
                    
                    # Check file exists and is valid
                    if not os.path.exists(temp_chunk_path) or os.path.getsize(temp_chunk_path) < 100:
                        raise ValueError(f"Failed to create valid temp audio file (size: {os.path.getsize(temp_chunk_path) if os.path.exists(temp_chunk_path) else 'file missing'})")
                    
                    # Transcribe with Whisper, with 30 second timeout
                    self.logger.info(f"Transcribing chunk with Whisper")
                    
                    # Use a timeout with our helper function
                    try:
                        # Create a function to run with timeout
                        def run_transcription():
                            with torch.no_grad():
                                return self.whisper_model.transcribe(
                                    temp_chunk_path,
                                    fp16=False,  # Explicitly disable fp16
                                    language='en',
                                    task='transcribe'
                                )
                        
                        # Run with 30 second timeout
                        result = await self.run_with_timeout(run_transcription, 30)
                        features['transcription'] = result["text"].strip()
                        self.logger.info(f"Transcription successful: {features['transcription'][:50]}...")
                            
                    except TimeoutError:
                        features['transcription'] = "[Transcription timed out]"
                        self.logger.error(f"Transcription timed out after 30 seconds")
                        
                finally:
                    # Clean up temporary file
                    if temp_chunk_path and os.path.exists(temp_chunk_path):
                        try:
                            os.remove(temp_chunk_path)
                            self.logger.info(f"Removed temp file {temp_chunk_path}")
                        except Exception as e:
                            self.logger.warning(f"Failed to remove temp file {temp_chunk_path}: {str(e)}")
            
            # Then process less resource-intensive features
            if AudioFeatureType.ACOUSTIC in feature_types:
                features['acoustic'] = self._extract_acoustic_features(audio_data, sr)
            
            if AudioFeatureType.PARALINGUISTIC in feature_types:
                features['paralinguistic'] = self._extract_paralinguistic_features(audio_data, sr)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error processing audio data: {str(e)}")
            
            # Ensure temp file cleanup even after errors
            if temp_chunk_path and os.path.exists(temp_chunk_path):
                try:
                    os.remove(temp_chunk_path)
                except Exception:
                    pass
                    
            # Return error in transcription if that was what was requested
            if AudioFeatureType.TRANSCRIPTION in feature_types:
                return {'transcription': f"[Error: {str(e)}]"}
                
            raise 