import librosa
import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks
from scipy.fft import rfft, rfftfreq
import logging
import soundfile as sf
import os
from pathlib import Path
import whisper
import torch
from typing import Dict, List, Any, Optional, Tuple
from ...schemas.audio import AudioFeatureType, AcousticFeatures, SpectralFeatures, ParalinguisticFeatures, AudioFeatures

# Define model directory
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "models")

logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        
        # Ensure model directory exists
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Initialize Whisper model with silent warnings and explicit device
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            # Explicitly use CPU to avoid CUDA warnings
            self.device = "cpu" 
            
            # Set environment variables to disable FP16 warnings
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
            
            # Load model with explicit fp16=False for CPU
            logger.info(f"Loading Whisper model from {MODEL_DIR}")
            self.whisper_model = whisper.load_model("tiny.en", device=self.device, download_root=MODEL_DIR)
            self.whisper_model.eval()

    def extract_features(self, audio_chunk: np.ndarray, feature_types: List[AudioFeatureType]) -> Dict[str, Any]:
        """Extract requested features from audio chunk"""
        features = {}
        
        try:
            if AudioFeatureType.ACOUSTIC in feature_types:
                acoustic_features = self._extract_acoustic_features(audio_chunk)
                features["acoustic"] = acoustic_features.model_dump()
                
            if AudioFeatureType.PARALINGUISTIC in feature_types:
                paralinguistic_features = self._extract_paralinguistic_features(audio_chunk)
                features["paralinguistic"] = paralinguistic_features.model_dump()
                
            if AudioFeatureType.TRANSCRIPTION in feature_types:
                transcription = self.process_transcription(audio_chunk, self.sample_rate)
                features["transcription"] = transcription
                
            return features
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            raise

    def _extract_acoustic_features(self, audio_chunk: np.ndarray) -> AcousticFeatures:
        """Extract acoustic features using optimized computations"""
        try:
            # 1. Compute FFT for frequency-domain features
            N = len(audio_chunk)
            yf = rfft(audio_chunk)
            xf = rfftfreq(N, 1 / self.sample_rate)
            spectrum = np.abs(yf)
            
            # 2. MFCCs (using librosa for this as it's optimized)
            mfccs = librosa.feature.mfcc(y=audio_chunk, sr=self.sample_rate, n_mfcc=13)
            mfcc_means = mfccs.mean(axis=1).tolist()

            # 3. Pitch using peak detection in frequency domain
            peaks, _ = find_peaks(spectrum, height=np.max(spectrum) * 0.1)
            pitch = float(xf[peaks[0]]) if len(peaks) > 0 else 0.0

            # 4. Formants using peak detection in specific frequency ranges
            formant_ranges = [(300, 1000), (850, 2500), (1950, 3000)]  # F1, F2, F3 ranges
            formants = []
            for f_min, f_max in formant_ranges:
                mask = (xf >= f_min) & (xf <= f_max)
                if np.any(mask):
                    formant_peak = xf[mask][np.argmax(spectrum[mask])]
                    formants.append(float(formant_peak))
                else:
                    formants.append(0.0)

            # 5. Energy (RMS)
            energy = float(np.sqrt(np.mean(audio_chunk**2)))

            # 6. Zero-crossing rate
            zcr = float(np.sum(np.abs(np.diff(np.signbit(audio_chunk)))) / (2 * len(audio_chunk)))

            # 7. Spectral features
            spectral = self._compute_spectral_features(spectrum, xf)

            # 8. Voice Onset Time (simplified)
            envelope = np.abs(audio_chunk)
            onset_threshold = np.mean(envelope) + 0.5 * np.std(envelope)
            onsets = np.where(envelope > onset_threshold)[0]
            vot = float(onsets[0] / self.sample_rate) if len(onsets) > 0 else None

            return AcousticFeatures(
                mfcc=mfcc_means,
                pitch=pitch,
                formants=formants,
                energy=energy,
                zcr=zcr,
                spectral=spectral,
                vot=vot
            )
        except Exception as e:
            logger.error(f"Error in acoustic feature extraction: {str(e)}")
            raise

    def _compute_spectral_features(self, spectrum: np.ndarray, frequencies: np.ndarray) -> SpectralFeatures:
        """Compute spectral features using optimized numpy operations"""
        try:
            # Normalize spectrum
            spectrum_norm = spectrum / np.sum(spectrum) if np.sum(spectrum) > 0 else spectrum
            
            # Spectral centroid
            centroid = float(np.sum(frequencies * spectrum_norm))
            
            # Spectral bandwidth
            bandwidth = float(np.sqrt(np.sum(((frequencies - centroid) ** 2) * spectrum_norm)))
            
            # Spectral flux - improved calculation
            # For a single frame, we'll estimate flux by measuring the average rate of change across frequency bins
            if len(spectrum) > 1:
                # Calculate differences between adjacent frequency bins
                diff_spectrum = np.diff(spectrum_norm)
                # Use absolute differences to capture both increases and decreases
                abs_diff = np.abs(diff_spectrum)
                # Average rate of change (normalized by number of bins)
                flux = float(np.sum(abs_diff) / (len(spectrum) - 1))
            else:
                flux = 0.0
            
            # Spectral rolloff
            cumsum = np.cumsum(spectrum_norm)
            rolloff_point = np.where(cumsum >= 0.85)[0]
            rolloff = float(frequencies[rolloff_point[0]]) if len(rolloff_point) > 0 else 0.0
            
            return SpectralFeatures(
                centroid=centroid,
                bandwidth=bandwidth,
                flux=flux,
                rolloff=rolloff
            )
        except Exception as e:
            logger.error(f"Error in spectral feature computation: {str(e)}")
            raise

    def _extract_paralinguistic_features(self, audio_chunk: np.ndarray) -> ParalinguisticFeatures:
        """Extract paralinguistic features using optimized computations"""
        try:
            # 1. Pitch Variability
            yf = rfft(audio_chunk)
            xf = rfftfreq(len(audio_chunk), 1 / self.sample_rate)
            spectrum = np.abs(yf)
            peaks, _ = find_peaks(spectrum, height=np.max(spectrum) * 0.1)
            pitch_values = xf[peaks]
            pitch_variability = float(np.std(pitch_values)) if len(pitch_values) > 0 else 0.0

            # 2. Speech Rate using energy-based syllable detection
            envelope = np.abs(audio_chunk)
            envelope_smooth = np.convolve(envelope, np.ones(512)/512, mode='same')
            peaks, _ = find_peaks(envelope_smooth, height=np.mean(envelope_smooth) * 1.5)
            duration = len(audio_chunk) / self.sample_rate
            speech_rate = float(len(peaks) / duration) if duration > 0 else 0.0

            # 3. Jitter calculation using zero-crossings
            zero_crossings = np.where(np.diff(np.signbit(audio_chunk)))[0]
            if len(zero_crossings) > 1:
                periods = np.diff(zero_crossings)
                jitter = float(np.std(periods) / np.mean(periods))
            else:
                jitter = 0.0

            # 4. Shimmer calculation using local maxima
            peaks, _ = find_peaks(envelope, distance=int(0.01 * self.sample_rate))
            if len(peaks) > 1:
                peak_amplitudes = envelope[peaks]
                shimmer = float(np.std(peak_amplitudes) / np.mean(peak_amplitudes))
            else:
                shimmer = 0.0

            # 5. Harmonics-to-Noise Ratio
            harmonic_peaks, _ = find_peaks(spectrum, height=np.mean(spectrum))
            if len(harmonic_peaks) > 0:
                harmonic_energy = np.sum(spectrum[harmonic_peaks])
                total_energy = np.sum(spectrum)
                noise_energy = total_energy - harmonic_energy
                hnr = float(10 * np.log10(harmonic_energy / noise_energy)) if noise_energy > 0 else 40.0
            else:
                hnr = 0.0

            return ParalinguisticFeatures(
                pitch_variability=pitch_variability,
                speech_rate=speech_rate,
                jitter=jitter,
                shimmer=shimmer,
                hnr=hnr
            )
        except Exception as e:
            logger.error(f"Error in paralinguistic feature extraction: {str(e)}")
            raise

    def _extract_pitch(self, audio_chunk: np.ndarray) -> float:
        """Extract fundamental frequency (pitch)"""
        pitches, magnitudes = librosa.piptrack(y=audio_chunk, sr=self.sample_rate)
        pitch = float(pitches[magnitudes > 0.5].mean()) if len(pitches[magnitudes > 0.5]) > 0 else 0.0
        return pitch

    def _extract_emotion_scores(self, audio_chunk: np.ndarray) -> Dict[str, float]:
        """Extract emotion-related features (arousal and valence)"""
        # Calculate basic audio features that correlate with emotions
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_chunk, sr=self.sample_rate).mean()
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_chunk, sr=self.sample_rate).mean()
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_chunk).mean()
        
        # Map these features to arousal and valence scores (simplified mapping)
        arousal = min(1.0, max(0.0, (spectral_centroid / 5000 + zero_crossing_rate * 100) / 2))
        valence = min(1.0, max(0.0, spectral_rolloff / 12000))
        
        return {
            "arousal": float(arousal),
            "valence": float(valence)
        }

    def _extract_speaking_rate(self, audio_chunk: np.ndarray) -> float:
        """Estimate speaking rate in syllables per second"""
        # Detect onsets as a proxy for syllables
        onset_env = librosa.onset.onset_strength(y=audio_chunk, sr=self.sample_rate)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=self.sample_rate)
        
        # Calculate syllables per second
        duration = len(audio_chunk) / self.sample_rate
        speaking_rate = len(onsets) / duration if duration > 0 else 0.0
        
        return float(speaking_rate)

    def _calculate_pitch_variability(self, audio_chunk: np.ndarray) -> float:
        """Calculate pitch variability using PYIN algorithm for better F0 estimation"""
        try:
            # Use PYIN for more accurate pitch tracking
            f0, voiced_flag, _ = librosa.pyin(
                audio_chunk,
                fmin=librosa.note_to_hz('C2'),  # ~65 Hz
                fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
                sr=self.sample_rate,
                frame_length=2048
            )
            
            # Filter out unvoiced segments and zeros
            f0_voiced = f0[voiced_flag & (f0 > 0)]
            
            if len(f0_voiced) > 0:
                # Calculate normalized standard deviation
                pitch_mean = np.mean(f0_voiced)
                pitch_std = np.std(f0_voiced)
                pitch_variability = float(pitch_std / pitch_mean if pitch_mean > 0 else 0.0)
                
                # Scale to a more interpretable range (0-100 Hz typical range)
                pitch_variability *= 100
            else:
                pitch_variability = 0.0
                
            return pitch_variability
            
        except Exception as e:
            logger.error(f"Error calculating pitch variability: {str(e)}")
            return 0.0

    def _calculate_speech_rate(self, audio_chunk: np.ndarray) -> float:
        """Calculate speech rate using enhanced syllable detection"""
        try:
            # 1. Get onset envelope with custom parameters
            hop_length = 512
            oenv = librosa.onset.onset_strength(
                y=audio_chunk,
                sr=self.sample_rate,
                hop_length=hop_length,
                aggregate=np.median  # More robust to noise
            )
            
            # 2. Detect onsets using peak picking
            onsets = librosa.onset.onset_detect(
                onset_envelope=oenv,
                sr=self.sample_rate,
                hop_length=hop_length,
                backtrack=True,
                normalize=True
            )
            
            # 3. Calculate speech rate
            duration = len(audio_chunk) / self.sample_rate
            if duration > 0:
                # Convert frame indices to time
                onset_times = librosa.frames_to_time(onsets, sr=self.sample_rate, hop_length=hop_length)
                # Estimate syllables (with correction factor for potential false positives)
                estimated_syllables = len(onset_times) * 0.85
                speech_rate = float(estimated_syllables / duration)
            else:
                speech_rate = 0.0
                
            return speech_rate
            
        except Exception as e:
            logger.error(f"Error calculating speech rate: {str(e)}")
            return 0.0

    def _calculate_voice_quality(self, audio_chunk: np.ndarray) -> tuple[float, float]:
        """Calculate jitter and shimmer using cycle-to-cycle analysis"""
        try:
            # 1. Get fundamental frequency using PYIN
            f0, voiced_flag, _ = librosa.pyin(
                audio_chunk,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sample_rate
            )
            
            # Get amplitude envelope
            hop_length = 256
            rms = librosa.feature.rms(y=audio_chunk, hop_length=hop_length)[0]
            
            # Calculate jitter
            f0_voiced = f0[voiced_flag]
            if len(f0_voiced) > 1:
                # Relative average perturbation (RAP)
                f0_diff = np.abs(np.diff(f0_voiced))
                jitter = float(np.mean(f0_diff) / np.mean(f0_voiced))
            else:
                jitter = 0.0
            
            # Calculate shimmer
            if len(rms) > 1:
                # Amplitude perturbation quotient (APQ)
                amp_diff = np.abs(np.diff(rms))
                shimmer = float(np.mean(amp_diff) / np.mean(rms))
            else:
                shimmer = 0.0
            
            # Scale to percentage values
            jitter *= 100
            shimmer *= 100
            
            return jitter, shimmer
            
        except Exception as e:
            logger.error(f"Error calculating voice quality: {str(e)}")
            return 0.0, 0.0

    def _calculate_hnr(self, audio_chunk: np.ndarray) -> float:
        """Calculate Harmonics-to-Noise Ratio using enhanced method"""
        try:
            # 1. Compute STFT
            D = librosa.stft(audio_chunk)
            S = np.abs(D)
            
            # 2. Harmonic-percussive source separation
            H, P = librosa.decompose.hpss(
                S,
                kernel_size=31,  # Larger kernel for better separation
                power=2.0,  # Soft mask
                margin=3.0  # Wider margin for clear separation
            )
            
            # 3. Calculate HNR
            harmonic_energy = np.sum(H**2)
            noise_energy = np.sum(P**2)
            
            if noise_energy > 0:
                # Convert to dB scale
                hnr = float(10 * np.log10(harmonic_energy / noise_energy))
                
                # Clip to reasonable range (-20 to 40 dB typical)
                hnr = np.clip(hnr, -20.0, 40.0)
            else:
                hnr = 40.0  # Maximum value when no noise detected
            
            return hnr
            
        except Exception as e:
            logger.error(f"Error calculating HNR: {str(e)}")
            return 0.0

    def process_transcription(self, audio_chunk: np.ndarray, sr: int) -> str:
        """Process audio chunk for transcription using Whisper"""
        try:
            # Save temporary chunk for Whisper
            temp_chunk_path = f"temp_chunk_{id(audio_chunk)}.wav"
            sf.write(temp_chunk_path, audio_chunk, sr)
            
            # Transcribe with Whisper
            try:
                result = self.whisper_model.transcribe(temp_chunk_path)
                return result["text"].strip()
            finally:
                # Clean up temporary file
                Path(temp_chunk_path).unlink(missing_ok=True)
        except Exception as e:
            logger.error(f"Error processing transcription: {str(e)}")
            return "" 