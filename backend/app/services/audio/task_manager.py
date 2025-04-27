import asyncio
import uuid
from typing import Dict, List, Optional, Set, Any, Deque
from collections import deque
import librosa
import numpy as np
from fastapi import WebSocket
from ...schemas.audio import AudioAnalysisResponse, ChunkStatus, AudioFeatures, AudioFeatureType
from .feature_extractor import FeatureExtractor
from .processor import AudioProcessor
import logging

logger = logging.getLogger(__name__)

class AudioTaskManager:
    def __init__(self, max_concurrent_tasks=3):
        self.tasks: Dict[str, AudioAnalysisResponse] = {}
        self.clients: Dict[str, Set[WebSocket]] = {}
        self.feature_extractor = FeatureExtractor()
        self.max_concurrent_tasks = max_concurrent_tasks
        self.active_tasks = 0
        self.task_queue: Deque[str] = deque()
        self.task_queue_lock = asyncio.Lock()

    async def create_task(self, file_path: str, feature_types: List[str], chunk_duration: float = 5.0, file_metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new audio analysis task"""
        task_id = str(uuid.uuid4())
        
        try:
            # Load audio file
            audio, sr = librosa.load(file_path, sr=None)
            
            # Calculate chunk size in samples
            chunk_size = int(chunk_duration * sr)
            total_chunks = int(np.ceil(len(audio) / chunk_size))
            
            # Initialize task status
            task_data = {
                "task_id": task_id,
                "total_chunks": total_chunks,
                "file_path": file_path,
                "chunks": [{
                    "chunk_id": i,
                    "start_time": i * chunk_duration,
                    "end_time": min((i + 1) * chunk_duration, len(audio) / sr),
                    "status": ChunkStatus.PENDING,  # Start as PENDING instead of PROCESSING
                    "features": None,
                    "error": None
                } for i in range(total_chunks)]
            }
            
            # Add metadata if provided
            if file_metadata:
                if "original_filename" in file_metadata:
                    task_data["original_filename"] = file_metadata["original_filename"]
            
            self.tasks[task_id] = AudioAnalysisResponse(**task_data)
            
            # Add task to the queue and start processing if there are available slots
            async with self.task_queue_lock:
                if self.active_tasks < self.max_concurrent_tasks:
                    # Start processing immediately
                    self.active_tasks += 1
                    asyncio.create_task(self._process_audio(
                        task_id=task_id,
                        audio=audio,
                        sr=sr,
                        chunk_size=chunk_size,
                        feature_types=feature_types
                    ))
                    logger.info(f"Started processing task {task_id} immediately ({self.active_tasks}/{self.max_concurrent_tasks} active tasks)")
                else:
                    # Add to queue
                    self.task_queue.append((task_id, audio, sr, chunk_size, feature_types))
                    logger.info(f"Task {task_id} added to the queue (position {len(self.task_queue)}, {self.active_tasks}/{self.max_concurrent_tasks} active tasks)")
                    
                    # Notify clients of queue position
                    for i, chunk in enumerate(self.tasks[task_id].chunks):
                        chunk.error = f"Queued for processing (position {len(self.task_queue)})"
                    await self._notify_clients(task_id)
            
            return task_id
            
        except Exception as e:
            logger.error(f"Error creating task: {str(e)}")
            raise

    async def _process_next_task(self):
        """Process the next task in the queue"""
        async with self.task_queue_lock:
            if not self.task_queue:
                return
                
            # Get the next task from the queue
            task_id, audio, sr, chunk_size, feature_types = self.task_queue.popleft()
            logger.info(f"Starting queued task {task_id} ({self.active_tasks}/{self.max_concurrent_tasks} active tasks)")
            
            # Start processing
            asyncio.create_task(self._process_audio(
                task_id=task_id,
                audio=audio,
                sr=sr,
                chunk_size=chunk_size,
                feature_types=feature_types
            ))

    async def _process_audio(self, task_id: str, audio: np.ndarray, sr: int, 
                           chunk_size: int, feature_types: List[str]):
        """Process audio file in chunks"""
        task = self.tasks[task_id]
        
        try:
            # Create processor once for all chunks (more efficient)
            processor = AudioProcessor()
            
            # Convert string feature types to enum if needed
            enum_feature_types = []
            for ft in feature_types:
                if isinstance(ft, str):
                    try:
                        enum_feature_types.append(AudioFeatureType(ft))
                    except ValueError:
                        logger.warning(f"Unknown feature type: {ft}")
                else:
                    enum_feature_types.append(ft)
            
            # Process each chunk with timeout protection
            for i in range(task.total_chunks):
                try:
                    # Update status to PROCESSING
                    task.chunks[i].status = ChunkStatus.PROCESSING
                    # Clear any queue-related message
                    task.chunks[i].error = None
                    await self._notify_clients(task_id)
                    
                    # Set a per-chunk timeout using wait_for
                    try:
                        # Extract chunk
                        start = i * chunk_size
                        end = min(start + chunk_size, len(audio))
                        chunk = audio[start:end]
                        
                        # Calculate time in seconds
                        start_time = start / sr
                        end_time = end / sr
                        
                        # Log processing status
                        logger.info(f"Processing chunk {i+1}/{task.total_chunks} for task {task_id}")
                        
                        # Create a processing function to run with timeout
                        async def process_chunk():
                            features = {}
                            
                            # Handle basic features (non-transcription)
                            basic_feature_types = [ft for ft in enum_feature_types if ft != AudioFeatureType.TRANSCRIPTION]
                            if basic_feature_types:
                                basic_features = self.feature_extractor.extract_features(chunk, basic_feature_types)
                                features.update(basic_features)
                            
                            # Handle transcription separately if needed
                            if AudioFeatureType.TRANSCRIPTION in enum_feature_types:
                                try:
                                    # Use the chunk data directly instead of reading from file
                                    # This avoids the file not found error
                                    transcription_result = await processor.process_audio_data(
                                        chunk, 
                                        sr,
                                        [AudioFeatureType.TRANSCRIPTION]
                                    )
                                    if 'transcription' in transcription_result:
                                        features['transcription'] = transcription_result['transcription']
                                except Exception as e:
                                    logger.error(f"Error processing transcription for chunk {i}: {str(e)}")
                                    features['transcription'] = f"[Transcription error: {str(e)}]"
                            
                            return features
                        
                        # Run with timeout
                        features = await asyncio.wait_for(process_chunk(), timeout=60)  # 60 second timeout
                        
                        # Update chunk status
                        audio_features = AudioFeatures(**features)
                        task.chunks[i].status = ChunkStatus.COMPLETED
                        task.chunks[i].features = audio_features
                        
                        # Notify clients after each chunk
                        await self._notify_clients(task_id)
                        
                        # Small delay to prevent overloading
                        await asyncio.sleep(0.1)
                    
                    except Exception as e:
                        # This covers all exceptions within this inner try block
                        logger.error(f"Error processing chunk details for {i}: {str(e)}")
                        raise  # Re-raise to be caught by the outer exception handler
                
                except asyncio.TimeoutError:
                    logger.error(f"Timeout processing chunk {i} for task {task_id}")
                    task.chunks[i].status = ChunkStatus.FAILED
                    task.chunks[i].error = "Processing timed out"
                    await self._notify_clients(task_id)
                
                except Exception as e:
                    logger.error(f"Error processing chunk {i}: {str(e)}")
                    task.chunks[i].status = ChunkStatus.FAILED
                    task.chunks[i].error = str(e)
                    await self._notify_clients(task_id)
            
            logger.info(f"Completed processing all chunks for task {task_id}")
            
        except Exception as e:
            logger.error(f"Fatal error in audio processing for task {task_id}: {str(e)}")
            # Mark all remaining chunks as failed
            for i in range(task.total_chunks):
                if task.chunks[i].status == ChunkStatus.PROCESSING or task.chunks[i].status == ChunkStatus.PENDING:
                    task.chunks[i].status = ChunkStatus.FAILED
                    task.chunks[i].error = f"Task processing failed: {str(e)}"
            
            # Notify about the failure
            await self._notify_clients(task_id)
        finally:
            # Reduce active task count and process next task in queue if available
            async with self.task_queue_lock:
                self.active_tasks -= 1
                logger.info(f"Task {task_id} completed. Active tasks: {self.active_tasks}/{self.max_concurrent_tasks}")
                
                # Immediately process the next task in the queue if there's room
                if self.task_queue and self.active_tasks < self.max_concurrent_tasks:
                    self.active_tasks += 1
                    # Start next task, but don't wait for it
                    asyncio.create_task(self._process_next_task())

    def get_task_status(self, task_id: str) -> Optional[AudioAnalysisResponse]:
        """Get the current status of a task"""
        return self.tasks.get(task_id)

    def register_client(self, task_id: str, websocket: WebSocket):
        """Register a WebSocket client for task updates"""
        if task_id not in self.clients:
            self.clients[task_id] = set()
        self.clients[task_id].add(websocket)

    def unregister_client(self, task_id: str, websocket: WebSocket):
        """Unregister a WebSocket client"""
        if task_id in self.clients:
            self.clients[task_id].discard(websocket)
            if not self.clients[task_id]:
                del self.clients[task_id]

    async def _notify_clients(self, task_id: str):
        """Notify all clients about task updates"""
        if task_id in self.clients and task_id in self.tasks:
            dead_clients = set()
            
            for websocket in self.clients[task_id]:
                try:
                    await websocket.send_json(self.tasks[task_id].model_dump())
                except Exception:
                    dead_clients.add(websocket)
            
            # Remove dead clients
            for websocket in dead_clients:
                self.unregister_client(task_id, websocket)

    async def wait_for_task_completion(self, task_id: str, timeout_seconds: float = 300) -> Optional[AudioAnalysisResponse]:
        """Wait for a task to complete with a timeout"""
        if task_id not in self.tasks:
            logger.warning(f"Attempted to wait for non-existent task: {task_id}")
            return None
        
        start_time = asyncio.get_event_loop().time()
        logger.info(f"Waiting for task {task_id} to complete (timeout: {timeout_seconds}s)")
        
        # Wait for completion with periodic status checks
        check_interval = 1.0  # Check every second
        last_progress_log = 0  # To avoid too many logs
        
        try:
            while asyncio.get_event_loop().time() - start_time < timeout_seconds:
                task = self.tasks[task_id]
                
                # Calculate progress
                completed_chunks = sum(1 for chunk in task.chunks if chunk.status != ChunkStatus.PROCESSING and chunk.status != ChunkStatus.PENDING)
                total_chunks = len(task.chunks)
                progress_pct = (completed_chunks / total_chunks) * 100 if total_chunks > 0 else 0
                
                # Log progress every 10%
                progress_decile = int(progress_pct / 10)
                if progress_decile > last_progress_log:
                    logger.info(f"Task {task_id} progress: {progress_pct:.1f}% ({completed_chunks}/{total_chunks} chunks)")
                    last_progress_log = progress_decile
                
                # Check if all chunks are processed (either completed or failed)
                all_processed = all(
                    chunk.status != ChunkStatus.PROCESSING and chunk.status != ChunkStatus.PENDING
                    for chunk in task.chunks
                )
                
                if all_processed:
                    logger.info(f"Task {task_id} fully processed (completed: {progress_pct:.1f}%)")
                    return task
                
                # Wait before checking again
                await asyncio.sleep(check_interval)
            
            # If we're here, we timed out
            completed_chunks = sum(1 for chunk in task.chunks if chunk.status != ChunkStatus.PROCESSING and chunk.status != ChunkStatus.PENDING)
            total_chunks = len(task.chunks)
            progress_pct = (completed_chunks / total_chunks) * 100 if total_chunks > 0 else 0
            logger.warning(f"Timeout waiting for task {task_id} (progress: {progress_pct:.1f}%)")
            
            # Mark any remaining processing chunks as failed
            for chunk in self.tasks[task_id].chunks:
                if chunk.status == ChunkStatus.PROCESSING or chunk.status == ChunkStatus.PENDING:
                    chunk.status = ChunkStatus.FAILED
                    chunk.error = "Timed out waiting for processing"
            
            # Return partially completed task
            return self.tasks[task_id]
            
        except Exception as e:
            logger.error(f"Error while waiting for task {task_id}: {str(e)}")
            return self.tasks[task_id] 