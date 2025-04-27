from fastapi import APIRouter, UploadFile, WebSocket, HTTPException, Form, File
from ...services.audio.task_manager import AudioTaskManager
from ...schemas.audio import AudioAnalysisRequest, AudioAnalysisResponse, AudioFeatureType
import os
import json
from tempfile import NamedTemporaryFile
import shutil
import logging
from typing import List
import uuid
import librosa

router = APIRouter()
task_manager = AudioTaskManager()
logger = logging.getLogger(__name__)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/analyze", response_model=AudioAnalysisResponse)
async def analyze_audio(
    file: UploadFile = File(...),
    feature_types: str = Form(...),
    chunk_duration: float = Form(60.0)
):
    """
    Upload and analyze an audio file.
    The analysis will be performed in chunks of 1 minute each, and results will be streamed via WebSocket.
    """
    try:
        logger.info(f"Received analysis request for file: {file.filename}")
        logger.info(f"Feature types: {feature_types}")
        
        # Parse feature types from JSON string
        try:
            feature_types_list = [AudioFeatureType(ft) for ft in json.loads(feature_types)]
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON for feature_types: {feature_types}")
            raise HTTPException(status_code=422, detail=f"Invalid feature types format: {str(e)}")
        except ValueError as e:
            logger.error(f"Invalid feature type value: {str(e)}")
            raise HTTPException(status_code=422, detail=f"Invalid feature type: {str(e)}")
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=422, detail="No file provided")
        
        # Create a unique filename to prevent conflicts
        file_extension = os.path.splitext(file.filename)[1]
        temp_file = NamedTemporaryFile(delete=False, suffix=file_extension, dir=UPLOAD_DIR)
        
        try:
            # Save the uploaded file
            with temp_file as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            logger.info(f"File saved successfully at: {temp_file.name}")
            
            # Create analysis task
            task_id = await task_manager.create_task(
                file_path=temp_file.name,
                feature_types=feature_types_list,
                chunk_duration=chunk_duration
            )
            
            logger.info(f"Analysis task created with ID: {task_id}")
            return task_manager.get_task_status(task_id)
            
        except Exception as e:
            # Clean up the temp file if task creation fails
            try:
                os.unlink(temp_file.name)
            except:
                pass
            raise e
            
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{task_id}", response_model=AudioAnalysisResponse)
async def get_analysis_status(task_id: str):
    """Get the current status of an audio analysis task"""
    task = task_manager.get_task_status(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@router.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for receiving real-time updates about the analysis"""
    await websocket.accept()
    
    task = task_manager.get_task_status(task_id)
    if not task:
        await websocket.close(code=4004, reason="Task not found")
        return
    
    try:
        # Register the WebSocket connection
        task_manager.register_client(task_id, websocket)
        
        # Send initial state
        await websocket.send_json(task.model_dump())
        
        # Keep the connection alive and handle disconnection
        while True:
            try:
                await websocket.receive_text()
            except Exception:
                break
                
    finally:
        task_manager.unregister_client(task_id, websocket)

@router.post("/analyze-batch", response_model=List[AudioAnalysisResponse])
async def analyze_audio_batch(
    files: List[UploadFile] = File(...),
    feature_types: str = Form(...),
    chunk_duration: float = Form(60.0),
    wait_for_completion: bool = Form(False)
):
    """
    Upload and analyze multiple audio files in batch.
    Each file will be processed sequentially and results will be returned as a list.
    If wait_for_completion is True, the API will wait for each analysis to complete before returning.
    """
    temp_files = []  # Track temp files for cleanup
    results = []
    
    try:
        logger.info(f"Received batch analysis request for {len(files)} files")
        logger.info(f"Feature types: {feature_types}")
        logger.info(f"Wait for completion: {wait_for_completion}")
        
        # Validate files
        if not files:
            raise HTTPException(status_code=422, detail="No files provided")
        
        # Parse feature types from JSON string
        try:
            feature_types_list = [AudioFeatureType(ft) for ft in json.loads(feature_types)]
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON for feature_types: {feature_types}")
            raise HTTPException(status_code=422, detail=f"Invalid feature types format: {str(e)}")
        except ValueError as e:
            logger.error(f"Invalid feature type value: {str(e)}")
            raise HTTPException(status_code=422, detail=f"Invalid feature type: {str(e)}")
        
        # Process each file with improved error handling
        for file in files:
            if not file.filename:
                logger.warning(f"Skipping file with no filename")
                continue
                
            logger.info(f"Processing file: {file.filename}")
            temp_file = None
            
            try:
                # Create a unique filename to prevent conflicts
                file_extension = os.path.splitext(file.filename)[1]
                temp_file = NamedTemporaryFile(delete=False, suffix=file_extension, dir=UPLOAD_DIR)
                temp_file_path = temp_file.name
                temp_files.append(temp_file_path)
                
                # Save the uploaded file with progress tracking
                logger.info(f"Saving file {file.filename} to {temp_file_path}")
                with temp_file as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                logger.info(f"File {file.filename} saved successfully at: {temp_file_path}")
                
                # Validate file is readable as audio
                try:
                    logger.info(f"Validating audio file: {temp_file_path}")
                    with open(temp_file_path, 'rb') as f:
                        # Just check if it's loadable
                        sample_audio, sr = librosa.load(temp_file_path, sr=None, duration=5)
                        if len(sample_audio) == 0:
                            raise ValueError(f"File contains no audio data")
                        logger.info(f"Audio file validated: {temp_file_path} (sr={sr}, duration={len(sample_audio)/sr:.2f}s)")
                except Exception as e:
                    logger.error(f"Invalid audio file {file.filename}: {str(e)}")
                    raise ValueError(f"Invalid audio file: {str(e)}")
                
                # Create analysis task
                logger.info(f"Creating analysis task for {file.filename}")
                task_id = await task_manager.create_task(
                    file_path=temp_file_path,
                    feature_types=feature_types_list,
                    chunk_duration=chunk_duration,
                    file_metadata={"original_filename": file.filename}
                )
                
                logger.info(f"Analysis task created with ID: {task_id}")
                
                # Wait for the analysis to complete if requested
                if wait_for_completion:
                    logger.info(f"Waiting for task {task_id} to complete...")
                    task = await task_manager.wait_for_task_completion(task_id)
                    logger.info(f"Task {task_id} completed")
                else:
                    task = task_manager.get_task_status(task_id)
                
                # Add task to results
                results.append(task)
                
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}", exc_info=True)
                # Create a failed task response with appropriate error message
                error_task = AudioAnalysisResponse(
                    task_id=str(uuid.uuid4()),
                    total_chunks=1,
                    original_filename=file.filename,
                    file_path=temp_file.name if temp_file else None,
                    chunks=[{
                        "chunk_id": 0,
                        "start_time": 0,
                        "end_time": 0,
                        "status": ChunkStatus.FAILED,
                        "error": f"Failed to process: {str(e)}"
                    }]
                )
                results.append(error_task)
        
        return results
            
    except Exception as e:
        logger.error(f"Error processing batch request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Clean up temporary files if they weren't cleaned up already
        for temp_file_path in temp_files:
            try:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    logger.info(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {temp_file_path}: {str(e)}") 