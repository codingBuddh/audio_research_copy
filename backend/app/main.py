import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from .api.v1 import audio

# Load environment variables
load_dotenv()

# Configure logging
def configure_logging():
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "app.log")),
            logging.StreamHandler()
        ]
    )
    
    # Suppress noisy libraries
    logging.getLogger("whisper").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("librosa").setLevel(logging.WARNING)
    
    # Create app logger
    logger = logging.getLogger("audio_analyzer")
    logger.setLevel(logging.INFO)
    
    return logger

# Initialize logger
logger = configure_logging()

app = FastAPI(
    title=os.getenv("PROJECT_NAME", "Audio Research API"),
    openapi_url=f"{os.getenv('API_V1_PREFIX', '/api/v1')}/openapi.json"
)

# Configure CORS
origins = eval(os.getenv("BACKEND_CORS_ORIGINS", '["http://localhost:5173"]'))
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    audio.router,
    prefix=os.getenv("API_V1_PREFIX", "/api/v1"),
    tags=["audio"]
)

@app.get("/")
async def root():
    return {"message": "Welcome to Audio Research API"}

@app.on_event("startup")
async def startup_event():
    logger.info("Application starting up")
    
    # Ensure upload directory exists
    upload_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    logger.info(f"Upload directory: {upload_dir}")
    
    # Ensure models directory exists
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    logger.info(f"Models directory: {models_dir}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutting down")

# Import and include routers here
# from app.api.v1 import some_router
# app.include_router(some_router.router, prefix=os.getenv("API_V1_PREFIX", "/api/v1")) 