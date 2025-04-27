from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.audio import router as audio_router

app = FastAPI(
    title="Audio Analysis API",
    description="API for analyzing audio files and extracting various features",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(audio_router, prefix="/api/v1", tags=["audio"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 