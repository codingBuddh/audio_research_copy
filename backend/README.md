# Backend Setup Instructions

## Prerequisites

1. Install FFmpeg (required for Whisper):
   ```bash
   # On macOS using Homebrew
   brew install ffmpeg

   # On Ubuntu/Debian
   sudo apt update && sudo apt install ffmpeg

   # On Windows using Chocolatey
   choco install ffmpeg
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Unix/macOS
   # or
   .\venv\Scripts\activate  # On Windows
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Server

1. Start the FastAPI server:
   ```bash
   uvicorn app.main:app --reload
   ```

The server will start on `http://localhost:8000`

## Features

- Audio analysis in chunks
- Real-time processing with WebSocket updates
- Transcription using Whisper (tiny.en model - optimized for English)
- Acoustic and paralinguistic feature extraction
- Progress tracking and error handling 