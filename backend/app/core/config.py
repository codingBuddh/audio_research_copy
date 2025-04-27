from pydantic_settings import BaseSettings
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    API_V1_PREFIX: str = os.getenv("API_V1_PREFIX", "/api/v1")
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "Audio Research API")
    BACKEND_CORS_ORIGINS: List[str] = eval(os.getenv("BACKEND_CORS_ORIGINS", '["http://localhost:5173"]'))
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./app.db")

    class Config:
        case_sensitive = True

settings = Settings() 