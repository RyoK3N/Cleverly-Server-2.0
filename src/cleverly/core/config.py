import os

class Config:
    HOST = os.getenv("API_HOST", "0.0.0.0")
    PORT = int(os.getenv("API_PORT", 8000))
    DEBUG = os.getenv("FLASK_ENV", "production") == "development"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
