# import os
# class Config:
#     HOST="0.0.0.0"
#     PORT=5000
#     UPLOAD_FOLDER="uploads"
#     LLM_API_URL="https://api.openai.com/v1/chat/completions"
#     LLM_API_KEY="YOUR_API_KEY_HERE"
#     RANDOM_SEED=42
#     MODEL_PATH="models"


import os

class Config:
    # Flask
    HOST = os.getenv("FLASK_HOST", "0.0.0.0")
    PORT = int(os.getenv("FLASK_PORT", "5000"))
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")

    # ML / Misc
    RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
    MODEL_PATH = os.getenv("MODEL_PATH", "models")

    # LLM: optional external integration (set OPENAI_API_KEY to enable)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
