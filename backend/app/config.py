import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
RAPIDAPI_HOST = "jsearch.p.rapidapi.com"
DATA_PATH = Path(__file__).resolve().parent.parent / "data"