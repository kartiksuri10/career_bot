import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.services.fetch_jobs import fetch_jobs_from_jsearch
from backend.app.services import embeddings_manager
from app.config import DATA_PATH

# List of 10 queries
QUERIES = [
    "Flutter developer jobs in India",
    "Backend developer jobs in India",
    "AI engineer jobs in India",
    "Data scientist jobs in India",
    "Mobile app development jobs in India",
    "Internship in India",
    "DevOps engineer jobs in India",
    "Frontend developer jobs in India",
    "Cloud engineer jobs in India",
    "Machine learning jobs in India",
]

STATE_FILE = DATA_PATH.parent / "scraper_state.json"


def get_current_day_index():
    """Tracks which 5 queries should run today"""
    if STATE_FILE.exists():
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
    else:
        state = {"last_index": 0}

    last_index = state.get("last_index", 0)

    # Pick 5 queries for today
    start = last_index
    end = (last_index + 5) % len(QUERIES)

    if start < end:
        today_queries = QUERIES[start:end]
    else:
        today_queries = QUERIES[start:] + QUERIES[:end]

    # Update state
    state["last_index"] = end
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f)

    return today_queries


def run_daily_scraper():
    print(f"\n🚀 Daily Scraper started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    today_queries = get_current_day_index()

    print(f"📌 Queries for today: {today_queries}")

    for query in today_queries:
        print(f"\n🔍 Fetching jobs for: {query}")
        fetch_jobs_from_jsearch(query)

    print("\n🧠 Building embeddings...")
    embeddings_manager.main()

    print("✅ Daily scraping completed successfully.")


if __name__ == "__main__":
    run_daily_scraper()
