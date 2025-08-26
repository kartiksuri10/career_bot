import requests
import pandas as pd
import json
from app.config import RAPIDAPI_KEY, RAPIDAPI_HOST, DATA_PATH
from datetime import datetime, timezone
JOBS_FILE = DATA_PATH / "job_postings.json"
DATA_PATH.mkdir(parents=True, exist_ok=True)

def fetch_jobs_from_jsearch(query):
    page = 1
    num_pages = 50
    url = "https://jsearch.p.rapidapi.com/search"
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST,
    }
    querystring = {
        "query": query,
        "page": str(page),
        "num_pages": str(num_pages),
        "date_posted": "month",
        "country": "in",
        "language": "en"
    }
    response = requests.get(url, headers=headers, params=querystring)
    if response.status_code != 200:
        print("Error:", response.text)
        return

    new_results = response.json().get("data", [])

    if JOBS_FILE.exists():
        with open(JOBS_FILE, "r", encoding="utf-8") as f:
            old_data = json.load(f)
    else:
        old_data = []

    existing_job_ids = set(job.get("job_id") for job in old_data)
    unique_new_jobs = [
        extract_relevant_fields(job)
        for job in new_results
        if job.get("job_id") not in existing_job_ids
    ]

    combined_data = old_data + unique_new_jobs

    with open(JOBS_FILE, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Appended {len(unique_new_jobs)} jobs. Total now: {len(combined_data)}")


def extract_relevant_fields(job_raw):
    apply_links = job_raw.get("apply_options") or []

    preferred_link = next(
        (
            opt.get("apply_link")
            for opt in apply_links
            if opt.get("publisher") and "linkedin" in opt["publisher"].lower()
        ),
        next(
            (
                opt.get("apply_link")
                for opt in apply_links
                if opt.get("publisher") and "glassdoor" in opt["publisher"].lower()
            ),
            job_raw.get("job_apply_link")  # fallback
        )
    )

    return {
        "job_id": job_raw.get("job_id", ""),
        "title": job_raw.get("job_title", ""),
        "company": job_raw.get("employer_name", ""),
        "location": job_raw.get("job_location", ""),
        "description": job_raw.get("job_description", ""),
        "employment_type": job_raw.get("job_employment_type", ""),
        "apply_link": preferred_link or "",
        "posted_at": job_raw.get("job_posted_at_datetime_utc", ""),
        "fetched_at": datetime.now(timezone.utc).isoformat()
    }


if __name__ == "__main__":
    fetch_jobs_from_jsearch("machine learning engineer jobs in India")
