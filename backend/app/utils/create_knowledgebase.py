import pandas as pd
import re
from app.config import DATA_PATH
import json
import spacy
import unicodedata
from tqdm import tqdm
from pathlib import Path

# File paths
json_path = DATA_PATH / 'knowledge_base_from_csv.json'
csv_path = DATA_PATH / 'it_job_profiles.csv'  # <- your new CSV

# Load existing knowledge base if it exists
if Path(json_path).exists():
    with open(json_path, 'r', encoding='utf-8') as f:
        knowledge_base = json.load(f)
    # Convert lists back to sets for merging
    knowledge_base = {k: set(v) for k, v in knowledge_base.items()}
else:
    knowledge_base = {}

# Load new CSV
df = pd.read_csv(csv_path, encoding='utf-8')
df['Keywords'] = df['skills'].fillna('')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    text = unicodedata.normalize("NFKD", text)
    return re.sub(r'\s+', ' ', text).strip()

# Process new CSV data
for _, row in tqdm(df.iterrows(), total=len(df), desc="Appending job profiles"):
    job_title = row['title'].strip().lower()
    keywords_text = row['Keywords']

    raw_keywords = re.split(r',\s*', keywords_text)
    processed_keywords = []

    for skill in raw_keywords:
        if skill:
            doc = nlp(skill.lower())
            lemmatized_skill = " ".join([token.lemma_ for token in doc])
            lemmatized_skill = clean_text(lemmatized_skill)
            processed_keywords.append(lemmatized_skill)

    if job_title not in knowledge_base:
        knowledge_base[job_title] = set()
    knowledge_base[job_title].update(processed_keywords)

# Convert sets to sorted lists for saving
knowledge_base = {k: sorted(v) for k, v in knowledge_base.items()}

# Save updated JSON
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(knowledge_base, f, indent=2, ensure_ascii=False)

print("\n✅ Successfully appended to knowledge_base_from_csv.json!")
