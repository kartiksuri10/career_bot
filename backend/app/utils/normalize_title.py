import json
from collections import defaultdict
from app.config import DATA_PATH
# Input and output file paths
input_file = DATA_PATH/"knowledge_base_from_csv.json"
output_file = DATA_PATH/"cleaned_knowledge_base.json"

# Load JSON
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Dictionary to store normalized titles and merged keywords
merged_data = defaultdict(set)

# Normalize titles and merge keywords
for title, keywords in data.items():
    normalized_title = title.strip().lower()
    
    merged_data[normalized_title].update(kw.strip().lower() for kw in keywords)

# Convert to dict with sorted keyword lists
normalized_data = {title: sorted(list(keywords)) for title, keywords in merged_data.items()}

# Save normalized data
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(normalized_data, f, indent=4, ensure_ascii=False)

print(f"Normalization complete. Output saved to {output_file}")
