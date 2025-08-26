from sentence_transformers import SentenceTransformer
# import pandas as pd
import numpy as np
from tqdm import tqdm
# import os
# from app.config import DATA_PATH
# import json
from typing import List, Dict, Optional
from pathlib import Path
class EmbeddingsManager:
    def __init__(self, model_name = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text"""
        return self.model.encode([text])[0]
    
    def build_job_embeddings(self, jobs_data: List[Dict], 
                           fields: List[str] = None,
                           output_path: Optional[Path] = None) -> np.ndarray:
        if fields is None:
            fields = ['title', 'company', 'location', 'description', 'employment_type']
        combined_texts = []
        for job in tqdm(jobs_data):
            text_parts = []
            for field in fields:
                if field in job and job[field]:
                    text_parts.append(str(job[field]))
            combined_texts.append(' '.join(text_parts))
        
        # logger.info(f"Building embeddings for {len(combined_texts)} jobs using fields: {fields}")
        embeddings = self.model.encode(combined_texts, show_progress_bar=True, convert_to_numpy=True)
        
        if output_path:
            np.save(output_path, embeddings)
            # logger.info(f"Embeddings saved to {output_path}")
        
        return embeddings
    
    def build_profile_embeddings(self, jobs_data: List[Dict],
                               output_path: Optional[Path] = None) -> np.ndarray:
        """
        Build embeddings specifically for job profile matching (title + description only)
        """
        return self.build_job_embeddings(
            jobs_data, 
            fields=['title', 'description'],
            output_path=output_path
        )
# JOBS_FILE = DATA_PATH / "job_postings.json"
# tqdm.pandas()



# def main():
#     with open(JOBS_FILE, "r", encoding="utf-8") as f:
#         job_data = json.load(f)

#     df = pd.DataFrame(job_data)

#     model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

#     df["combined_text"] = (
#         df["title"].fillna("") + " " +
#         df["company"].fillna("") + " " +
#         df["location"].fillna("") + " " +
#         df["description"].fillna("") + " " +
#         df["employment_type"].fillna("")
#     )

#     embeddings = model.encode(df['combined_text'].tolist(), show_progress_bar=True, convert_to_numpy=True)
#     np.save(DATA_PATH/"job_embeddings.npy", embeddings)
#     print("✅ Embeddings updated successfully.")

# if __name__ == "__main__":
#     main()