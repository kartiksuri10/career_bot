from pathlib import Path
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional
import logging

from app.services.embeddings_manager import EmbeddingsManager
from app.utils import extract_resume

logger = logging.getLogger(__name__)

class JobProfileQueryGenerator:
    """
    Generate best matching job profile query from resume using pre-computed embeddings
    """
    
    def __init__(self, embeddings_path: Path, jobs_data_path: Path, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize with paths to saved embeddings and job data
        
        Args:
            embeddings_path: Path to saved job profile embeddings (.npy file)
            jobs_data_path: Path to job data JSON file
            model_name: SentenceTransformer model name
        """
        self.embeddings_path = embeddings_path
        self.jobs_data_path = jobs_data_path
        self.embeddings_manager = EmbeddingsManager(model_name)
        
        # Load job data and embeddings
        self.jobs_data = self._load_jobs_data()
        self.job_profile_embeddings = self._load_or_create_embeddings()
        
        # Extract unique job titles for profile matching
        self.unique_job_profiles = self._extract_unique_profiles()
    
    def _load_jobs_data(self) -> List[Dict]:
        """Load job data from JSON file"""
        try:
            with open(self.jobs_data_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Job data file not found: {self.jobs_data_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in job data file: {self.jobs_data_path}")
            raise
    
    def _load_or_create_embeddings(self) -> np.ndarray:
        """Load existing embeddings or create new ones"""
        if self.embeddings_path.exists():
            logger.info(f"Loading existing embeddings from {self.embeddings_path}")
            return np.load(self.embeddings_path)
        else:
            logger.info("Creating new job profile embeddings...")
            return self._create_and_save_embeddings()
    
    def _create_and_save_embeddings(self) -> np.ndarray:
        """Create and save job profile embeddings"""
        # Create embeddings for job profiles (title + description only)
        embeddings = self.embeddings_manager.build_profile_embeddings(
            self.jobs_data,
            output_path=self.embeddings_path
        )
        logger.info(f"Job profile embeddings saved to {self.embeddings_path}")
        return embeddings
    
    def _extract_unique_profiles(self) -> List[str]:
        """Extract unique job profiles from job titles"""
        job_titles = [job.get("title", "").strip() for job in self.jobs_data]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_profiles = []
        for title in job_titles:
            if title and title.lower() not in seen:
                unique_profiles.append(title)
                seen.add(title.lower())
        
        return unique_profiles
    
    def generate_query_from_resume(self, resume_path: str, top_k: int = 3) -> Dict:
        """
        Generate job profile query from resume
        
        Args:
            resume_path: Path to resume PDF
            top_k: Number of top matching profiles to consider
        
        Returns:
            Dictionary containing the best matching job profile and alternatives
        """
        try:
            # Extract and preprocess resume
            resume_text = extract_resume.preprocess(resume_path)
            
            if not resume_text or len(resume_text.strip()) < 10:
                return {
                    "error": "Could not extract meaningful content from resume",
                    "best_profile": None,
                    "alternatives": []
                }
            
            # Get resume embedding
            resume_embedding = self.embeddings_manager.get_text_embedding(resume_text)
            resume_embedding = resume_embedding.reshape(1, -1)
            
            # Calculate similarities with all job profiles
            similarities = cosine_similarity(resume_embedding, self.job_profile_embeddings)[0]
            
            # Get top k matches
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # Prepare results
            matches = []
            for idx in top_indices:
                job = self.jobs_data[idx]
                matches.append({
                    "job_title": job.get("title", ""),
                    "company": job.get("company", ""),
                    "similarity_score": float(similarities[idx]),
                    "confidence": self._calculate_confidence(similarities[idx])
                })
            
            # Best match is the first one
            best_match = matches[0] if matches else None
            
            return {
                "success": True,
                "best_profile": best_match["job_title"] if best_match else None,
                "best_match_details": best_match,
                "alternatives": matches[1:] if len(matches) > 1 else [],
                "query_confidence": best_match["confidence"] if best_match else "low"
            }
            
        except Exception as e:
            logger.error(f"Error generating query from resume: {str(e)}")
            return {
                "error": f"Processing error: {str(e)}",
                "best_profile": None,
                "alternatives": []
            }
    
    def generate_query_from_text(self, user_input: str, top_k: int = 3) -> Dict:
        """
        Generate job profile query from user text input
        
        Args:
            user_input: User's description of desired job
            top_k: Number of top matching profiles to return
        
        Returns:
            Dictionary containing matching job profiles
        """
        try:
            if not user_input or len(user_input.strip()) < 5:
                return {
                    "error": "Please provide more detailed job description",
                    "best_profile": None,
                    "alternatives": []
                }
            
            # Get embedding for user input
            input_embedding = self.embeddings_manager.get_text_embedding(user_input)
            input_embedding = input_embedding.reshape(1, -1)
            
            # Calculate similarities
            similarities = cosine_similarity(input_embedding, self.job_profile_embeddings)[0]
            
            # Get top k matches
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            matches = []
            for idx in top_indices:
                job = self.jobs_data[idx]
                matches.append({
                    "job_title": job.get("title", ""),
                    "similarity_score": float(similarities[idx]),
                    "confidence": self._calculate_confidence(similarities[idx])
                })
            
            best_match = matches[0] if matches else None
            
            return {
                "success": True,
                "best_profile": best_match["job_title"] if best_match else None,
                "best_match_details": best_match,
                "alternatives": matches[1:] if len(matches) > 1 else [],
                "query_confidence": best_match["confidence"] if best_match else "low"
            }
            
        except Exception as e:
            logger.error(f"Error generating query from text: {str(e)}")
            return {
                "error": f"Processing error: {str(e)}",
                "best_profile": None,
                "alternatives": []
            }
    
    def _calculate_confidence(self, similarity_score: float) -> str:
        """Calculate confidence level based on similarity score"""
        if similarity_score >= 0.8:
            return "very_high"
        elif similarity_score >= 0.6:
            return "high"
        elif similarity_score >= 0.4:
            return "medium"
        elif similarity_score >= 0.2:
            return "low"
        else:
            return "very_low"
    
    def get_all_available_profiles(self) -> List[str]:
        """Get all unique job profiles available"""
        return self.unique_job_profiles.copy()
    
    def refresh_embeddings(self) -> bool:
        """Recreate embeddings from current job data"""
        try:
            # Reload job data
            self.jobs_data = self._load_jobs_data()
            
            # Recreate embeddings
            self.job_profile_embeddings = self._create_and_save_embeddings()
            
            # Update unique profiles
            self.unique_job_profiles = self._extract_unique_profiles()
            
            logger.info("Embeddings refreshed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error refreshing embeddings: {str(e)}")
            return False


# Usage example and helper functions
def setup_job_profile_generator(data_path: Path) -> JobProfileQueryGenerator:
    """
    Setup job profile generator with default paths
    
    Args:
        data_path: Base data directory path
    
    Returns:
        Initialized JobProfileQueryGenerator
    """
    jobs_data_path = data_path / "job_postings.json"
    embeddings_path = data_path / "job_profile_embeddings.npy"
    
    return JobProfileQueryGenerator(embeddings_path, jobs_data_path)


def main():
    """Example usage"""
    from app.config import DATA_PATH
    
    # Setup generator
    generator = setup_job_profile_generator(DATA_PATH)
    
    # Example 1: Generate query from resume
    resume_path = r"C:\Users\karti\Downloads\sakku_resume.pdf"
    
    print("=== GENERATING JOB PROFILE QUERY FROM RESUME ===")
    result = generator.generate_query_from_resume(resume_path, top_k=3)
    
    if result.get("success"):
        print(f"Best Job Profile: '{result['best_profile']}'")
        print(f"Confidence: {result['query_confidence']}")
        
        if result.get('best_match_details'):
            details = result['best_match_details']
            print(f"Similarity Score: {details['similarity_score']:.3f}")
        
        if result.get('alternatives'):
            print("\nAlternative profiles:")
            for i, alt in enumerate(result['alternatives'], 1):
                print(f"  {i}. {alt['job_title']} (Score: {alt['similarity_score']:.3f})")
    else:
        print(f"Error: {result.get('error')}")
    
    print("\n" + "="*50)
    
    # Example 2: Generate query from text description
    user_input = "I want to work with machine learning and artificial intelligence, building ML models"
    
    print("=== GENERATING JOB PROFILE QUERY FROM TEXT ===")
    result = generator.generate_query_from_text(user_input, top_k=3)
    
    if result.get("success"):
        print(f"Best Job Profile: '{result['best_profile']}'")
        print(f"Confidence: {result['query_confidence']}")
        
        if result.get('alternatives'):
            print("\nAlternative profiles:")
            for i, alt in enumerate(result['alternatives'], 1):
                print(f"  {i}. {alt['job_title']} (Score: {alt['similarity_score']:.3f})")
    else:
        print(f"Error: {result.get('error')}")


if __name__ == "__main__":
    main()