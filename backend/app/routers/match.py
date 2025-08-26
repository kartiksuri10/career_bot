# #match.py
# from fastapi import APIRouter, UploadFile, File, Form
# from app.utils.extract_resume import extract_text_from_pdf, preprocess

# from app.config import DATA_PATH
# import os

# from backend.app.services.query import JobProfileQueryGenerator

# router = APIRouter()

# @router.post("/match-resume")
# async def match_resume(file: UploadFile = File(...), top_n: int = Form(5)):
#     # Save uploaded file temporarily
#     file_location = os.path.join(DATA_PATH, file.filename)
#     with open(file_location, "wb") as f:
#         f.write(await file.read())
        
#     profile_generator = JobProfileQueryGenerator(
#         embeddings_path=DATA_PATH / "job_profile_embeddings.npy",
#         jobs_data_path=DATA_PATH / "job_postings.json"
#     )
#     profile_results = profile_generator.generate_query_from_resume(file_location, top_k=3)
#     return {
#         "profiles": profile_results.get("alternatives", []),
#         "best_profile": profile_results.get("best_profile"),
#         "query_confidence": profile_results.get("query_confidence"),
        
#     }
# match.py - Streamlined version
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.utils.extract_resume import extract_text_from_pdf, preprocess
from app.services.job_recommendation_service import JobRecommendationService
from app.services.embeddings_manager import EmbeddingsManager
from app.services.query import JobProfileQueryGenerator

from app.config import DATA_PATH
import os
import json
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services at startup
def initialize_services():
    """Initialize all services with proper error handling"""
    try:
        # Load job data
        jobs_data_path = DATA_PATH / "job_postings.json"
        with open(jobs_data_path, "r", encoding="utf-8") as f:
            jobs_data = json.load(f)
        
        # Initialize embeddings manager
        embeddings_manager = EmbeddingsManager()
        
        # Load job embeddings
        job_embeddings_path = DATA_PATH / "job_embeddings.npy"
        if not job_embeddings_path.exists():
            # Create embeddings if they don't exist
            job_embeddings = embeddings_manager.build_job_embeddings(
                jobs_data, 
                output_path=job_embeddings_path
            )
        else:
            job_embeddings = np.load(job_embeddings_path)
        
        # Initialize recommendation service
        recommendation_service = JobRecommendationService(
            embeddings_manager=embeddings_manager,
            jobs_data=jobs_data,
            job_embeddings=job_embeddings,
            cluster_model_path=DATA_PATH / "kmeans_model.joblib"
        )
        
        # Create clusters if they don't exist
        if recommendation_service.kmeans is None:
            recommendation_service.create_clusters(
                n_clusters=10, 
                save_path=DATA_PATH / "kmeans_model.joblib"
            )
        
        # Initialize profile query generator
        profile_generator = JobProfileQueryGenerator(
            embeddings_path=DATA_PATH / "job_profile_embeddings.npy",
            jobs_data_path=jobs_data_path
        )
        
        return recommendation_service, profile_generator
        
    except Exception as e:
        logger.error(f"Error initializing services: {str(e)}")
        raise

# Global service instances
try:
    recommendation_service, profile_generator = initialize_services()
except Exception as e:
    logger.error(f"Failed to initialize services: {e}")
    recommendation_service = None
    profile_generator = None

@router.post("/analyze-resume")
async def analyze_resume(file: UploadFile = File(...)):
    """
    Step 1: Analyze resume and suggest job profiles
    """
    if not profile_generator:
        raise HTTPException(status_code=500, detail="Profile generator service not available")
    
    try:
        # Save uploaded file temporarily
        file_location = os.path.join(DATA_PATH, file.filename)
        with open(file_location, "wb") as f:
            f.write(await file.read())
        
        # Generate job profile suggestions
        profile_results = profile_generator.generate_query_from_resume(file_location, top_k=5)
        
        # Clean up temporary file
        os.remove(file_location)
        
        if not profile_results.get("success"):
            raise HTTPException(status_code=400, detail=profile_results.get("error", "Failed to analyze resume"))
        
        return {
            "success": True,
            "profiles": profile_results.get("alternatives", []),
            "best_profile": profile_results.get("best_profile"),
            "best_match_details": profile_results.get("best_match_details"),
            "query_confidence": profile_results.get("query_confidence"),
        }
        
    except Exception as e:
        logger.error(f"Error in analyze_resume: {str(e)}")
        # Clean up file if it exists
        if 'file_location' in locals() and os.path.exists(file_location):
            os.remove(file_location)
        raise HTTPException(status_code=500, detail=f"Error analyzing resume: {str(e)}")

@router.post("/recommend-jobs")
async def recommend_jobs(
    file: UploadFile = File(...), 
    job_query: str = Form(...),
    top_n: int = Form(10),
    use_clustering: bool = Form(True)
):
    """
    Step 2: Get job recommendations based on resume and selected job profile
    """
    if not recommendation_service:
        raise HTTPException(status_code=500, detail="Recommendation service not available")
    
    try:
        # Save uploaded file temporarily
        file_location = os.path.join(DATA_PATH, file.filename)
        with open(file_location, "wb") as f:
            f.write(await file.read())
        
        # Extract and preprocess resume text
        resume_text = preprocess(file_location)
        
        # Enhance resume text with job query for better matching
        enhanced_resume_text = f"{job_query} {resume_text}"
        
        # Get job recommendations
        matches = recommendation_service.get_job_recommendations(
            resume_text=enhanced_resume_text,
            top_n=top_n,
            use_clustering=use_clustering
        )
        
        # Clean up temporary file
        os.remove(file_location)
        
        return {
            "success": True,
            "job_query": job_query,
            "matches": matches,
            "total_matches": len(matches),
            "clustering_used": use_clustering
        }
        
    except Exception as e:
        logger.error(f"Error in recommend_jobs: {str(e)}")
        # Clean up file if it exists
        if 'file_location' in locals() and os.path.exists(file_location):
            os.remove(file_location)
        raise HTTPException(status_code=500, detail=f"Error getting job recommendations: {str(e)}")

@router.post("/quick-recommend")
async def quick_recommend(
    file: UploadFile = File(...),
    top_n: int = Form(5)
):
    """
    Single-step recommendation: analyze resume and return jobs in one call
    """
    if not recommendation_service or not profile_generator:
        raise HTTPException(status_code=500, detail="Services not available")
    
    try:
        # Save uploaded file temporarily
        file_location = os.path.join(DATA_PATH, file.filename)
        with open(file_location, "wb") as f:
            f.write(await file.read())
        
        # Step 1: Get best job profile
        profile_results = profile_generator.generate_query_from_resume(file_location, top_k=3)
        
        if not profile_results.get("success"):
            raise HTTPException(status_code=400, detail="Failed to analyze resume")
        
        # Step 2: Get job recommendations using the best profile
        resume_text = preprocess(file_location)
        best_profile = profile_results.get("best_profile", "")
        enhanced_resume_text = f"{best_profile} {resume_text}"
        
        matches = recommendation_service.get_job_recommendations(
            resume_text=enhanced_resume_text,
            top_n=top_n,
            use_clustering=True
        )
        
        # Clean up temporary file
        os.remove(file_location)
        
        return {
            "success": True,
            "detected_profile": best_profile,
            "profile_confidence": profile_results.get("query_confidence"),
            "matches": matches,
            "alternative_profiles": profile_results.get("alternatives", [])
        }
        
    except Exception as e:
        logger.error(f"Error in quick_recommend: {str(e)}")
        # Clean up file if it exists
        if 'file_location' in locals() and os.path.exists(file_location):
            os.remove(file_location)
        raise HTTPException(status_code=500, detail=f"Error in quick recommendation: {str(e)}")

@router.get("/health")
async def health_check():
    """Check if all services are properly initialized"""
    return {
        "recommendation_service": recommendation_service is not None,
        "profile_generator": profile_generator is not None,
        "data_path_exists": DATA_PATH.exists(),
        "services_ready": all([recommendation_service, profile_generator])
    }