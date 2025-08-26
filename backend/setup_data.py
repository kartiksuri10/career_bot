# setup_data.py - Setup required data files and embeddings
import json
import numpy as np
from pathlib import Path
from app.config import DATA_PATH
from app.services.embeddings_manager import EmbeddingsManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_data():
    """Setup all required data files and embeddings"""
    
    # Ensure data directory exists
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    
    jobs_file = DATA_PATH / "job_postings.json"
    job_embeddings_file = DATA_PATH / "job_embeddings.npy"
    profile_embeddings_file = DATA_PATH / "job_profile_embeddings.npy"
    
    # Check if job postings exist
    if not jobs_file.exists():
        logger.error(f"Job postings file not found: {jobs_file}")
        logger.info("Please run fetch_jobs.py first to download job data")
        return False
    
    # Load job data
    with open(jobs_file, "r", encoding="utf-8") as f:
        jobs_data = json.load(f)
    
    logger.info(f"Loaded {len(jobs_data)} job postings")
    
    # Initialize embeddings manager
    embeddings_manager = EmbeddingsManager()
    
    # Create job embeddings if they don't exist
    if not job_embeddings_file.exists():
        logger.info("Creating job embeddings (for full job matching)...")
        job_embeddings = embeddings_manager.build_job_embeddings(
            jobs_data,
            fields=['title', 'company', 'location', 'description', 'employment_type'],
            output_path=job_embeddings_file
        )
        logger.info(f"Job embeddings saved to {job_embeddings_file}")
    else:
        logger.info("Job embeddings already exist")
        job_embeddings = np.load(job_embeddings_file)
    
    # Create profile embeddings if they don't exist  
    if not profile_embeddings_file.exists():
        logger.info("Creating profile embeddings (for job profile detection)...")
        profile_embeddings = embeddings_manager.build_profile_embeddings(
            jobs_data,
            output_path=profile_embeddings_file
        )
        logger.info(f"Profile embeddings saved to {profile_embeddings_file}")
    else:
        logger.info("Profile embeddings already exist")
    
    # Create clustering model if it doesn't exist
    cluster_model_file = DATA_PATH / "kmeans_model.joblib"
    cluster_labels_file = DATA_PATH / "job_cluster_labels.npy"
    
    if not cluster_model_file.exists():
        logger.info("Creating clustering model...")
        from sklearn.cluster import KMeans
        import joblib
        
        # Use the job embeddings for clustering
        kmeans = KMeans(n_clusters=10, random_state=42)
        cluster_labels = kmeans.fit_predict(job_embeddings)
        
        # Save model and labels
        joblib.dump(kmeans, cluster_model_file)
        np.save(cluster_labels_file, cluster_labels)
        
        logger.info(f"Clustering model saved to {cluster_model_file}")
        logger.info(f"Cluster labels saved to {cluster_labels_file}")
    else:
        logger.info("Clustering model already exists")
    
    logger.info("✅ Data setup completed successfully!")
    logger.info("\nFiles created:")
    logger.info(f"  - Job embeddings: {job_embeddings_file}")
    logger.info(f"  - Profile embeddings: {profile_embeddings_file}")
    logger.info(f"  - Clustering model: {cluster_model_file}")
    logger.info(f"  - Cluster labels: {cluster_labels_file}")
    
    return True

def validate_setup():
    """Validate that all required files exist"""
    required_files = [
        DATA_PATH / "job_postings.json",
        DATA_PATH / "job_embeddings.npy",
        DATA_PATH / "job_profile_embeddings.npy",
        DATA_PATH / "kmeans_model.joblib",
        DATA_PATH / "job_cluster_labels.npy"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not file_path.exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.error("❌ Missing required files:")
        for file_path in missing_files:
            logger.error(f"  - {file_path}")
        return False
    else:
        logger.info("✅ All required files exist")
        return True

if __name__ == "__main__":
    print("🚀 Setting up Career Bot data files...")
    
    if setup_data():
        print("\n🎉 Setup completed successfully!")
        print("\nYou can now start the FastAPI server:")
        print("  uvicorn main:app --reload")
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        print("\n📋 Quick setup guide:")
        print("1. Run fetch_jobs.py to download job data")
        print("2. Run setup_data.py to create embeddings")
        print("3. Start the FastAPI server")