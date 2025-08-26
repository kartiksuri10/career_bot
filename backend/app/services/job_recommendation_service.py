# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.cluster import KMeans
# import joblib
# import numpy as np
# from typing import List, Dict, Any, Optional
# from pathlib import Path

# from app.services.embeddings_manager import EmbeddingsManager
# import logging
# logger = logging.getLogger(__name__)
# class JobRecommendationService:
#     """Service for job recommendations using embeddings and clustering"""
    
#     def __init__(self, embeddings_manager: EmbeddingsManager, 
#                  jobs_data: List[Dict], 
#                  job_embeddings: np.ndarray,
#                  cluster_model_path: Optional[Path] = None):
        
#         self.embeddings_manager = embeddings_manager
#         self.jobs_data = jobs_data
#         self.job_embeddings = job_embeddings.astype(np.float64)
        
#         # Load or create clustering model
#         if cluster_model_path and cluster_model_path.exists():
#             self.kmeans = joblib.load(cluster_model_path)
#             self.job_cluster_labels = self.kmeans.labels_
#         else:
#             self.kmeans = None
#             self.job_cluster_labels = None
    
#     def create_clusters(self, n_clusters: int = 10, save_path: Optional[Path] = None):
#         """Create job clusters using K-means"""
#         self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#         self.job_cluster_labels = self.kmeans.fit_predict(self.job_embeddings)
        
#         if save_path:
#             joblib.dump(self.kmeans, save_path)
#             logger.info(f"Cluster model saved to {save_path}")
    
#     def get_job_recommendations(self, resume_text: str, top_n: int = 10, 
#                               use_clustering: bool = True) -> List[Dict[str, Any]]:
#         """
#         Get job recommendations for a resume
        
#         Args:
#             resume_text: Preprocessed resume text
#             top_n: Number of recommendations to return
#             use_clustering: Whether to use clustering for faster search
#         """
#         # Get resume embedding
#         resume_embedding = self.embeddings_manager.get_text_embedding(resume_text)
#         resume_embedding = np.array(resume_embedding, dtype=np.float64).reshape(1, -1)
        
#         if use_clustering and self.kmeans is not None:
#             return self._get_clustered_recommendations(resume_embedding, top_n)
#         else:
#             return self._get_direct_recommendations(resume_embedding, top_n)
    
#     def _get_clustered_recommendations(self, resume_embedding: np.ndarray, top_n: int) -> List[Dict]:
#         """Get recommendations using clustering"""
#         # Predict cluster
#         resume_embedding_32 = resume_embedding.astype(np.float32)
#         predicted_cluster = self.kmeans.predict(resume_embedding_32)[0]
        
#         # Filter jobs in the predicted cluster
#         cluster_indices = np.where(self.job_cluster_labels == predicted_cluster)[0]
#         cluster_job_embeddings = self.job_embeddings[cluster_indices].astype(np.float32)
        
#         # Calculate similarities within cluster
#         similarities = cosine_similarity(resume_embedding_32, cluster_job_embeddings)[0]
        
#         # Get top matches
#         top_cluster_indices = similarities.argsort()[-top_n:][::-1]
        
#         results = []
#         for i in top_cluster_indices:
#             idx = cluster_indices[i]
#             job = self.jobs_data[idx].copy()
#             job["similarity_score"] = float(similarities[i])
#             results.append(job)
        
#         return results
    
#     def _get_direct_recommendations(self, resume_embedding: np.ndarray, top_n: int) -> List[Dict]:
#         """Get recommendations without clustering"""
#         similarities = cosine_similarity(resume_embedding, self.job_embeddings)[0]
#         top_indices = similarities.argsort()[-top_n:][::-1]
        
#         results = []
#         for idx in top_indices:
#             job = self.jobs_data[idx].copy()
#             job["similarity_score"] = float(similarities[idx])
#             results.append(job)
        
#         return results
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import joblib
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

from app.services.embeddings_manager import EmbeddingsManager
import logging

logger = logging.getLogger(__name__)

class JobRecommendationService:
    """Enhanced service for job recommendations using embeddings and clustering"""
    
    def __init__(self, embeddings_manager: EmbeddingsManager, 
                 jobs_data: List[Dict], 
                 job_embeddings: np.ndarray,
                 cluster_model_path: Optional[Path] = None):
        
        self.embeddings_manager = embeddings_manager
        self.jobs_data = jobs_data
        self.job_embeddings = job_embeddings.astype(np.float32)  # Ensure consistent dtype
        
        # Load or create clustering model
        self.kmeans = None
        self.job_cluster_labels = None
        
        if cluster_model_path and cluster_model_path.exists():
            try:
                self.kmeans = joblib.load(cluster_model_path)
                # Load cluster labels if available
                cluster_labels_path = cluster_model_path.parent / "job_cluster_labels.npy"
                if cluster_labels_path.exists():
                    self.job_cluster_labels = np.load(cluster_labels_path)
                    logger.info(f"Loaded clustering model with {len(np.unique(self.job_cluster_labels))} clusters")
                else:
                    # Predict labels if not saved
                    self.job_cluster_labels = self.kmeans.predict(self.job_embeddings)
                    logger.info("Predicted cluster labels from loaded model")
            except Exception as e:
                logger.warning(f"Failed to load clustering model: {e}")
                self.kmeans = None
                self.job_cluster_labels = None
    
    def create_clusters(self, n_clusters: int = 10, save_path: Optional[Path] = None):
        """Create job clusters using K-means"""
        try:
            logger.info(f"Creating {n_clusters} job clusters...")
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.job_cluster_labels = self.kmeans.fit_predict(self.job_embeddings)
            
            if save_path:
                joblib.dump(self.kmeans, save_path)
                # Also save cluster labels
                labels_path = save_path.parent / "job_cluster_labels.npy"
                np.save(labels_path, self.job_cluster_labels)
                logger.info(f"Cluster model and labels saved to {save_path.parent}")
                
            # Log cluster distribution
            unique, counts = np.unique(self.job_cluster_labels, return_counts=True)
            logger.info(f"Cluster distribution: {dict(zip(unique, counts))}")
            
        except Exception as e:
            logger.error(f"Error creating clusters: {e}")
            self.kmeans = None
            self.job_cluster_labels = None
    
    def get_job_recommendations(self, resume_text: str, top_n: int = 10, 
                              use_clustering: bool = True) -> List[Dict[str, Any]]:
        """
        Get job recommendations for a resume
        
        Args:
            resume_text: Preprocessed resume text
            top_n: Number of recommendations to return
            use_clustering: Whether to use clustering for faster search
        """
        try:
            # Get resume embedding
            resume_embedding = self.embeddings_manager.get_text_embedding(resume_text)
            resume_embedding = np.array(resume_embedding, dtype=np.float32).reshape(1, -1)
            
            if use_clustering and self.kmeans is not None and self.job_cluster_labels is not None:
                return self._get_clustered_recommendations(resume_embedding, top_n)
            else:
                return self._get_direct_recommendations(resume_embedding, top_n)
                
        except Exception as e:
            logger.error(f"Error getting job recommendations: {e}")
            return []
    
    def _get_clustered_recommendations(self, resume_embedding: np.ndarray, top_n: int) -> List[Dict]:
        """Get recommendations using clustering for faster search"""
        try:
            # Predict cluster for resume
            predicted_cluster = self.kmeans.predict(resume_embedding)[0]
            logger.info(f"Resume assigned to cluster {predicted_cluster}")
            
            # Get jobs in the predicted cluster
            cluster_indices = np.where(self.job_cluster_labels == predicted_cluster)[0]
            
            if len(cluster_indices) == 0:
                logger.warning(f"No jobs found in cluster {predicted_cluster}, falling back to direct search")
                return self._get_direct_recommendations(resume_embedding, top_n)
            
            cluster_job_embeddings = self.job_embeddings[cluster_indices]
            
            # Calculate similarities within cluster
            similarities = cosine_similarity(resume_embedding, cluster_job_embeddings)[0]
            
            # Get top matches within cluster
            # If cluster has fewer jobs than requested, get all of them
            actual_top_n = min(top_n, len(similarities))
            top_cluster_indices = np.argsort(similarities)[-actual_top_n:][::-1]
            
            results = []
            for i in top_cluster_indices:
                original_idx = cluster_indices[i]
                job = self.jobs_data[original_idx].copy()
                job["similarity_score"] = float(similarities[i])
                job["cluster_id"] = int(predicted_cluster)
                job["recommendation_method"] = "clustered"
                results.append(job)
            
            logger.info(f"Found {len(results)} recommendations in cluster {predicted_cluster}")
            return results
            
        except Exception as e:
            logger.error(f"Error in clustered recommendations: {e}")
            return self._get_direct_recommendations(resume_embedding, top_n)
    
    def _get_direct_recommendations(self, resume_embedding: np.ndarray, top_n: int) -> List[Dict]:
        """Get recommendations without clustering (full search)"""
        try:
            # Calculate similarities with all jobs
            similarities = cosine_similarity(resume_embedding, self.job_embeddings)[0]
            
            # Get top N matches
            top_indices = np.argsort(similarities)[-top_n:][::-1]
            
            results = []
            for idx in top_indices:
                job = self.jobs_data[idx].copy()
                job["similarity_score"] = float(similarities[idx])
                job["recommendation_method"] = "direct"
                if self.job_cluster_labels is not None:
                    job["cluster_id"] = int(self.job_cluster_labels[idx])
                results.append(job)
            
            logger.info(f"Found {len(results)} recommendations using direct search")
            return results
            
        except Exception as e:
            logger.error(f"Error in direct recommendations: {e}")
            return []
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get information about the clustering model"""
        if self.kmeans is None or self.job_cluster_labels is None:
            return {"clustering_available": False}
        
        unique, counts = np.unique(self.job_cluster_labels, return_counts=True)
        
        return {
            "clustering_available": True,
            "n_clusters": len(unique),
            "cluster_distribution": dict(zip(unique.tolist(), counts.tolist())),
            "total_jobs": len(self.job_cluster_labels)
        }
    
    def search_by_keywords(self, keywords: List[str], top_n: int = 10) -> List[Dict[str, Any]]:
        """Search jobs by keywords"""
        try:
            # Create a query from keywords
            query = " ".join(keywords)
            query_embedding = self.embeddings_manager.get_text_embedding(query)
            query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
            
            return self._get_direct_recommendations(query_embedding, top_n)
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []