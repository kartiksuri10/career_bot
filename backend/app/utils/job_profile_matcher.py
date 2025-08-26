from typing import List, Dict, Any
from app.utils import extract_resume
import re
import json
from app.config import DATA_PATH
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import yake  # For keyword extraction
from datetime import datetime
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)
# Load lightweight English model
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

class JobProfileMatcher:
    def __init__(self, jobs_data: List[Dict]):
        self.jobs_data = jobs_data
        self.job_titles = [job.get("title", "") for job in jobs_data]
        self.job_skills = {}
        self.vectorizer = None
        self.job_vectors = None
        self.kw_extractor = yake.KeywordExtractor(top=20, stopwords=None)
        
        self._prepare_job_database()
    
    def _prepare_job_database(self):
        """Prepare job database using only title and description for profile matching"""
        job_descriptions = []
        
        for i, job in enumerate(self.jobs_data):
            title = job.get("title", "").strip()
            description = job.get("description", "").strip()
            
            # Clean and prepare text
            cleaned_desc = self._clean_text(description)
            cleaned_title = self._clean_text(title)
            
            # Weight title higher for profile matching
            job_text = f"{cleaned_title} {cleaned_title} {cleaned_title} {cleaned_desc}"
            
            # Extract skills from description
            skill_set = self._extract_skills_from_text(description)
            self.job_skills[title] = skill_set
            
            job_descriptions.append(job_text)
        
        # Fit TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.8
        )
        self.job_vectors = self.vectorizer.fit_transform(job_descriptions)

    def _extract_skills_from_text(self, text: str) -> set:
        """Extract skills/keywords from job description"""
        skill_set = set()
        if not text:
            return skill_set
            
        try:
            keywords = self.kw_extractor.extract_keywords(text)
            for kw, _ in keywords:
                skill_set.add(self._clean_text(kw))
        except Exception as e:
            logger.warning(f"Keyword extraction error: {e}")
        
        return skill_set
    
    def _clean_text(self, text):
        """Fast text cleaning with regex"""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        return text.strip()
    
    def _lemmatize_text(self, text):
        """Efficient lemmatization with spaCy"""
        if not nlp or not text:
            return text
        doc = nlp(text)
        return " ".join([token.lemma_ for token in doc if token.is_alpha])
    
    def _extract_resume_features(self, sections):
        """Extract and weight features with keyword boosting"""
        features = []
        full_text = []
        
        # Process sections with weights
        section_weights = {
            'skills': 3,
            'work_experience': 2,
            'education': 1,
            'certifications': 1,
            'projects': 1
        }
        
        for section, weight in section_weights.items():
            if section in sections and sections[section]:
                text = self._clean_text(sections[section])
                lemmatized = self._lemmatize_text(text)
                features.extend([lemmatized] * weight)
                full_text.append(text)
        
        # Add YAKE keywords with high weight
        combined_text = " ".join(full_text)
        if combined_text:
            try:
                keywords = self.kw_extractor.extract_keywords(combined_text)
                keyword_string = " ".join([kw[0] for kw in keywords])
                features.extend([keyword_string] * 2)  # Boost keywords
            except Exception as e:
                logger.warning(f"Resume keyword extraction error: {e}")
        
        return " ".join(features)
    
    
    def _calculate_skill_overlap(self, resume_keywords: set, job_title: str) -> float:
        """Calculate skill overlap percentage"""
        job_skills = self.job_skills.get(job_title, set())
        if not job_skills:
            return 0.0
        
        # Count matches
        matches = sum(1 for skill in job_skills if skill in resume_keywords)
        return matches / len(job_skills)
    def _get_year_of_completion(self,education_text):
        if not education_text:
            return None

        ignorePatterns = [r"10th", r"12th", r"ssc", r"hsc", r"secondary school", r"high school"]
        lines = education_text.lower().split("\n")
        filtered_lines = [
            line for line in lines
            if not any(re.search(pat, line) for pat in ignorePatterns)
        ]

        current_year = datetime.now().year
        max_year = current_year + 5

        # Allow either a year or the word 'present'
        range_pattern = re.compile(
            rf"(19[8-9]\d|20[0-9]\d|{max_year})\s*[-–]\s*(19[8-9]\d|20[0-9]\d|{max_year}|present)",
            re.IGNORECASE
        )
        year_pattern = re.compile(rf"\b(19[8-9]\d|20[0-9]\d|{max_year})\b")

        years_found = []

        for line in filtered_lines:
            # First, check for year ranges like '2022 - present'
            for start, end in range_pattern.findall(line):
                if end.lower() == "present":
                    return current_year
                else:
                    years_found.append(int(end))

            # Then, check for standalone years
            for match in year_pattern.findall(line):
                year_int = int(match)
                if 1980 <= year_int <= current_year + 1:
                    years_found.append(year_int)

        if years_found:
            return max(years_found)
        return None
    
    # def find_best_job_profiles(self, resume_path, top_k=5):
    #     text = extract_resume.extract_text_from_pdf(resume_path)
    #     sections = extract_resume.sectionize_resume(text)
    #     print("Year of completion: ",self._get_year_of_completion(sections.get('education', '')))
    #     resume_features = self._extract_resume_features(sections)
        
    #     if not resume_features.strip():
    #         return {"error": "No meaningful content extracted"}
        
    #     try:
    #         # Vectorize resume
    #         resume_vector = self.vectorizer.transform([resume_features])
            
    #         # Get semantic similarity
    #         semantic_similarities = self._calculate_similarity(resume_vector)
            
    #         # Get resume keywords for skill matching
    #         resume_keywords = set(resume_features.split())
            
    #         # Combine scores
    #         combined_scores = []
    #         for idx, job_title in enumerate(self.job_titles):
    #             semantic_score = semantic_similarities[idx]
    #             keyword_score = self._keyword_overlap(resume_keywords, job_title)
    #             combined = 0.7 * semantic_score + 0.3 * keyword_score
    #             combined_scores.append(combined)
            
    #         # Get top matches
    #         top_indices = np.argsort(combined_scores)[::-1][:top_k]
    #         results = []
    #         for idx in top_indices:
    #             if combined_scores[idx] > 0:
    #                 results.append({
    #                     'job_title': self.job_titles[idx],
    #                     'combined_score': combined_scores[idx],
    #                     'semantic_score': semantic_similarities[idx],
    #                     'keyword_match': self._keyword_overlap(resume_keywords, self.job_titles[idx])
    #                 })
            
    #         return {
    #             'top_matches': results,
    #             'resume_features': resume_features[:300] + "..." if len(resume_features) > 300 else resume_features
    #         }
            
    #     except Exception as e:
    #         return {"error": f"Processing error: {str(e)}"}
    def find_best_job_profiles(self, resume_path: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Find the best matching job profiles for a resume
        
        Returns:
            Dictionary with top matches and metadata
        """
        try:
            # Extract resume content
            text = extract_resume.extract_text_from_pdf(resume_path)
            sections = extract_resume.sectionize_resume(text)
            
            # Extract graduation year
            graduation_year = self._get_year_of_completion(sections.get('education', ''))
            
            # Extract resume features
            resume_features = self._extract_resume_features(sections)
            
            if not resume_features.strip():
                return {"error": "No meaningful content extracted from resume"}
            
            # Vectorize resume
            resume_vector = self.vectorizer.transform([resume_features])
            
            # Calculate semantic similarity
            semantic_similarities = cosine_similarity(resume_vector, self.job_vectors).flatten()
            
            # Calculate skill overlap
            resume_keywords = set(resume_features.split())
            
            # Combine scores (70% semantic, 30% skill overlap)
            combined_scores = []
            for idx, job_title in enumerate(self.job_titles):
                semantic_score = semantic_similarities[idx]
                skill_score = self._calculate_skill_overlap(resume_keywords, job_title)
                combined = 0.7 * semantic_score + 0.3 * skill_score
                combined_scores.append(combined)
            
            # Get top matches
            top_indices = np.argsort(combined_scores)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if combined_scores[idx] > 0:
                    job = self.jobs_data[idx]
                    results.append({
                        'job_title': job.get('title', ''),
                        'company': job.get('company', ''),
                        'location': job.get('location', ''),
                        'employment_type': job.get('employment_type', ''),
                        'combined_score': float(combined_scores[idx]),
                        'semantic_score': float(semantic_similarities[idx]),
                        'skill_match_percentage': float(self._calculate_skill_overlap(resume_keywords, self.job_titles[idx]) * 100),
                        'job_id': job.get('id', idx)  # Assuming jobs have IDs
                    })
            
            return {
                'success': True,
                'top_matches': results,
                'candidate_info': {
                    'graduation_year': graduation_year,
                    'sections_found': {k: bool(v) for k, v in sections.items()},
                    'resume_strength': len(resume_keywords)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in job profile matching: {str(e)}")
            return {"error": f"Processing error: {str(e)}"}
