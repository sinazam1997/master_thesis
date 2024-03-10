from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd


model = SentenceTransformer('stsb-xlm-r-multilingual')
collaboration_matrix = pd.read_csv("./resume_positions_collaborative_matrix.csv")

user_logs = list(collaboration_matrix.columns)

def encode_text(text):
    return model.encode([text])[0]


def recommend_jobs(cv_text, user_id):
    encoded_cv = encode_text(cv_text)
    
    
    # Check if user has a cold start problem
    if user_id not in user_logs or len(user_logs[user_id]) == 0:
        # Calculate cosine similarity of the user's CV with job ads
        similarities = np.dot(collaboration_matrix, encoded_cv) / (np.linalg.norm(collaboration_matrix, axis=1) * np.linalg.norm(encoded_cv))
        
        # Get indices of top recommended job ads based on similarity
        top_job_indices = np.argsort(similarities)[::-1][:5]  # Get top 5 recommendations
        
        return top_job_indices
    else:
        # Use collaborative filtering matrix to recommend job ads based on user behavior
        user_interactions = user_logs[user_id]
        recommended_jobs = np.mean(collaboration_matrix[user_interactions], axis=0)
        
        # Calculate cosine similarity of the user's CV with recommended jobs
        similarities = np.dot(recommended_jobs, encoded_cv) / (np.linalg.norm(recommended_jobs) * np.linalg.norm(encoded_cv))
        
        # Get indices of top recommended job ads based on similarity
        top_job_indices = np.argsort(similarities)[::-1][:5]  # Get top 5 recommendations
        
        return top_job_indices

# Example usage:
cv_text = "DESIRED_CV_TEXT"
user_id = "unique_uuid_here"

# Cold start scenario
recommended_jobs_cold_start = recommend_jobs(cv_text, user_id)
print("Recommendations for cold start problem:", recommended_jobs_cold_start)

# User behavior scenario (assuming user has interacted with job ads 0, 1, and 2)
user_logs[user_id] = [0, 1, 2]
recommended_jobs_user_behavior = recommend_jobs(cv_text, user_id)
print("Recommendations based on user behavior:", recommended_jobs_user_behavior)