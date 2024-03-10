import uuid

from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score
from recommendation import recommend_jobs


user1 = "SAMPLE_CV_1"
user2 = "SAMPLE_CV_2"


# Sample ground truth data 
ground_truth = {
    "user1": [1, 3, 5],  # Job ad indices relevant to user1
    "user2": [2, 4],     # Job ad indices relevant to user2
}

def evaluate_model(predictions, ground_truth):
    precisions = []
    average_precisions = []
    
    for user_id, true_relevant_jobs in ground_truth.items():
        predicted_jobs = predictions.get(user_id, [])
        
        # Calculate precision
        precision = precision_score(true_relevant_jobs, predicted_jobs, average='micro')
        precisions.append(precision)
        
        # Calculate average precision
        average_precision = average_precision_score(true_relevant_jobs, predicted_jobs, average='micro')
        average_precisions.append(average_precision)
    
    # Calculate mean precision and mean average precision
    mean_precision = sum(precisions) / len(precisions)
    mean_average_precision = sum(average_precisions) / len(average_precisions)
    
    return mean_precision, mean_average_precision

predictions = {
    "user1": recommend_jobs(user1, uuid.uuid1()),
    "user2": recommend_jobs(user2, uuid.uuid1()),
}

mean_precision, mean_average_precision = evaluate_model(predictions, ground_truth)

print("Mean Precision:", mean_precision)
print("Mean Average Precision:", mean_average_precision)
