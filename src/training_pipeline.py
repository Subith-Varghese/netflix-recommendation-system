from src.data_preprocessing import load_netflix_data, create_benchmarks_filter
from src.model import train_svd_model
import os
from src.logger import logger

# File paths
ratings_path = "data/combined_data_1.txt"
filtered_ratings_path = "data/ratings_filtered.csv" 
svd_model_path = "models/svd_model.pkl"

# Ensure models directory exists
os.makedirs(os.path.dirname(svd_model_path), exist_ok=True)

if __name__ == "__main__":
    # Load data
    logger.info("Starting training pipeline")

    ratings_df = load_netflix_data(ratings_path)

    #  Apply benchmarks & filter dataset
    ratings_filtered = create_benchmarks_filter(ratings_df)

    # Save filtered dataset to CSV
    ratings_filtered.to_csv(filtered_ratings_path, index=False)
    logger.info(f"Filtered dataset saved to {filtered_ratings_path}")

    # Train SVD model
    model, metrics  = train_svd_model(ratings_filtered,n_rows=9000000,save_model_path=svd_model_path)
    logger.info(f"Training complete with metrics: {metrics}")


