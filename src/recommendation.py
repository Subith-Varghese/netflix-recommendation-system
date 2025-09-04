# recommend.py
import pandas as pd
from src.data_preprocessing import load_movie_titles
import joblib
from src.logger import logger
import os

model = joblib.load("models/svd_model.pkl")

titles_path = "data/movie_titles.csv"
titles_df = load_movie_titles(titles_path)

ratings_filtered_path = "data/ratings_filtered.csv"
ratings_filtered = pd.read_csv(ratings_filtered_path)

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

logger.info("Data and model loaded successfully")

def recommend_movies(model, df_title, df_ratings, user_id, top_n=10):
    # Copy titles and reset index to have Movie_Id column
    logger.info(f"Generating recommendations for user {user_id}")
    user_movies = df_title.copy()
    
    # Filter out movies already rated by the user
    rated_movies = df_ratings[df_ratings['Cust_Id'] == user_id]['Movie_Id'].unique()
    user_movies = user_movies[~user_movies['Movie_Id'].isin(rated_movies)]
    
    # Estimate ratings for each movie for this user
    user_movies['Estimate_Score'] = user_movies['Movie_Id'].apply(lambda x: model.predict(user_id, x).est)
    
    # Sort by estimated rating and return top N
    top_recommendations = user_movies.sort_values('Estimate_Score', ascending=False).head(top_n)
    logger.info(f"Top {top_n} recommendations generated for user {user_id}")

    return top_recommendations

if __name__ == "__main__":
    # Recommend movies for a user
    user_id = 712664
    top_recommendations = recommend_movies(model, titles_df, ratings_filtered, user_id, top_n=10)

    # Save recommendations to output folder
    output_path = os.path.join(output_dir, f"{user_id}.csv")
    top_recommendations.to_csv(output_path, index=False)

    logger.info("Top movie recommendations saved to {output_path}")
    logger.info(f"\n{top_recommendations[['Movie_Id', 'Name', 'Estimate_Score']]}")
