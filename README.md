#🎬 Netflix Movie Recommendation System

A machine learning project that builds a personalized recommendation system using the Netflix Prize dataset.
We use matrix factorization (SVD) with the Surprise library to predict user ratings and recommend top movies.

---
## 📂 Project Structure

```
Netflix-Recommendation/
│
├── data/
│   ├── combined_data_1.txt        # Raw Netflix ratings dataset
│   ├── movie_titles.csv           # Movie metadata (Movie_Id, Year, Title)
│   ├── ratings_filtered.csv       # Filtered ratings dataset (after preprocessing)
│
├── models/
│   ├── svd_model.pkl              # Trained SVD model (saved using joblib)
│
├── output/
│   ├── 712664.csv                 # Example: saved top recommendations for user_id 712664
│
├── notebooks/
│   ├── experiments.ipynb          # Experiments with filtering & models
│
├── src/
│   ├── data_preprocessing.py      # Functions for loading & filtering data
│   ├── model.py                   # SVD training and evaluation
│   ├── training_pipeline.py       # Full pipeline: preprocess → train → save
│   ├── recommendation.py          # Movie recommendation script
│   ├── logger.py                  # Custom logger for consistent logging
│
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation

```
