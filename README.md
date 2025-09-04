# 🎬 Netflix Movie Recommendation System

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
│   ├── Recommendation_Engine_Using_Netflix.ipynb       
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
---
## ⚙ Project Workflow
### 1. Data Preprocessing

- Load Netflix ratings (combined_data_1.txt)
- Assign Movie_Id from headers in dataset
- Apply benchmarks filtering (60th percentile):
  - Remove unpopular movies
  - Remove inactive customers
- Save processed dataset → data/ratings_filtered.csv

---

### 2. Model Training

- Load filtered dataset
- Train SVD (Singular Value Decomposition) model with Surprise
- Evaluate performance with RMSE & MAE
- Save trained model → models/svd_model.pkl

### Run pipeline:
```
python src/training_pipeline.py

```
---
### 3 Recommendation System

- Load trained model & filtered dataset
- Predict estimated ratings for movies a user hasn’t rated
- Return Top-N recommendations
- Save recommendations as a CSV in output/ folder
  - File is named after the user_id (e.g., output/712664.csv)
 
### Run recommendation:

```
python src/recommendation.py

```
---

### 📊 Example Output

```
Top movie recommendations:
      Movie_Id                          Name  Estimate_Score
1234      5678  The Shawshank Redemption (1994)       4.89
5678      8910               The Godfather (1972)       4.82
...

```

### Saved automatically as:

```
output/712664.csv

```

### ⚙️ Installation

```
# Clone the repository
git clone https://github.com/Subith-Varghese/netflix-recommendation-system.git

# Navigate to the project directory
cd Netflix-Recommendation

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate  

# Install dependencies
pip install -r requirements.txt

# Run the training pipeline
python src/training_pipeline.py

# Run the recommendation script
python src/recommendation.py

```

### 🔑 Key Features

✅ Uses Netflix Prize dataset
✅ Matrix factorization with SVD
✅ Preprocessing with benchmarks filtering
✅ Custom logging system for monitoring
✅ Modular code with clear pipelines
✅ Saves user-specific recommendations in CSV files inside output/
✅ Includes Jupyter notebooks for EDA & experiments
