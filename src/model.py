from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
import joblib
from src.logger import logger


def train_svd_model(df, n_rows=9000000, test_size=0.2, save_model_path="models/svd_model.pkl"):
    logger.info(f"Training SVD model on top {n_rows} rows (test_size={test_size})")
    reader = Reader()
    data = Dataset.load_from_df(df[['Cust_Id', 'Movie_Id', 'Rating']][:n_rows], reader)
    model = SVD()
    
    # Split into train/test
    trainset, testset = train_test_split(data, test_size=test_size, random_state=42)
    logger.info(f"Trainset and testset created: train size={trainset.n_ratings}, test size={len(testset)}")

    # Initialize SVD
    model = SVD()
    
    # Train on trainset
    logger.info("Fitting the SVD model...")
    model.fit(trainset)
    
    # Predict on testset
    predictions = model.test(testset)
    
    # Evaluate
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)
    
    metrics = {'RMSE': rmse, 'MAE': mae}
    logger.info(f"Evaluation completed - RMSE: {rmse}, MAE: {mae}")
    
    joblib.dump(model, save_model_path)
    logger.info(f"Model saved to {save_model_path}")

    return model, metrics