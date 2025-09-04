import pandas as pd
import numpy as np
from src.logger import logger


def load_netflix_data(ratings_path):
    #Load Netflix Prize dataset and create Movie_Id.
    logger.info(f"Loading Netflix data from {ratings_path}")
    df = pd.read_csv(ratings_path, header=None, names=['Cust_Id', 'Rating'], usecols=[0,1])
    
    # Identify movie rows (NaN in Rating)
    df_nan = df[df['Rating'].isnull()].reset_index()
    logger.info(f"Found {len(df_nan)} movie headers in dataset")
 
    # Assign Movie_Id
    df['Movie_Id'] = np.nan
    movie_id = 1
    for i, j in zip(df_nan['index'][:-1], df_nan['index'][1:]):
        df.loc[i:j, 'Movie_Id'] = movie_id
        movie_id += 1
    df.loc[j:, 'Movie_Id'] = movie_id
    
    # Drop NaNs and convert Cust_Id to int
    df.dropna(inplace=True)
    df['Cust_Id'] = df['Cust_Id'].astype(int)
    logger.info(f"Netflix dataset loaded with {df.shape[0]} ratings and {df['Movie_Id'].nunique()} movies")

    return df

def load_movie_titles(title_path):
    # Load movie titles and return DataFrame with Movie_Id as index.
    logger.info(f"Loading movie titles from {title_path}")
    df_title = pd.read_csv(title_path, encoding='ISO-8859-1', header=None, usecols=[0,1,2],
                           names=['Movie_Id','Year','Name'])
    
    logger.info(f"Loaded {len(df_title)} movie titles")

    return df_title

def create_benchmarks_filter(df):
    #Create movie and customer benchmarks using 60th percentile.
    logger.info("Creating benchmarks for movies and customers")
    dataset_movie_summary = df['Movie_Id'].value_counts()
    movie_benchmark = round(dataset_movie_summary.quantile(0.6), 0)
    
    dataset_cust_summary = df['Cust_Id'].value_counts()
    cust_benchmark = round(dataset_cust_summary.quantile(0.6), 0)
    logger.info(f"Movie benchmark: {movie_benchmark}, Customer benchmark: {cust_benchmark}")

    # Movies and customers above benchmark
    keep_movie_list = dataset_movie_summary[dataset_movie_summary >= movie_benchmark].index
    keep_cust_list = dataset_cust_summary[dataset_cust_summary >= cust_benchmark].index

    # Keep only them
    df_filtered = df[df['Movie_Id'].isin(keep_movie_list)]
    df_filtered = df_filtered[df_filtered['Cust_Id'].isin(keep_cust_list)]
    
    logger.info(f"Filtered dataset has {df_filtered.shape[0]} ratings")

    return df_filtered
