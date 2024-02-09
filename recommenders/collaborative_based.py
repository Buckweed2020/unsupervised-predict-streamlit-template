# Collaborative-based filtering for item recommendation.

# Script dependencies
import pandas as pd
import numpy as np
import pickle
from surprise import Reader, Dataset
from surprise import SVD
from sklearn.metrics.pairwise import cosine_similarity

# Importing data
movies_df = pd.read_csv('resources/data/movies.csv', sep=',')
ratings_df = pd.read_csv('resources/data/ratings.csv')
ratings_df.drop(['timestamp'], axis=1, inplace=True)
trs_df = pd.read_csv('resources/data/TRS.csv')  # Assuming TRS.csv contains genre information

# We make use of an SVD model trained on a subset of the MovieLens 10k dataset.
model = pickle.load(open('resources/models/SVD.pkl', 'rb'))

def get_user_genre_preferences(user_id):
    """Get user genre preferences based on their past ratings.
    
    Parameters
    ----------
    user_id : int
        User ID.
        
    Returns
    -------
    set of str
        Genre preferences of the user.
    """
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    user_genre_preferences = set(trs_df[trs_df['movieId'].isin(user_ratings['movieId'])]['genre'].unique())
    return user_genre_preferences

def get_user_genre_similarity(user_id_1, user_id_2):
    """Calculate the similarity between two users based on their genre preferences.
    
    Parameters
    ----------
    user_id_1 : int
        First user ID.
    user_id_2 : int
        Second user ID.
        
    Returns
    -------
    float
        Similarity score between the two users based on genre preferences.
    """
    user1_preferences = get_user_genre_preferences(user_id_1)
    user2_preferences = get_user_genre_preferences(user_id_2)
    intersection = len(user1_preferences.intersection(user2_preferences))
    union = len(user1_preferences.union(user2_preferences))
    similarity = intersection / union if union != 0 else 0
    return similarity

def get_similar_users(movie_list):
    """Get similar users based on the given list of movies.
    
    Parameters
    ----------
    movie_list : list of str
        List of movie titles.
        
    Returns
    -------
    list of int
        User IDs of users with similar genre preferences for the given movies.
    """
    similar_users = set()
    for movie_title in movie_list:
        movie_id = movies_df[movies_df['title'] == movie_title]['movieId'].iloc[0]
        reader = Reader(rating_scale=(0, 5))
        load_df = Dataset.load_from_df(ratings_df, reader)
        a_train = load_df.build_full_trainset()
        predictions = model.test(a_train.build_testset())
        predictions = [pred for pred in predictions if pred.iid == movie_id]
        predictions.sort(key=lambda x: x.est, reverse=True)
        for pred in predictions[:10]:
            similar_users.add(pred.uid)
    return list(similar_users)

def collab_model(movie_list, top_n=10):
    """Performs Collaborative filtering based upon a list of movies supplied by the app user.
    
    Parameters
    ----------
    movie_list : list of str
        Favorite movies chosen by the app user.
    top_n : int, optional
        Number of top recommendations to return to the user.
        
    Returns
    -------
    list of str
        Titles of the top-n movie recommendations to the user.
    """
    similar_users = get_similar_users(movie_list)
    # Filter ratings dataframe to include only the movieIds rated by similar users
    ratings_filtered = ratings_df[ratings_df['userId'].isin(similar_users)]
    # Group by movieId and compute mean rating
    movie_ratings = ratings_filtered.groupby('movieId')['rating'].mean()
    # Sort ratings in descending order and get top N movies
    top_movies = movie_ratings.sort_values(ascending=False).head(top_n)
    # Get movie titles corresponding to movieIds
    top_movie_titles = movies_df[movies_df['movieId'].isin(top_movies.index)]['title'].tolist()
    
    # If there are fewer than top_n movies, fill the remaining slots with top-rated movies overall
    if len(top_movie_titles) < top_n:
        top_movies_remaining = top_n - len(top_movie_titles)
        # Filter out movies already recommended
        recommended_movie_ids = set(movies_df[movies_df['title'].isin(top_movie_titles)]['movieId'])
        remaining_movie_ratings = movie_ratings[~movie_ratings.index.isin(recommended_movie_ids)]
        # Sort remaining ratings and get top-rated movies
        top_remaining_movies = remaining_movie_ratings.sort_values(ascending=False).head(top_movies_remaining)
        # Get remaining movie titles and append to recommendations
        top_movie_titles.extend(movies_df[movies_df['movieId'].isin(top_remaining_movies.index)]['title'].tolist())
    
    return top_movie_titles[:top_n]