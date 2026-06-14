"""
Movie Recommendation System
Professional Recommendation Engine with Multiple Algorithms

This module provides a comprehensive movie recommendation system featuring:
- Collaborative Filtering (user-based and item-based)
- Content-Based Filtering (genre, cast, director similarity)
- Hybrid approach combining multiple algorithms
- Similarity metrics (cosine, Euclidean)
- Model evaluation and validation
- Real-time recommendations
"""

import logging
import pickle
import warnings
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """Data class for movie recommendations"""
    movie_id: int
    movie_title: str
    score: float
    reason: str


class MovieRecommender:
    """Professional Movie Recommendation Engine"""
    
    def __init__(self, algorithm: str = 'hybrid'):
        """
        Initialize recommender with specified algorithm.
        
        Args:
            algorithm: 'collaborative', 'content_based', or 'hybrid'
        """
        self.algorithm = algorithm
        self.movies_df = None
        self.ratings_df = None
        self.user_item_matrix = None
        self.movie_features = None
        self.user_similarity_matrix = None
        self.movie_similarity_matrix = None
        self.scaler = StandardScaler()
        logger.info(f"Initialized MovieRecommender with {algorithm} algorithm")
    
    def generate_synthetic_data(self, 
                               n_users: int = 200, 
                               n_movies: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate synthetic movie and rating data.
        
        Args:
            n_users: Number of users
            n_movies: Number of movies
            
        Returns:
            Tuple of (movies_df, ratings_df)
        """
        logger.info(f"Generating synthetic data: {n_users} users, {n_movies} movies")
        
        np.random.seed(42)
        
        # Movie data
        genres_pool = ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Horror', 'Romance', 'Thriller']
        movies_data = {
            'movie_id': range(1, n_movies + 1),
            'title': [f'Movie {i}' for i in range(1, n_movies + 1)],
            'genre': [','.join(np.random.choice(genres_pool, 2, replace=False)) 
                     for _ in range(n_movies)],
            'release_year': np.random.randint(1990, 2024, n_movies),
            'budget': np.random.normal(50e6, 30e6, n_movies).astype(int),
            'imdb_score': np.random.uniform(4.0, 9.0, n_movies).round(1),
            'runtime': np.random.randint(80, 180, n_movies),
        }
        self.movies_df = pd.DataFrame(movies_data)
        
        # Ratings data (sparse)
        n_ratings = int(n_users * n_movies * 0.15)  # 15% sparsity
        ratings_data = {
            'user_id': np.random.randint(1, n_users + 1, n_ratings),
            'movie_id': np.random.randint(1, n_movies + 1, n_ratings),
            'rating': np.random.choice([1, 2, 3, 4, 5], n_ratings),
            'timestamp': np.random.randint(1600000000, 1700000000, n_ratings)
        }
        self.ratings_df = pd.DataFrame(ratings_data).drop_duplicates(subset=['user_id', 'movie_id'])
        
        logger.info(f"Generated {len(self.ratings_df)} ratings (sparsity: {1 - len(self.ratings_df)/(n_users*n_movies):.1%})")
        
        return self.movies_df, self.ratings_df
    
    def build_user_item_matrix(self) -> np.ndarray:
        """Build user-item rating matrix."""
        logger.info("Building user-item matrix")
        
        self.user_item_matrix = self.ratings_df.pivot_table(
            index='user_id',
            columns='movie_id',
            values='rating',
            fill_value=0
        )
        
        logger.info(f"User-item matrix shape: {self.user_item_matrix.shape}")
        return self.user_item_matrix.values
    
    def compute_user_similarity(self, metric: str = 'cosine') -> np.ndarray:
        """
        Compute similarity between users.
        
        Args:
            metric: 'cosine' or 'euclidean'
            
        Returns:
            User similarity matrix
        """
        logger.info(f"Computing user similarity ({metric})")
        
        if metric == 'cosine':
            self.user_similarity_matrix = cosine_similarity(self.user_item_matrix)
        elif metric == 'euclidean':
            distances = euclidean_distances(self.user_item_matrix)
            self.user_similarity_matrix = 1 / (1 + distances)
        
        return self.user_similarity_matrix
    
    def compute_movie_similarity(self, metric: str = 'cosine') -> np.ndarray:
        """
        Compute similarity between movies (content-based).
        
        Args:
            metric: 'cosine' or 'euclidean'
            
        Returns:
            Movie similarity matrix
        """
        logger.info(f"Computing movie similarity ({metric})")
        
        # Create genre features
        mlb = MultiLabelBinarizer()
        genre_features = mlb.fit_transform(self.movies_df['genre'].str.split(','))
        
        # Normalize numeric features
        numeric_features = self.scaler.fit_transform(
            self.movies_df[['release_year', 'budget', 'imdb_score', 'runtime']]
        )
        
        # Combine features
        movie_features = np.hstack([genre_features, numeric_features])
        self.movie_features = movie_features
        
        if metric == 'cosine':
            self.movie_similarity_matrix = cosine_similarity(movie_features)
        elif metric == 'euclidean':
            distances = euclidean_distances(movie_features)
            self.movie_similarity_matrix = 1 / (1 + distances)
        
        return self.movie_similarity_matrix
    
    def collaborative_filtering(self, 
                               user_id: int, 
                               n_recommendations: int = 5,
                               n_similar_users: int = 5) -> List[Recommendation]:
        """
        Generate recommendations using collaborative filtering.
        
        Args:
            user_id: Target user ID
            n_recommendations: Number of recommendations
            n_similar_users: Number of similar users to consider
            
        Returns:
            List of Recommendation objects
        """
        logger.info(f"Generating collaborative filtering recommendations for user {user_id}")
        
        if self.user_similarity_matrix is None:
            self.compute_user_similarity()
        
        # Find similar users
        user_idx = user_id - 1
        similar_users = self.user_similarity_matrix[user_idx].argsort()[-n_similar_users-1:-1][::-1]
        
        # Get ratings from similar users
        user_ratings = self.user_item_matrix.iloc[user_idx].values
        user_watched = set(np.where(user_ratings > 0)[0])
        
        recommendations = {}
        for similar_user_idx in similar_users:
            similar_user_ratings = self.user_item_matrix.iloc[similar_user_idx].values
            
            for movie_idx, rating in enumerate(similar_user_ratings):
                if rating > 0 and movie_idx not in user_watched:
                    score = rating * self.user_similarity_matrix[user_idx, similar_user_idx]
                    recommendations[movie_idx] = recommendations.get(movie_idx, 0) + score
        
        # Sort and return top recommendations
        top_movies = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
        
        result = []
        for movie_idx, score in top_movies:
            movie_id = movie_idx + 1
            movie = self.movies_df[self.movies_df['movie_id'] == movie_id].iloc[0]
            result.append(Recommendation(
                movie_id=movie_id,
                movie_title=movie['title'],
                score=score,
                reason=f"Similar users rated this movie {score:.2f}/5"
            ))
        
        return result
    
    def content_based_filtering(self, 
                               movie_id: int, 
                               n_recommendations: int = 5) -> List[Recommendation]:
        """
        Generate recommendations based on similar content.
        
        Args:
            movie_id: Reference movie ID
            n_recommendations: Number of recommendations
            
        Returns:
            List of Recommendation objects
        """
        logger.info(f"Generating content-based recommendations for movie {movie_id}")
        
        if self.movie_similarity_matrix is None:
            self.compute_movie_similarity()
        
        movie_idx = movie_id - 1
        similarities = self.movie_similarity_matrix[movie_idx]
        
        # Get top similar movies
        top_indices = np.argsort(similarities)[-n_recommendations-1:-1][::-1]
        
        result = []
        reference_movie = self.movies_df[self.movies_df['movie_id'] == movie_id].iloc[0]
        
        for idx in top_indices:
            similar_movie_id = idx + 1
            movie = self.movies_df[self.movies_df['movie_id'] == similar_movie_id].iloc[0]
            similarity_score = similarities[idx]
            
            result.append(Recommendation(
                movie_id=similar_movie_id,
                movie_title=movie['title'],
                score=similarity_score,
                reason=f"Similar genre/attributes to '{reference_movie['title']}'"
            ))
        
        return result
    
    def hybrid_recommendation(self, 
                             user_id: int, 
                             movie_id: Optional[int] = None,
                             n_recommendations: int = 5,
                             cf_weight: float = 0.6,
                             cb_weight: float = 0.4) -> List[Recommendation]:
        """
        Generate recommendations using hybrid approach.
        
        Args:
            user_id: Target user ID
            movie_id: Optional reference movie for content-based component
            n_recommendations: Number of recommendations
            cf_weight: Weight for collaborative filtering
            cb_weight: Weight for content-based
            
        Returns:
            List of Recommendation objects
        """
        logger.info(f"Generating hybrid recommendations (CF: {cf_weight}, CB: {cb_weight})")
        
        # Collaborative filtering recommendations
        cf_recs = self.collaborative_filtering(user_id, n_recommendations * 2)
        cf_scores = {rec.movie_id: rec.score for rec in cf_recs}
        
        # Content-based recommendations
        if movie_id is None:
            # Use most recently watched movie
            user_idx = user_id - 1
            user_ratings = self.user_item_matrix.iloc[user_idx].values
            movie_id = np.where(user_ratings > 0)[0][-1] + 1 if any(user_ratings > 0) else 1
        
        cb_recs = self.content_based_filtering(movie_id, n_recommendations * 2)
        cb_scores = {rec.movie_id: rec.score for rec in cb_recs}
        
        # Combine scores
        all_movies = set(cf_scores.keys()) | set(cb_scores.keys())
        combined_scores = {
            movie_id: (cf_scores.get(movie_id, 0) * cf_weight + 
                      cb_scores.get(movie_id, 0) * cb_weight)
            for movie_id in all_movies
        }
        
        # Sort and return top recommendations
        top_movies = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
        
        result = []
        for movie_id, score in top_movies:
            movie = self.movies_df[self.movies_df['movie_id'] == movie_id].iloc[0]
            result.append(Recommendation(
                movie_id=movie_id,
                movie_title=movie['title'],
                score=score,
                reason="Hybrid: Similar users + Similar content"
            ))
        
        return result
    
    def get_recommendations(self, 
                           user_id: int, 
                           movie_id: Optional[int] = None,
                           n_recommendations: int = 5) -> List[Recommendation]:
        """
        Get recommendations based on configured algorithm.
        
        Args:
            user_id: Target user ID
            movie_id: Optional reference movie
            n_recommendations: Number of recommendations
            
        Returns:
            List of Recommendation objects
        """
        if self.algorithm == 'collaborative':
            return self.collaborative_filtering(user_id, n_recommendations)
        elif self.algorithm == 'content_based':
            return self.content_based_filtering(movie_id or 1, n_recommendations)
        elif self.algorithm == 'hybrid':
            return self.hybrid_recommendation(user_id, movie_id, n_recommendations)
    
    def evaluate(self, test_split: float = 0.2) -> Dict[str, float]:
        """
        Evaluate recommendation quality using rating prediction.
        
        Args:
            test_split: Fraction of data for testing
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model")
        
        # Split ratings into train/test
        train_ratings, test_ratings = train_test_split(
            self.ratings_df, test_size=test_split, random_state=42
        )
        
        # Train on train set
        train_matrix = train_ratings.pivot_table(
            index='user_id', columns='movie_id', values='rating', fill_value=0
        )
        
        if self.user_similarity_matrix is None:
            self.user_item_matrix = train_matrix
            self.compute_user_similarity()
        
        # Predict test ratings
        predictions = []
        actuals = []
        
        for _, row in test_ratings.iterrows():
            user_id = int(row['user_id'])
            movie_id = int(row['movie_id'])
            actual_rating = row['rating']
            
            try:
                recs = self.collaborative_filtering(user_id, n_recommendations=10)
                rec_movies = {rec.movie_id: rec.score for rec in recs}
                predicted_rating = rec_movies.get(movie_id, np.mean(self.ratings_df['rating']))
                predictions.append(predicted_rating)
                actuals.append(actual_rating)
            except:
                continue
        
        if len(predictions) == 0:
            return {'mae': 0, 'rmse': 0}
        
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        
        logger.info(f"Evaluation - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        
        return {'mae': mae, 'rmse': rmse}
    
    def save_model(self, filepath: str = 'movie_recommender_model.pkl'):
        """Save trained model."""
        model_data = {
            'movies_df': self.movies_df,
            'ratings_df': self.ratings_df,
            'user_item_matrix': self.user_item_matrix,
            'movie_similarity_matrix': self.movie_similarity_matrix,
            'algorithm': self.algorithm,
            'scaler': self.scaler
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str = 'movie_recommender_model.pkl'):
        """Load trained model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        recommender = cls(algorithm=model_data['algorithm'])
        recommender.movies_df = model_data['movies_df']
        recommender.ratings_df = model_data['ratings_df']
        recommender.user_item_matrix = model_data['user_item_matrix']
        recommender.movie_similarity_matrix = model_data['movie_similarity_matrix']
        recommender.scaler = model_data['scaler']
        
        logger.info(f"Model loaded from {filepath}")
        return recommender


def main():
    """Main execution pipeline."""
    
    logger.info("=" * 70)
    logger.info("PROFESSIONAL MOVIE RECOMMENDATION SYSTEM")
    logger.info("=" * 70)
    
    # Initialize recommender
    recommender = MovieRecommender(algorithm='hybrid')
    
    # Generate synthetic data
    movies_df, ratings_df = recommender.generate_synthetic_data(n_users=200, n_movies=50)
    
    print(f"\n[DATASET] Overview:")
    print(f"   Total Users: {ratings_df['user_id'].nunique()}")
    print(f"   Total Movies: {len(movies_df)}")
    print(f"   Total Ratings: {len(ratings_df)}")
    print(f"   Sparsity: {1 - len(ratings_df)/(ratings_df['user_id'].nunique() * len(movies_df)):.1%}")
    
    # Build matrices
    logger.info("\n" + "=" * 70)
    logger.info("BUILDING RECOMMENDATION SYSTEM")
    logger.info("=" * 70)
    
    recommender.build_user_item_matrix()
    recommender.compute_user_similarity()
    recommender.compute_movie_similarity()
    
    # Evaluate model
    logger.info("\n" + "=" * 70)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 70)
    
    metrics = recommender.evaluate()
    print(f"\n[METRICS] Performance:")
    print(f"   Mean Absolute Error (MAE): {metrics['mae']:.4f}")
    print(f"   Root Mean Square Error (RMSE): {metrics['rmse']:.4f}")
    
    # Test recommendations
    logger.info("\n" + "=" * 70)
    logger.info("RECOMMENDATION EXAMPLES")
    logger.info("=" * 70)
    
    test_user = 42
    
    print(f"\n[USER {test_user}] Collaborative Filtering Recommendations:")
    cf_recs = recommender.collaborative_filtering(test_user, n_recommendations=5)
    for i, rec in enumerate(cf_recs, 1):
        print(f"   {i}. {rec.movie_title} (Score: {rec.score:.2f}) - {rec.reason}")
    
    print(f"\n[CONTENT] Similarity-Based Recommendations (Similar to Movie 1):")
    cb_recs = recommender.content_based_filtering(1, n_recommendations=5)
    for i, rec in enumerate(cb_recs, 1):
        print(f"   {i}. {rec.movie_title} (Score: {rec.score:.4f}) - {rec.reason}")
    
    print(f"\n[HYBRID] Combined Recommendations for User {test_user}:")
    hybrid_recs = recommender.hybrid_recommendation(test_user, n_recommendations=5)
    for i, rec in enumerate(hybrid_recs, 1):
        print(f"   {i}. {rec.movie_title} (Score: {rec.score:.4f}) - {rec.reason}")
    
    # Display top movies by rating
    print(f"\n[TOP] Rated Movies:")
    top_movies = movies_df.nlargest(5, 'imdb_score')[['title', 'imdb_score', 'genre']]
    for idx, (_, row) in enumerate(top_movies.iterrows(), 1):
        print(f"   {idx}. {row['title']} ({row['imdb_score']}/10) - Genres: {row['genre']}")
    
    # Save model
    recommender.save_model()
    
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
