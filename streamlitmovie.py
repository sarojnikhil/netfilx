import streamlit as st
import pickle
import pandas as pd
import requests
from pydantic import BaseModel
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import HTTPException

# TMDB API details
TMDB_API_KEY = '88f402126ce96431d1bb56587cea4458'
TMDB_API_URL = 'https://api.themoviedb.org/3'

# Define a class for the recommendation request
class RecommendationRequest(BaseModel):
    movie_title: str

# Define a class for the recommendation response
class RecommendationResponse(BaseModel):
    movie_title: str
    recommendations: List[dict]
# Google Drive direct download link for the model
url = 'https://drive.google.com/file/d/1wzcpzGs-mAnJXznDy2Kg-XldoXY2pGHD/view?usp=sharing'  # Replace with your actual link

# Download the model file from Google Drive
r = requests.get(url)
with open('movie_recommender_model.pkl', 'wb') as f:
    f.write(r.content)
# Load the model components from the pickle file
with open('movie_recommender_model.pkl', 'rb') as file:
    model_components = pickle.load(file)

# Extract components from the loaded model
new_df = model_components['new_df']
vectors = model_components['vectors']
similarity = model_components['similarity']

# Load the movie details from the CSV file
movies_df = pd.read_csv('movies.csv')  # Assuming this CSV has 'movie_id', 'title', 'overview'

def check_internet_connection():
    try:
        requests.get("https://www.google.com", timeout=5)
        return True
    except requests.ConnectionError:
        return False

def get_top_indices(similarity_matrix, vector_index, top_n=6):
    similarity_scores = similarity_matrix[vector_index]
    sorted_indices = sorted(enumerate(similarity_scores), key=lambda x: x[1], reverse=True)
    top_indices = [index for index, _ in sorted_indices[:top_n]]
    return top_indices

def recommend_movies(movie_title, new_df, similarity_matrix, movies_df, top_n=5):
    try:
        movie_index = new_df[new_df['title'] == movie_title].index[0]
    except IndexError:
        raise HTTPException(status_code=404, detail=f"Movie '{movie_title}' not found in the database.")

    top_indices = get_top_indices(similarity_matrix, movie_index, top_n + 1)[1:]  # Skip the first index (itself)
    
    # Get the recommended movie IDs
    recommended_movie_ids = new_df.iloc[top_indices]['movie_id'].values
    
    # Fetch the corresponding overviews from movies_df
    recommendations = movies_df[movies_df['movie_id'].isin(recommended_movie_ids)]
    
    return recommendations

def search_movie_by_title(title):
    response = requests.get(f"{TMDB_API_URL}/search/movie", params={
        'api_key': TMDB_API_KEY,
        'query': title,
        'language': 'en-US'
    })
    data = response.json()
    return data['results'][0] if data['results'] else None

def fetch_movie_details(title, timeout=5):
    movie = search_movie_by_title(title)
    if movie:
        movie_id = movie['id']
        try:
            # Fetch movie details with a timeout
            details_response = requests.get(f"{TMDB_API_URL}/movie/{movie_id}", params={
                'api_key': TMDB_API_KEY,
                'language': 'en-US'
            }, timeout=timeout)  # Set a timeout here
            details_response.raise_for_status()
            details = details_response.json()

            # Fetch movie videos (trailers) with a timeout
            videos_response = requests.get(f"{TMDB_API_URL}/movie/{movie_id}/videos", params={
                'api_key': TMDB_API_KEY,
                'language': 'en-US'
            }, timeout=timeout)  # Set a timeout here
            videos_response.raise_for_status()
            videos = videos_response.json()

            # Extract trailer information
            trailers = [video for video in videos.get('results', []) if video['type'] == 'Trailer']
            trailer_url = None
            if trailers:
                trailer_key = trailers[0]['key']
                trailer_url = f"https://www.youtube.com/embed/{trailer_key}"

            # Extract additional movie details
            release_date = details.get('release_date', 'N/A')
            genres = ", ".join([genre['name'] for genre in details.get('genres', [])])
            runtime = f"{details.get('runtime', 0)}m" if details.get('runtime') else "N/A"
            status = details.get('status', 'N/A')
            original_language = details.get('original_language', 'N/A')
            budget = f"${details.get('budget', 0):,.2f}" if details.get('budget') else "N/A"
            revenue = f"${details.get('revenue', 0):,.2f}" if details.get('revenue') else "N/A"

            # Fetch the top-billed cast
            cast_response = requests.get(f"{TMDB_API_URL}/movie/{movie_id}/credits", params={
                'api_key': TMDB_API_KEY,
                'language': 'en-US'
            }, timeout=timeout)
            cast_response.raise_for_status()
            cast_details = cast_response.json()
            top_billed_cast = [
                f"{actor['name']} as {actor['character']}" for actor in cast_details.get('cast', [])[:5]  # Get top 5 cast
            ]

            return {
                'title': details.get('title'),
                'poster_url': f"https://image.tmdb.org/t/p/w500{details.get('poster_path', '')}",
                'overview': details.get('overview'),
                'trailer_url': trailer_url,
                'release_date': release_date,
                'genres': genres,
                'runtime': runtime,
                'status': status,
                'original_language': original_language,
                'budget': budget,
                'revenue': revenue,
                'top_billed_cast': top_billed_cast
            }
        except requests.Timeout:
            raise HTTPException(status_code=504, detail="Request to TMDB timed out. Please try again later.")
        except requests.RequestException as e:
            raise HTTPException(status_code=500, detail="Error fetching movie details from TMDB.")

    raise HTTPException(status_code=404, detail="Movie details not found")


# Streamlit app
st.title('Netflix Hollywood Movie Recommendation System')

# Add Bootstrap CSS
st.markdown('<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">', unsafe_allow_html=True)

# Add custom CSS styles
st.markdown(
    """
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f4f4f4;
            color: #000;
            margin: 0;
            padding: 0;
        }

        .container {
            max-width: 1100px;
            margin: 40px auto;
        }

        .welcome-section {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
            margin-bottom: 40px;
        }

        #recommendation-form {
            margin: 20px 0;
        }

        #recommendation-form input[type="text"] {
            margin: 10px 0;
            background-color: white;
            color: #000;
            border: 1px solid #ccc;
            border-radius: 4px;
            padding: 10px;
        }

        #recommendation-form button {
            background-color: #e50914;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            font-size: 16px;
            font-weight: bold;
            color: white;
        }

        #result {
            margin-top: 20px;
        }

        .movie {
            display: flex;
            height: 50%;
            flex-direction: row;
            margin-bottom: 30px;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .movie:hover {
            transform: scale(1.05);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
        }

        .movie-poster {
            max-width: 40%;
            height: 500px;
            margin-right: 20px;
            border-radius: 8px;
        }

        .movie-details {
            text-align: left;
        }

        .movie-details h2 {
            margin-top: 0;
            font-size: 24px;
        }

        .movie-details p {
            font-size: 16px;
        }

        .trailer-video {
            margin-top: 20px;
        }

        iframe {
            width: 100%;
            height: 350px;
            border-radius: 8px;
        }

        footer {
            margin-top: 40px;
            padding: 20px;
            background-color: white;
            border-top: 1px solid #333;
            text-align: center;
        }

        .spinner {
            display: none;
            margin: 20px auto;
            border: 8px solid #f3f3f3;
            border-top: 8px solid #333;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% {
                transform: rotate(0deg);
            }

            100% {
                transform: rotate(360deg);
            }
        }

        @media (max-width: 768px) {
            .movie {
                flex-direction: column;
                align-items: center;
            }

            .movie-poster {
                max-width: 40%;
                height: 20%;
                margin-right: 0;
            }

            .movie-details {
                text-align: center;
            }
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Input for the movie title
movie_title = st.text_input('Enter a movie title to get recommendations')

if st.button('Recommend', key='recommend_button'):
    if movie_title:
        if check_internet_connection():
            try:
                recommendations = recommend_movies(movie_title, new_df, similarity, movies_df, top_n=10)
                if not recommendations.empty:
                    st.subheader(f"Recommended movies for '{movie_title}':")
                    for i, (_, row) in enumerate(recommendations.iterrows(), start=1):
                        movie_details = fetch_movie_details(row['title'])
                        st.markdown(
                            f"""
                            <div class="movie">
                                <img src="{movie_details['poster_url']}" alt="{movie_details['title']} poster" class="movie-poster">
                                <div class="movie-details">
                                    <h2>{movie_details['title']}</h2>
                                    <p><strong>Overview:</strong> {movie_details['overview']}</p>
                                    <p><strong>Release Date:</strong> {movie_details['release_date']}</p>
                                    <p><strong>Genres:</strong> {movie_details['genres']}</p>
                                    <p><strong>Runtime:</strong> {movie_details['runtime']}</p>
                                    <p><strong>Status:</strong> {movie_details['status']}</p>
                                    <p><strong>Original Language:</strong> {movie_details['original_language']}</p>
                                    <p><strong>Budget:</strong> {movie_details['budget']}</p>
                                    <p><strong>Revenue:</strong> {movie_details['revenue']}</p>
                                    <p><strong>Top Billed Cast:</strong> {', '.join(movie_details['top_billed_cast'])}</p>
                                    <a href="https://www.themoviedb.org/movie/{row['movie_id']}">View on TMDB</a>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        if movie_details['trailer_url']:
                            st.markdown(
                                f"""
                                <div class="trailer-video">
                                    <h3>Trailer:</h3>
                                    <iframe src="{movie_details['trailer_url']}" allowfullscreen></iframe>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                else:
                    st.error(f"No recommendations found for '{movie_title}'.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.error("No internet connection. Please check your connection.")
            st.warning("No internet connection detected. Only movie names and overviews will be displayed.")
            recommendations = recommend_movies(movie_title, new_df, similarity, movies_df, top_n=10)
            if not recommendations.empty:
                st.subheader(f"Recommended movies for '{movie_title}':")
                for i, (_, row) in enumerate(recommendations.iterrows(), start=1):
                    st.markdown(
                        f"""
                        <div class="movie-card">
                            <div class="movie-info">
                                <h3>{i}. {row['title']}</h3>
                                <p>{row['overview']}</p>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
    else:
        st.warning("Please enter a movie title.")
        