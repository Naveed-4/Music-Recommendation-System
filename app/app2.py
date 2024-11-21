# Github Repositary: https://github.com/Naveed-4/Music-Recommendation-System

# Replace YOUR_CLIENT_ID and YOUR_CLIENT_SECRET in Spotify API Setup below with your Spotify API Credentials

import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Spotify API setup. 
# Replace YOUR_CLIENT_ID and YOUR_CLIENT_SECRET with your Spotify Api client id and client secret id 
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET"
))

# Load dataset
@st.cache_data
def load_data():
    df1 = pd.read_csv("../data/2audio_features.csv")
    df1.drop_duplicates(subset=['id'], keep='first', inplace=True)
    scaler = StandardScaler()
    numerical_features = ['danceability', 'energy', 'tempo', 'loudness', 'speechiness',
                          'acousticness', 'instrumentalness', 'liveness', 'valence']
    df1[numerical_features] = scaler.fit_transform(df1[numerical_features])
    return df1, scaler, numerical_features

df1, scaler, numerical_features = load_data()

song_library = pd.read_csv("../data/tracks.csv")
 #Taking only top 100k based on popularity, so that DBSCAN performs fast. for 600k only DBSCAN takes time(alot), others dont, others take only couple secs even for whole dataset
df2 = song_library.sort_values(by=['popularity'], ascending=False).head(100000).reset_index(drop=True) 

df2.rename(columns={'name': 'track_name'}, inplace=True)

#Find the common columns between df and df2
common_cols = df1.columns.intersection(df2.columns)

#Concatenate df and df2 vertically, keeping only the common columns
df = pd.concat([df1[common_cols], df2[common_cols]], ignore_index=True)

# df = df2.copy()
df.drop_duplicates(subset=['id'], keep='first', inplace=True)
# df.drop_duplicates(subset=['track_name'], keep='first', inplace=True)
df.reset_index(drop=True, inplace=True)
df.dropna()
df.reset_index(drop=True, inplace=True)

# Functions
def get_track_info(song_name, artist_name):
    """Fetch track ID, name, album cover URL, and Spotify link."""
    query = f"track:{song_name} artist:{artist_name}"
    results = sp.search(q=query, type='track', limit=1)
    if results['tracks']['items']:
        track_info = results['tracks']['items'][0]
        return {
            'id': track_info['id'],
            'name': track_info['name'],
            'album_cover': track_info['album']['images'][0]['url'],
            'spotify_link': track_info['external_urls']['spotify']
        }
    return None

def add_song_to_dataset(track_id):
    """Fetch audio features and add the song to the dataset."""
    audio_features = sp.audio_features([track_id])[0]
    if audio_features:
        track_data = sp.track(track_id)
        new_row = {feature: audio_features.get(feature, 0) for feature in numerical_features}
        new_row.update({
            'id': track_id,
            'track_name': track_data['name'],
            'album_cover': track_data['album']['images'][0]['url'],
            'spotify_link': track_data['external_urls']['spotify']
        })
        new_row = pd.DataFrame([new_row])
        new_row[numerical_features] = scaler.transform(new_row[numerical_features])
        global df
        df = pd.concat([df, new_row], ignore_index=True)
        return df
    return None

def recommend_using_knn(track_id, n_neighbors=5):
    if track_id in df['id'].values:
        track_idx = df.index[df['id'] == track_id].tolist()[0]
        knn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='auto')
        knn.fit(df[numerical_features])
        distances, indices = knn.kneighbors([df.loc[track_idx, numerical_features]])
        return df.iloc[indices[0][1:]]
    return pd.DataFrame()

def recommend_using_cosine_similarity(track_id, n_neighbors=5):
    if track_id in df['id'].values:
        track_idx = df.index[df['id'] == track_id].tolist()[0]
        similarity = cosine_similarity([df.loc[track_idx, numerical_features]], df[numerical_features])
        similar_indices = similarity.argsort()[0, -n_neighbors-1:-1][::-1]
        return df.iloc[similar_indices]
    return pd.DataFrame()

def recommend_using_pca_and_knn(track_id, n_neighbors=5, n_components=5):
    if track_id in df['id'].values:
        pca = PCA(n_components=n_components)
        reduced_features = pca.fit_transform(df[numerical_features])
        track_idx = df.index[df['id'] == track_id].tolist()[0]
        knn_pca = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='auto')
        knn_pca.fit(reduced_features)
        distances, indices = knn_pca.kneighbors([reduced_features[track_idx]])
        return df.iloc[indices[0][1:]]
    return pd.DataFrame()

from sklearn.metrics.pairwise import cosine_similarity

def recommend_using_dbscan(track_id, eps=1.5, min_samples=5, n_neighbors=5):
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(df[numerical_features])
    df['cluster'] = cluster_labels

    # Check if the input track ID is valid
    if track_id in df['id'].values:
        track_idx = df.index[df['id'] == track_id].tolist()[0]
        cluster = df.loc[track_idx]['cluster']

        # Handle noise points
        if cluster == -1:
            return pd.DataFrame()

        # Filter tracks from the same cluster
        similar_tracks = df[df['cluster'] == cluster].copy()

        # Calculate cosine similarity within the cluster
        input_features = df.loc[track_idx, numerical_features].values.reshape(1, -1)
        cluster_features = similar_tracks[numerical_features].values
        similarity_scores = cosine_similarity(input_features, cluster_features)[0]

        # Rank tracks by similarity (excluding the input track itself)
        similar_tracks['similarity'] = similarity_scores
        similar_tracks = similar_tracks[similar_tracks['id'] != track_id]  # Exclude the input track
        similar_tracks = similar_tracks.sort_values(by='similarity', ascending=False).head(n_neighbors)

        return similar_tracks[['track_name', 'id', 'similarity']]
    
    # If the track ID is not found, return an empty DataFrame
    return pd.DataFrame()


def fetch_recommendation_metadata(recommendations):
    """Fetch metadata for recommended songs."""
    if not recommendations.empty:
        ids = recommendations['id'].tolist()
        track_infos = []
        for track_id in ids:
            track_data = sp.track(track_id)
            track_infos.append({
                'name': track_data['name'],
                'artist': ', '.join([artist['name'] for artist in track_data['artists']]),
                'album_cover': track_data['album']['images'][0]['url'],
                'spotify_link': track_data['external_urls']['spotify']
            })
        return track_infos
    return []

import os
import base64
import streamlit as st

# Function to set background image
def set_bg_hack(main_bg):
    main_bg_ext = "jpg"
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        .title {{
            color: #1DB954;  /* Spotify green color */
            text-align: center;
            padding: 2rem 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            font-size: 2.5em;
            font-weight: bold;
        }}
        .note {{
            background-color: rgba(29, 185, 84, 0.15);  /* Slightly more visible green background */
            border-left: 4px solid #1DB954;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
            color: #1a8e40;  /* Darker green for better readability */
            font-weight: 500;
        }}
        .stButton>button {{
            width: 100%;
            background-color: #1DB954;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            font-weight: bold;
            transition: background-color 0.3s ease;
        }}
        .stButton>button:hover {{
            background-color: #1ed760;  /* Lighter green on hover */
        }}
        .recommendation-box {{
            background-color: rgba(29, 185, 84, 0.08);  /* Very light green background */
            padding: 1.5rem;
            border-radius: 8px;
            margin: 1rem 0;
            box-shadow: 0 2px 8px rgba(29, 185, 84, 0.1);
            border: 1px solid rgba(29, 185, 84, 0.2);
        }}
        .stTextInput>div>div>input {{
            border: 2px solid #1DB954;
            border-radius: 5px;
            color: #1a8e40;  /* Dark green text */
            background-color: rgba(255, 255, 255, 0.9);
        }}
        .stTextInput>div>div>input:focus {{
            border-color: #1ed760;
            box-shadow: 0 0 0 2px rgba(29, 185, 84, 0.2);
        }}
        /* Styling for expander */
        .streamlit-expanderHeader {{
            background-color: rgba(29, 185, 84, 0.1) !important;
            color: #1a8e40 !important;
            border-radius: 5px;
        }}
        .streamlit-expanderContent {{
            border: 1px solid rgba(29, 185, 84, 0.2);
            border-top: none;
            border-radius: 0 0 5px 5px;
        }}
        /* Override default text color */
        .stMarkdown {{
            color: #1a8e40;
        }}
        h3 {{
            color: #1DB954;
            margin-bottom: 1rem;
        }}
        a {{
            color: #1DB954 !important;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
            color: #1ed760 !important;
        }}
        /* Custom styles for image captions */
        .caption {{
            color: #1a8e40;
            text-align: center;
            font-size: 0.9em;
            margin-top: 0.5rem;
        }}
        /* Error message styling */
        .stAlert {{
            background-color: rgba(255, 76, 76, 0.1);
            color: #ff4c4c;
            border-radius: 5px;
            padding: 1rem;
            margin: 1rem 0;
        }}
        /* Success message styling */
        .success {{
            background-color: rgba(29, 185, 84, 0.1);
            color: #1DB954;
            border-radius: 5px;
            padding: 1rem;
            margin: 1rem 0;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set the background image
bg_image_path = r"C:/Users/mdnav/OneDrive/Desktop/background.jpg"
if os.path.exists(bg_image_path):
    set_bg_hack(bg_image_path)
else:
    st.error("Background image not found!")

# Title and Project Note
st.markdown("<h1 class='title'> Music Recommendation System</h1>", unsafe_allow_html=True)
st.markdown(
    "<div class='note'><b>Note:</b> This Music Recommendation System is part of the "
    "Capstone Project done during the 7th semester (2024)</div>", 
    unsafe_allow_html=True
)

# Warning about input accuracy
st.markdown(
    "<div class='note'><b>Important:</b> Song name and artist name must be "
    "absolutely correct for accurate results.</div>",
    unsafe_allow_html=True
)

# Input fields
col1, col2 = st.columns(2)
with col1:
    song_name = st.text_input("Enter the song name:")
with col2:
    artist_name = st.text_input("Enter the artist name:")

if st.button("Get Recommendations"):
    track_info = get_track_info(song_name, artist_name)
   
    if track_info:
        st.markdown(
            f"<div class='recommendation-box'>"
            f"<h3>Found track: {track_info['name']}</h3>"
            "</div>",
            unsafe_allow_html=True
        )
        
        # Display album cover with custom styling
        st.markdown(
            f"<div style='text-align: center;'>",
            unsafe_allow_html=True
        )
        st.image(track_info['album_cover'], width=300)
        st.markdown(
            f"<p class='caption'>{track_info['name']} by {artist_name}</p>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div style='text-align: center;'>"
            f"<a href='{track_info['spotify_link']}' target='_blank'>Listen on Spotify</a>"
            "</div>",
            unsafe_allow_html=True
        )
       
        if track_info['id'] not in df['id'].values:
            st.markdown(
                "<div class='note'>Song not found in dataset. Adding it now...</div>",
                unsafe_allow_html=True
            )
            df = add_song_to_dataset(track_info['id'])
       
        # Models and Results
        models = {
            "KNN": recommend_using_knn,
            "Cosine Similarity": recommend_using_cosine_similarity,
            "PCA + KNN": recommend_using_pca_and_knn,
            "DBSCAN": recommend_using_dbscan
        }
       
        for model_name, model_func in models.items():
            with st.expander(f"Recommendations using {model_name}"):
                st.markdown("<div class='recommendation-box'>", unsafe_allow_html=True)
                recommendations = model_func(track_info['id'])
                metadata = fetch_recommendation_metadata(recommendations)
                if metadata:
                    cols = st.columns(3)
                    for idx, rec in enumerate(metadata):
                        with cols[idx % 3]:
                            st.image(rec['album_cover'], width=150)
                            st.markdown(
                                f"<p class='caption'>{rec['name']} by {rec['artist']}</p>"
                                f"<div style='text-align: center;'>"
                                f"<a href='{rec['spotify_link']}' target='_blank'>Listen on Spotify</a>"
                                "</div>",
                                unsafe_allow_html=True
                            )
                else:
                    st.markdown(
                        f"<div class='note'>No recommendations found using {model_name}.</div>",
                        unsafe_allow_html=True
                    )
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<div class='stAlert'>Could not find the track based on your input.</div>",
            unsafe_allow_html=True
        )

# Footer
st.markdown(
    "<div class='note' style='text-align: center; margin-top: 2rem;'>"
    "Created with ❤️ by Mudassir(2103A52058), Ajaz(2103A52069) and Naveed(2103A52159) | "
    "<a href='https://github.com/Naveed-4/Music-Recommendation-System' target='_blank'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True
)