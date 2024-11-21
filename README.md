# 🎵 Music Recommendation System

A sophisticated, personalized music recommendation platform leveraging Spotify's API and advanced machine learning models to enhance music discovery.

## 🌟 Project Overview

This capstone project addresses the challenge of personalized music recommendation by utilizing Spotify's extensive music library and advanced audio feature analysis.

## 🎯 Key Features

### 🔍 Advanced Recommendation Models
- **K-Nearest Neighbors (KNN)**: Identifies songs similar to input tracks
- **Cosine Similarity**: Analyzes angular similarity between songs
- **PCA + KNN**: Reduces dimensionality for efficient processing
- **DBSCAN**: Creates song clusters for diverse recommendations

### 🌐 Real-Time Capabilities
- Dynamic song data retrieval from Spotify
- Automatic dataset expansion
- Real-time recommendation generation

### 🖥️ User Experience
- Interactive Streamlit web application
- Visualizations of music trends
- Album cover and Spotify link displays, Click on Listen on Spotify and it directly takes you to spotify page of the song where you can easily listen to song

## 🛠️ Technical Stack

### Languages & Libraries
- **Language**: Python
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **API Integration**: Spotipy
- **Web Interface**: Streamlit

### Machine Learning Models
- K-Nearest Neighbors
- Cosine Similarity
- Principal Component Analysis (PCA)
- DBSCAN Clustering

## 📊 Dataset Details

### Data Sources
1. **Spotify Audio Features (`2audio_features.csv`) with Spotify API**: 
   - Manually curated by hand-picking tracks from 50+ Spotify playlists
   - Collected and processed by the project team
   - Contains detailed audio features for selected tracks
   - Check out Jupter Notebook to see how I did it [Jupyter Notebook](2MRS.ipynb)

2. **Historical Tracks (`tracks.csv`)**: 
   - Sourced from Kaggle: [Spotify Dataset 1921-2020, 600k+ Tracks](https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks?select=tracks.csv)
   - Comprehensive dataset spanning nearly a century of music
   - Provides a broad base for music recommendations

## 🚀 Quick Setup

### Prerequisites
- Python 3.8+
- Spotify Developer Account(Keep reading to know how to make one)

## 🚀 Quick Start

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/Music-Recommendation-System.git
   cd Music-Recommendation-System
   ```

2. Install Dependencies
   ```bash
   pip install -r app/requirements.txt
   ```
   
3. Add cliend id and client secret id of your Spotify API. Get these by following 🔐 Spotify API Credentials Section below. Goto Step 4 only after completing Step 3
 
4. Run the Streamlit App
   ```bash
   streamlit run app2.py
   ```

## 🔐 Spotify API Credentials

### Quick Spotify API Setup
1. Create a Spotify Developer Account at [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/)

2. Create App. Select Web API in types of API. 

3. Find your Client ID and Client Secret. Typically found in settings of the app you created.

4. In `app2.py`, directly add your Spotify credentials:
   ```python
   # Spotify API setup.
   # Replace YOUR_CLIENT_ID and YOUR_CLIENT_SECRET with your Spotify Api client id and client secret id 
   sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
       client_id="YOUR_CLIENT_ID",
       client_secret="YOUR_CLIENT_SECRET"
   ))
   ```

## Repository Structure Options

### Option 1: 📁 Recommended Project Structure
```
Music-Recommendation-System/
├── app/
│   ├── app2.py           # Main Streamlit application
│   └── requirements.txt  # Project dependencies
├── data/
│   ├── tracks.csv        # Kaggle dataset
│   └── 2audio_features.csv  # Manually curated tracks
└── README.md
```
### Option 2: Modify Code Paths
If you prefer a different structure, update file paths in `app2.py`:
```python
# Find these lines in the code 
...
df1 = pd.read_csv("../data/2audio_features.csv")
...
song_library = pd.read_csv("../data/tracks.csv")
```

## 📝 License
MIT License - See [LICENSE](LICENSE) for details

## 🔗 Additional Resources
- [Medium Blog Post](https://medium.com/@2103a52159/discover-your-next-favorite-song-building-a-music-recommendation-system-ce998fd229a3)
- [LinkedIn Post](https://www.linkedin.com/posts/naveed-sharief-b0ba1b252_machinelearning-musictech-aiinnovation-activity-7264335292780748800-zSgd?utm_source=share&utm_medium=member_desktop)
- [Project Report](reports/Final-Report.pdf)
- [Project Poster](reports/Poster.pdf)
- [Project PPT](reports/PPT.pdf)
- [Youtube Video giving the PPT presentation](https://youtu.be/McpnVNAPNQo?si=87_OhFmZ4dh3LOug)

## 🐛 Issues & Contributions
Found a bug? Want to contribute? Please open an issue or submit a pull request!

---

**Happy Music Discovering! 🎧**
