import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np

# Loading and previewing data
df = pd.read_csv('~/Downloads/spotify-recommender/spotify.csv')

# Identifying relevant columns
columns = ['Energy','Danceability','Liveness','Valence','Acousticness','Speechiness']

# Plotting distribution of each feature
for feature in columns:
    df[feature].plot(kind='hist', title=feature) 
    plt.show()

# Scaling data
scaler = MinMaxScaler()
df[columns] = scaler.fit_transform(df[columns])

# Extracting the relevant columns
spotify_df = df[columns]

# Applying PCA with two components
pca = PCA(n_components=2)
pca_components = pca.fit_transform(spotify_df)
print(pca_components.shape)
print(pca_components)

pca_coef = pd.DataFrame(pca.components_, columns=columns, index=['PC1', 'PC2'])

# Plotting the two PCA components
pca_coef.T.plot(kind='bar', figsize=(8, 6), title='PCA Coefficients of Features')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.legend(title='Principal Components')
plt.show()

# transform 6 axes to 2 PCA axes --> how do you visualize the linear transformation from the 6 axes to the 2 PCA axes
# and how do you 


# TO-DO:
# 1. Incorporate PCA into computing Euclidean distances
# 2. Consider other song features
# 3. Implement PCA from scratch
#
# OTHER:
# 3. Perform k-means clustering

# Recommend n songs given the title of a favorable song
def recommend_song(song_title, top_n=5):
    if song_title.lower() not in df['Title'].str.lower().values:
        print(f"Song '{song_title}' not found.")
        return
    
    # Select row corresponding to song title
    song_row = df[df['Title'].str.lower() == song_title.lower()]

    query_index = song_row.index[0]
    query_features = df.loc[query_index, columns].values

    # Calculate distance between songs in database and queried song
    distances = []
    for i, song in df.iterrows():
        if i == query_index:
            continue
    
        song_features = song[columns].values
        dist = np.linalg.norm(song_features - query_features)
        distances.append((i, dist))

    # Sort songs by closeness
    distances = sorted(distances, key=lambda x: x[1])
    top_songs = distances[:top_n] # (index of song, distance)

    # Select top songs and sort by popularity
    top_songs = df.loc[top_songs[0]].sort_values(by='Popularity', ascending=False)

    # Print user's recommended songs 
    print(f"\nBecause you like the song '{df.loc[query_index, 'Title']}' by {df.loc[query_index, 'Artist']}...")
    print("We think you may also like the songs:")
    for i, song_index in enumerate(top_songs, start=1):
        song = df.loc[song_index]
        print(f"{i}. '{song['Title']}' by {song['Artist']} (Popularity: {song['Popularity']})")

# Main 
while True:
    status = input("\nEnter 'recommend' for song recommendations or 'exit' to exit: ").strip().lower()
    if status == "recommend":
        title = input("Enter the song name: ").strip()
        recommend_song(title)
    elif status == "exit":
        print("Program successfully exited.")
        break
    else:
        print("Invalid input.")