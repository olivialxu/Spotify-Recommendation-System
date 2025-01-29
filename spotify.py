import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np

# Load song database
def load_data(filename):

    # Load data
    df = pd.read_csv(filename)

    # Identify relevant features
    features = ['Energy','Danceability','Liveness','Valence','Acousticness','Speechiness']

    # Normalize data
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    # Apply PCA with two components
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(df[features])

    return df, features, pca_components, pca

# Visualize song data
def print_data(df, features, pca):

    # Plot distributions of energy, danceability, liveness, valence, acousticness, speechiness
    plt.figure(figsize=(12, 8))
    # Loop over six features and create histograms to visualize distribution of each feature
    for i, feature in enumerate(features, 1):
        plt.subplot(2, 3, i)
        df[feature].hist()
        plt.title(feature)
    plt.tight_layout()
    plt.show()

    # Plot PCA coefficients
    pca_coef = pd.DataFrame(pca.components_, columns=features, index=['PC1', 'PC2'])
    plt.figure(figsize=(10, 6))
    # Create graph to compare two PCA coefficients
    pca_coef.T.plot(kind='bar')
    plt.title('PCA Coefficients')
    plt.tight_layout()
    plt.show()

    # Plot explained variance ratio of PCA components
    plt.figure(figsize=(6, 4))
    plt.bar(range(2), pca.explained_variance_ratio_)
    plt.ylabel('Explained Variance Ratio')
    plt.xlabel('Principal Components')
    plt.tight_layout()
    plt.show()

# Recommend songs to user
def recommend_songs(song_title, df, features, pca_components, n=5):

    song_title = song_title.lower()

    # Search for song title in database
    if song_title not in df['Title'].str.lower().values:
        print(f"'{song_title}' not in song database")
        return

    for index, row in df.iterrows():
        if row['Title'].lower() == song_title:
            query_index = index
            break

    # Extract PCA coefficients of queried song
    query_pca = pca_components[query_index]

    # Create list of distances of each song in database (excluding queried song)
    distances = []
    for index, row in df.iterrows():
        if index != query_index:
            song_pca = pca_components[index]
            dist = np.linalg.norm(song_pca - query_pca)
            distances.append((index, dist))

    # Sort distances
    distances = sorted(distances, key=lambda x:x[1])

    # Print song recommendations
    print("Here are your recommendations:")
    for i, (index, _) in enumerate(distances[:n], 1):
        song = df.loc[index]
        print(f"{song['Title']}, {song['Artist']}")

# Print songs with highest and lowest PC1 and PC2 coefficients
def find_top_pca_songs(df, features, pca_components, pca):

    # songs with highest PCA 1 and 2 components
    pca_coef = pd.DataFrame(pca.components_, columns=features, index=['PC1', 'PC2'])

    # create a table with rows as indices of songs, two columns (PCA1), (PCA2)
    song_pca = []
    for index, row in df.iterrows():
        song_pca.append((index, np.dot(df.loc[index, features], pca_coef.iloc[0]).item(), np.dot(df.loc[index, features], pca_coef.iloc[1]).item()))
    
    # PCA 1
    song_pca = sorted(song_pca, key=lambda x:x[1])
    print(f"The song with the lowest PC1 is '{df.loc[song_pca[0][0]]['Title']}' by '{df.loc[song_pca[0][0]]['Artist']}' with a PC1 coefficient of {song_pca[0][1]:.4f}")

    song_pca = sorted(song_pca, key=lambda x:x[1], reverse=True)
    print(f"The song with the highest PC1 is '{df.loc[song_pca[0][0]]['Title']}' by '{df.loc[song_pca[0][0]]['Artist']}' with a PC1 coefficient of {song_pca[0][1]:.4f}")

    # PCA 2
    song_pca = sorted(song_pca, key=lambda x:x[2])
    print(f"The song with the lowest PC2 is '{df.loc[song_pca[0][0]]['Title']}' by '{df.loc[song_pca[0][0]]['Artist']}' with a PC2 coefficient of {song_pca[0][2]:.4f}")

    song_pca = sorted(song_pca, key=lambda x:x[2], reverse=True)
    print(f"The song with the highest PC2 is '{df.loc[song_pca[0][0]]['Title']}' by '{df.loc[song_pca[0][0]]['Artist']}' with a PC2 coefficient of {song_pca[0][2]:.4f}")

# Main function
def main():

    # Load song database
    df, features, pca_components, pca = load_data('spotify.csv')

    find_top_pca_songs(df, features, pca_components, pca)

    while True:
        command = input("\nEnter 'song lookup', 'print data', or 'exit': ").strip().lower()

        if command == 'song lookup':
            title = input("what's your favorite song: ").strip()
            recommend_songs(title, df, features, pca_components)

        elif command == 'print data':
            print_data(df, features, pca)

        elif command == 'exit':
            print("adios")
            break

        else:
            print("try again?")

if __name__ == "__main__":
    main()