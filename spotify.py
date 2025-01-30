import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np

# This function loads and preprocesses the song database
# using a min-max scaler (most features are not normally
# distributed), then applying principal component analysis
# (PCA) with two components.
#
# @param  filename - path to song database as a csv file
#
# @return df - song database as a dataframe 
#         features - names of relevant features as an array
#         pca_components - principal components from PCA model
#         pca - PCA model fit to df
def load_data(filename):

    # Read data from filename
    df = pd.read_csv(filename)

    # Identify relevant features
    features = ['Energy','Danceability','Liveness','Valence','Acousticness','Speechiness']

    # Normalize data using MinMaxScaler, which sends range of data to [0, 1]
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    # Apply PCA with two components
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(df[features])

    return df, features, pca_components, pca

# This function uses matplotlib.pyplot to display three plots
# to visualize data and features from the database.
#
# @param  df - song database as a dataframe
#         features - names of relevant features as an array
#         pca - PCA model fit to df
#
# @return Plot 1 - displays histogram visualizing distribution of each feature
#         Plot 2 - displays bar graph comparing PCA coefficients
#         Plot 3 - displays bar graph comparing explained variance ratio of
#                  PCA coefficients
def print_data(df, features, pca):

    # Plot distributions of features
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
    pca_coef.T.plot(kind='bar')
    plt.title('PCA Coefficients')
    plt.tight_layout()
    plt.show()

    # Plot explained variance ratio of principal components
    plt.figure(figsize=(6, 4))
    plt.bar(range(2), pca.explained_variance_ratio_)
    plt.ylabel('Explained Variance Ratio')
    plt.xlabel('Principal Components')
    plt.xticks([1, 2]);
    plt.tight_layout()
    plt.show()

# This function recommends songs to the user given an input song.
# First, the song and its features are extracted from the dataframe.
# Then, we compute the Euclidean distance (in the space of the principal
# components) between the query song and each song in the database.
# Finally, we print the songs closest in distance to the query song.
#
# @param song_title - query song name as a string
#        df - song database as a dataframe
#        features - names of relevant features as an array
#        pca_components - principal components of database from model
#        n - top number of songs recommended; default value is 5
def recommend_songs(song_title, df, features, pca_components, n=5):

    # Convert song title to lowercase for easier comparison
    song_title = song_title.lower()

    # Check that song is in database
    if song_title not in df['Title'].str.lower().values:
        print(f"'{song_title}' not in song database")
        return
    
    # Search for song title in database
    for index, row in df.iterrows():
        if row['Title'].lower() == song_title:
            query_index = index
            break

    # Extract PCA coefficients of query song
    query_pca = pca_components[query_index]

    # Create list of distances between query song and each song in database
    distances = []
    # Loop through songs in database
    for index, row in df.iterrows():
        # Exclude the query song itself
        if index != query_index:
            # Extract principal components of current song
            song_pca = pca_components[index]
            # Compute Euclidean distance in space of principal components
            dist = np.linalg.norm(song_pca - query_pca)
            # Append index and distance
            distances.append((index, dist))

    # Sort songs in ascending order of distance
    distances = sorted(distances, key=lambda x:x[1])

    # Print features of query song
    # print("FEATURES")
    # print(df.loc[query_index][features])
    # print("\n")

    # Print song recommendations
    print("Based on your liked song, we recommend:")
    # Loop through top n closest songs
    for i, (index, _) in enumerate(distances[:n], 1):
        # Extract row corresponding to current song
        song = df.loc[index]
        # Print title and artist of song
        print(f"{song['Title']}, {song['Artist']}")
        # print("FEATURES")
        # print(song[features].transpose())
        # print("\n")

# This function prints the songs with the highest and lowest PC1 and PC2 
# coefficients. We take the dot product of the feature values and PC1 and
# PC2 coefficients for each song. Then, we sort the songs by PC1 and PC2
# components to identify the songs with the highest and lowest PC coefficients.
#
# @param df - song database as a dataframe
#        features - names of relevant features as an array
#        pca_components - principal components from PCA model
#        pca - PCA model fit to df
def find_top_pca_songs(df, features, pca_components, pca):

    # songs with highest PCA 1 and 2 components
    pca_coef = pd.DataFrame(pca.components_, columns=features, index=['PC1', 'PC2'])

    # create a table with rows as indices of songs, two columns (PCA1), (PCA2)
    song_pca = []
    for index, row in df.iterrows():
        song_pca.append((index, np.dot(df.loc[index, features], pca_coef.iloc[0]).item(), np.dot(df.loc[index, features], pca_coef.iloc[1]).item()))
    
    # PCA 1
    # Sort songs by                                  ;jflksdjflk;asdjfslkfasdf
    song_pca = sorted(song_pca, key=lambda x:x[1])
    print(f"The song with the lowest PC1 is '{df.loc[song_pca[0][0]]['Title']}' by '{df.loc[song_pca[0][0]]['Artist']}' with a PC1 coefficient of {song_pca[0][1]:.4f}")

    song_pca = sorted(song_pca, key=lambda x:x[1], reverse=True)
    print(f"The song with the highest PC1 is '{df.loc[song_pca[0][0]]['Title']}' by '{df.loc[song_pca[0][0]]['Artist']}' with a PC1 coefficient of {song_pca[0][1]:.4f}")

    # PCA 2
    song_pca = sorted(song_pca, key=lambda x:x[2])
    print(f"The song with the lowest PC2 is '{df.loc[song_pca[0][0]]['Title']}' by '{df.loc[song_pca[0][0]]['Artist']}' with a PC2 coefficient of {song_pca[0][2]:.4f}")

    song_pca = sorted(song_pca, key=lambda x:x[2], reverse=True)
    print(f"The song with the highest PC2 is '{df.loc[song_pca[0][0]]['Title']}' by '{df.loc[song_pca[0][0]]['Artist']}' with a PC2 coefficient of {song_pca[0][2]:.4f}")

# The main function prompts the user for a song and recommends songs
# The user may also view data from the song database
def main():

    # Load song database
    df, features, pca_components, pca = load_data('spotify.csv')

    # Print PC1, 2 song data
    # find_top_pca_songs(df, features, pca_components, pca)

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