import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_data(filename):
    df = pd.read_csv(filename)
    features = ['Energy', 'Danceability', 'Liveness', 'Valence', 'Acousticness', 'Speechiness']

    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features]) # scale data

    pca = PCA(n_components = 2)
    pca_components = pca.fit_transform(df[features]) # use 2 PCs

    return df, features, pca_components, pca

def print_data(df, features, pca):
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(features, 1): # loop over features
        plt.subplot(2, 3, i)
        df[feature].hist()
        plt.title(feature)
    plt.tight_layout()
    plt.show()

    pca_coef = pd.DataFrame(pca.components_, columns=features, index=['PC1', 'PC2'])

    plt.figure(figsize=(10, 6))
    pca_coef.T.plot(kind='bar')
    plt.title('PCA Coefficients')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.bar(range(2), pca.explained_variance_ratio_)
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.tight_layout()
    plt.show()

def recommend_songs(song_title, df, features, pca_components, n=5):

    song_title = song_title.lower()
    if song_title not in df['Title'].str.lower().values:
        print(f"'{song_title}' not in song database")
        return

    for index, row in df.iterrows():
        if row['Title'].lower() == song_title:
            query_index = index
            break

    query_pca = pca_components[query_index]

    distances = []
    for index, row in df.iterrows():
        if index != query_index:
            song_pca = pca_components[index]
            dist = np.linalg.norm(song_pca - query_pca)
            distances.append((index, dist))

    print("Here are your recommendations:")

    for i, (index, _) in enumerate(distances[:n], 1):
        song = df.loc[index]
        print(f"{song['Title']}, {song['Artist']}")

def main():
    df, features, pca_components, pca = load_data('Spotify-2000.csv')

    while True:
        command = input("\nEnter 'song lookup', 'print data', or 'exit': ").strip().lower()

        if command == 'song lookup':
            title = input("what's your favorite song:").strip()
            recommend_songs(title, df, features, pca_components)

        elif command == 'print data':
            print_data(df, features, pca)

        else:
            print("try again?")

if __name__ == "__main__":
    main()
