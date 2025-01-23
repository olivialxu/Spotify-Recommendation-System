import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('/Users/oliviaxu/Downloads/Spotify 2000.csv')
# print(df.columns)

# Previewing Data
# print(df[['Title','Energy','Danceability','Liveness','Valence','Acousticness','Speechiness']])
# for feature in ['Energy','Danceability','Liveness','Valence','Acousticness','Speechiness']:
#     df[feature].plot(kind='hist', title=feature) 
#     plt.show()

# Scaling Data
scaler = MinMaxScaler()
columns = ['Energy','Danceability','Liveness','Valence','Acousticness','Speechiness']
df[columns] = scaler.fit_transform(df[columns])
