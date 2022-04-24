import pandas as pd
import datetime
from scipy.spatial import distance
import pandas as pd
import plotly.express as px
from sklearn.neighbors import NearestNeighbors 
import numpy as np
pd.options.mode.chained_assignment = None
from data_cache import pandas_cache

#------------------------------------------------------------------------
# NOTES
# - code not tested yet lol
# - Optimization is required, like: feature selection, mapping artists to
# broader genres
# use playlist database to generate new playlist might be a more optimial 
# solution: https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge
# - spotify api connection missing
#------------------------------------------------------------------------

@pandas_cache
def load_song_db():
    """loads song db downloaded from kaggel

    Returns:
        _type_: return pandas df with songs
    """
    df = pd.read_csv('data/tracks.csv.zip', compression='zip', header=0, sep=',', quotechar='"')
    df["song"] = df["name"] + "_" + df["artists"]
    df["id_artists"] = df['id_artists'].str.strip('[]').str.strip("''")
    # filter songs that are shorter than 7 minutes
    df = df[df["duration_ms"] < 420000]
    return df


@pandas_cache
def load_artist_db():
    """load inforamtion about artist into df

    Returns:
        _type_: pandas dataframe containg artist, id, genres
    """
    artists = pd.read_csv('data/artists.csv.zip', 
    compression='zip', header=0, sep=',', quotechar='"')
    artists['name'] = artists['name'].str.strip('[]').str.strip("''")
    artists['genres'] = artists['genres'].str.strip('[]').str.strip("''")
    return artists


# in progress
def subset_according_artistgenre(df, artist_id=None, genre=None):
    """fin
    Args:
        df (_type_): _description_
        artist_id (_type_, optional): _description_. Defaults to None.
        genre (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    artists = load_artist_db()
    try:
        genre = artists[artists["id"] == artist_id]["genres"].values[0]
        genre_artists = artists[artists["genres"].str.contains(genre)]["id_artists"].tolist()
        df = df[df["id_artists"].isin(genre_artists)]
    except IndexError:
        print("Genre of artist couldnt be identified, using full database for playlist generation.")
    return df


def subset_according_genre(genre=None):
    """subset song database according to genre 
    (Note: the Genre is depending on the artist and not the song itself)

    Args:
        genre (str, optional): string with name of genre. Defaults to None.

    Returns:
        _type_: database with specific genre
    """
    df = load_song_db()
    artists = load_artist_db()
    if genre:
        genre_artists = artists[artists["genres"].str.contains(genre)]["id"].tolist()
        df = df[df["id_artists"].isin(genre_artists)]
    return df


def subset_similar_artists():
    """subset dataframe by similar artists
    """
    # placeholder
    # https://developer.spotify.com/console/get-artist-related-artists/
    pass


def get_similar_songs(genre_df, songs_favourite_df, n_songs= 25):
    """generates playlist of similar songs usning Jaccard Similarity
    Note: both 

    Args:
        genre_df (_type_): dataframe subsetted by specific genre (use playlist_of_the_day())
        songs_favourite_df (_type_): dataframe containing favourite songs
        n_songs (int, optional): size of playlist. Defaults to 25.

    Returns:
        _type_: dataframe with songs for playlist
    """

    features_list =  ['danceability', 'energy', 'key',
       'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
       'liveness', 'valence', 'tempo']

    combine = pd.concat(genre_df, songs_favourite_df)
    j = distance.pdist(combine[features_list], "jaccard")
    k = combine["id"].to_numpy()
    d = pd.DataFrame(1 - distance.squareform(j), index=k, columns=k)
    d['mean'] = d.mean(axis=1) 
    return  combine[combine["id"].isin(d.sort_values("mean",ascending =False).index[0:n_songs])] 


def playlist_of_the_day():
    """subsets Song Database to specifc genre according to day

    Returns:
        _type_: pandas dataframe
    """
    weekday = datetime.datetime.today().weekday()
    # 0 = Monday, 6 = Sunday
    if weekday == 0:
        genre_otd = "lo-fi"
        print("Today is Monday, generating Lo-Fi playlist")
    elif weekday == 4:
        genre_otd = "pop"
        print("Today is Friday, generating pop playlist")
    else:
        genre_otd = None
        print("No genre selected using full song database")
    df = subset_according_genre(genre = genre_otd)
    return df
    

def knn_song_recommendation(song_id, df = None, n_songs = 25, subset_genre = False, related_artists = False):
    """finds k nearest neigbourhs of a particular song using the song id of song db

    Args:
        song_id (_type_): the song id where similar songs should be found. song id is found in the song_db column "id"
        n_songs (int, optional): number of recommended songs. Defaults to 25.
        subset_genre (bool, optional): whether the search should be narrowed down to the genre of the songs artists. Defaults to True.

    Returns:
        _type_: Pandas dataframe with recommended songs
    """
    # list of features
    # might subset features for fine tunning
    features_list =  ['danceability', 'energy', 'key',
       'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
       'liveness', 'valence', 'tempo']
    if df is None:
        df = load_song_db() 
    # generate df with only songs from the same genre
    if subset_genre:
        df = subset_according_genre(df) 

    # finds similar aritists before doing kmean to get more accurate results
    # not implemented yet
    if related_artists:
        df = subset_similar_artists(df)
    
    # KNN
    df = df.set_index(np.arange(df.shape[0]))
    row_number = df[df["id"]==song_id].index[0]
    X = df[features_list].to_numpy()
    nbrs = NearestNeighbors(n_neighbors=n_songs)
    nbrs.fit(X)
    neighbors_f_id = nbrs.kneighbors_graph([X[row_number]]).indices
    # row numbers with recommended songs
    recom = neighbors_f_id.tolist()
    # return df with song recommendation
    return df.iloc[recom]