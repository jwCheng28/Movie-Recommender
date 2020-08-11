import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import warnings
import os.path
import pickle

def get_overview():
    movie_data = pd.read_csv("data/movies_metadata.csv")
    # Assume high vote count == high viewer amount, thus using 90th percentile for more popular movie
    m = movie_data['vote_count'].quantile(.9)
    movie_data = movie_data[movie_data['vote_count'] > m]
    movie_data = movie_data[["original_title", "overview"]].reset_index(drop=True)
    movie_data['overview'] = movie_data['overview'].fillna('')
    return movie_data

def get_tfidf_matrix(df):
    # Using stop words to remove unneccasary words 
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(df['overview'])
    return matrix

def get_similarity(M):
    vector_cosine = linear_kernel(M, M)
    return vector_cosine

def movie_index(df):
    title_series = df['original_title']
    # Change all to lowercase to avoid case mismatch later on
    title_series = title_series.str.lower()
    return title_series

def pipeline():
    if os.path.isfile('process_data/similarity.pyb') and os.path.isfile('process_data/title_series.pyb'):
        similarity = pickle.load(open("process_data/similarity.pyb", "rb"))
        title_series = pickle.load(open("process_data/title_series.pyb", "rb"))
        return similarity, title_series

    movie_data = get_overview()
    title_series = movie_index(movie_data)
    M = get_tfidf_matrix(movie_data)
    movie_data = None
    similarity = get_similarity(M)
    M = None
    pickle.dump(similarity, open("process_data/similarity.pyb", "wb"))
    pickle.dump(title_series, open("process_data/title_series.pyb", "wb"))
    return similarity, title_series

def recommend(title, similarity, series, top=10):
    movie_ind = series[series == title].index[0]
    # list of movie index and movie similarity score
    movie_scores = list(enumerate(similarity[movie_ind]))
    movie_scores = sorted(movie_scores, key=lambda x : x[1], reverse=True)

    if top > len(movie_scores) - 1:
        top = len(movie_scores) - 1

    # top movie index, skip 1 as it's itself
    top_ind = [index for index, _ in movie_scores[1:top + 1]]
    top_movie = [series[i] for i in top_ind]
    return top_movie

def start_recommend():
    print("Processing Data...")
    similarity, title_series = pipeline()
    print("Data Ready")
    accepted = False
    while not accepted:
        title = input("Enter a Movie you previously like: ")
        title = title.lower()
        if title not in title_series.values:
            print("Movie not in Database. Try another Movie.")
        else:
            accepted = True
    amount = int(input("Enter number of Movies for recommendation (Max 50): "))
    if amount > 50 : amount = 50
    recommend_movie = recommend(title, similarity, title_series, amount)
    for rank, movie in enumerate(recommend_movie, 1):
        print(rank, movie)

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    start_recommend()    