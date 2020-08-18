import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import warnings
from os import path
import pickle
import re
import numpy as np
import random as rd

def get_overview():
    movie_data = pd.read_csv("used_data/movies_metadata.csv")
    
    # Assume high vote count == high viewer amount, thus using 90th percentile could get us more popular movie
    m = movie_data['vote_count'].quantile(.9)

    # Get only vote count 90th percentile movies, and drop any duplicates with same movie title
    movie_data = movie_data[movie_data['vote_count'] > m].drop_duplicates(subset='original_title')
    
    # Sort alphabetically for better searching later on
    movie_data = movie_data.sort_values(by=['original_title'])

    # Data Cleanup by reseting messy index after cut and fill missing data
    movie_data = movie_data[["original_title", "overview"]].reset_index(drop=True)
    movie_data['overview'] = movie_data['overview'].fillna('')
    return movie_data

def get_tfidf_matrix(df):
    # Using TF-IDF with stop words to remove unneccasary words 
    tfidf = TfidfVectorizer(stop_words="english")
    
    # Create TF-IDF Matrix for Movie overview
    matrix = tfidf.fit_transform(df['overview'])
    return matrix

def get_similarity(M):
    # Get pair similarity of each Movie
    vector_cosine = linear_kernel(M, M)
    return vector_cosine

def movie_index(df):
    title_series = df['original_title']

    # Change all to lowercase to avoid case mismatch later on
    title_series = title_series.str.lower()
    return list(title_series)

def pipeline():
    # If the processed data has been saved before just use that
    if path.isfile('process_data/similarity.pyb') and path.isfile('process_data/title_series.pyb'):
        similarity = pickle.load(open("process_data/similarity.pyb", "rb"))
        title_series = pickle.load(open("process_data/title_series.pyb", "rb"))

    # Pipeline for processing the data and save data
    else:    
        movie_data = get_overview()
        title_series = movie_index(movie_data)
        M = get_tfidf_matrix(movie_data)
        similarity = get_similarity(M)
        movie_data, M = None, None
        pickle.dump(similarity, open("process_data/similarity.pyb", "wb"))
        pickle.dump(title_series, open("process_data/title_series.pyb", "wb"))
    return similarity, title_series

def searchText(title, series):
    for movie in series:
        # Remove any symbols or punctuations for better search
        clean_title = re.sub(r'[^\s\w]', ' ', movie)

        # Search if part of user's input title is part of a movie's fullname
        name_search = re.search(r'(^|[ ]){}\s'.format(title), clean_title)
        split_search =  re.search(r'(^|[ ]){}\s'.format(title.split()[0]), clean_title)

        search = name_search or split_search
        if search:
            print("\nYour inputed Movie is not in Database. We assume you meant: " + movie.title())
            return True, movie
    print(title.title() + " not in Movie Database. Try another Movie.")
    return False, None

def db_check(title, title_series):
    title = title.lower()
    if title not in title_series:
        # Use searchText function to check if movies of similar titles exist
        accepted, title = searchText(title, title_series)
    else:
        accepted = True
    return accepted, title

def display_recommend(recommend_movie):
    print("\nMovie Recommendation for You:")
    for rank, movie in enumerate(recommend_movie, 1):
        print(rank, movie.title())


def recommend(title, similarity, series, top=10):
    # Get the index of the user's input movie
    movie_ind = series.index(title)

    # List of movie index and movie similarity score
    movie_scores = list(enumerate(similarity[movie_ind]))
    
    # Sort from highest similarity score to lowest
    movie_scores = sorted(movie_scores, key=lambda x : x[1], reverse=True)

    # Prevent going out of Movie List
    if top > len(movie_scores) - 1:
        top = len(movie_scores) - 1

    # Get index of top movies, skip 1 as it's itself
    top_ind = [index for index, _ in movie_scores[1:top + 1]]

    # Get title of top movies with given index
    top_movie = [series[i] for i in top_ind]
    return top_movie

def history_rec(hist):
    print("\nProcessing Data...")
    similarity, title_series = pipeline()
    print("Data Ready!\n")

    accepted_movie = sorted([db_check(t, title_series) for t in hist])
    accepted_movie = [m for b, m in accepted_movie if b]

    ind = [title_series.index(m) for m in accepted_movie]

    avg_score = np.zeros(len(similarity))
    for i in ind:
        avg_score += np.asarray(similarity[i])
    avg_score /= len(ind)

    movie_score = list(enumerate(avg_score))
    movie_score = sorted(movie_score, key=lambda x : x[1], reverse=True)

    top_ind = [index for index, _ in movie_score[1:11]]
    top_movie = [title_series[i] for i in top_ind]

    print("\nSome of Your Movie History: {}".format(accepted_movie[:10]))

    display_recommend(top_movie)

def start_recommend():
    # Start pipeline to get necessary data
    print("\nProcessing Data...")
    similarity, title_series = pipeline()
    print("Data Ready!\n")

    # Let user input a Movie that's in our database
    accepted = False
    while not accepted:
        title = input("Enter a Movie you previously like: ")
        accepted, title = db_check(title, title_series)

    # Limit the max amount for better viewing        
    amount = int(input("\nEnter number of Movies for recommendation (Max 50): "))
    if amount > 50 : 
        amount = 50

    # Get list of Recommender Movie
    recommend_movie = recommend(title, similarity, title_series, amount)
    display_recommend(recommend_movie)

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    #start_recommend()    
    _, title_series = pipeline()
    x = rd.randint(0, len(title_series)//2)
    y = rd.randint(len(title_series)//2, len(title_series))

    hist = title_series[x:y]
    history_rec(hist)