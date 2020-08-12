import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import warnings, os.path, pickle, re

def get_overview():
    movie_data = pd.read_csv("used_data/movies_metadata.csv")
    
    # Assume high vote count == high viewer amount, thus using 90th percentile could get us more popular movie
    m = movie_data['vote_count'].quantile(.9)

    # Get only vote count 90th percentile movies, and drop any duplicates with same movie title
    movie_data = movie_data[movie_data['vote_count'] > m].drop_duplicates(subset='original_title')
    
    # Data Cleanup by reseting messy index after cut and fill missing data
    movie_data = movie_data[["original_title", "overview"]].reset_index(drop=True)
    movie_data['overview'] = movie_data['overview'].fillna('')
    return movie_data

def get_tfidf_matrix(df):
    # Using stop words to remove unneccasary words 
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
    return title_series

def pipeline():
    # If the processed data has been saved before just use that
    if os.path.isfile('process_data/similarity.pyb') and os.path.isfile('process_data/title_series.pyb'):
        similarity = pickle.load(open("process_data/similarity.pyb", "rb"))
        title_series = pickle.load(open("process_data/title_series.pyb", "rb"))
        return similarity, title_series
    
    # Pipeline for processing the data and save data
    movie_data = get_overview()
    title_series = movie_index(movie_data)
    M = get_tfidf_matrix(movie_data)
    movie_data = None
    similarity = get_similarity(M)
    M = None
    pickle.dump(similarity, open("process_data/similarity.pyb", "wb"))
    pickle.dump(title_series, open("process_data/title_series.pyb", "wb"))
    return similarity, title_series

def searchText(title, series):
    for movie in series:
        # Search if user input title is part of a movie's fullname
        search = re.search(title + r'\s', movie)
        if search:
            print("\nYour inputed Movie is not in Database. We assume you meant: " + movie.title())
            return True, movie
    print("Movie not in Database. Try another Movie.\n")
    return False, None


def recommend(title, similarity, series, top=10):
    # Get the index of the user's input movie
    movie_ind = series[series == title].index[0]

    # List of movie index and movie similarity score
    movie_scores = list(enumerate(similarity[movie_ind]))
    movie_scores = sorted(movie_scores, key=lambda x : x[1], reverse=True)

    # Prevent going out of Movie List
    if top > len(movie_scores) - 1:
        top = len(movie_scores) - 1

    # Get index of top movies, skip 1 as it's itself
    top_ind = [index for index, _ in movie_scores[1:top + 1]]
    top_movie = [series[i] for i in top_ind]
    return top_movie

def start_recommend():
    # Start pipeline to get necessary data
    print("\nProcessing Data...")
    similarity, title_series = pipeline()
    print("Data Ready!\n")

    # Let user input a Movie that's in our database
    accepted = False
    while not accepted:
        title = input("Enter a Movie you previously like: ")
        title = title.lower()
        if title not in title_series.values:
            # Use searchText function to check if movies of similar titles exist
            accepted, title = searchText(title, title_series)
        else:
            accepted = True

    # Limit the max amount for better viewing        
    amount = int(input("\nEnter number of Movies for recommendation (Max 50): "))
    if amount > 50 : amount = 50

    # Get list of Recommender Movie
    recommend_movie = recommend(title, similarity, title_series, amount)
    print("\nMovie Recommendation for You:")
    for rank, movie in enumerate(recommend_movie, 1):
        print(rank, movie.title())

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    start_recommend()    