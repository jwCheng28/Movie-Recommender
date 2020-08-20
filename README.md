# Movie Recommender
This project attempts to build a very simple Recommender System that recommends Movies to the User.

Currently, there's 2 type of recommender system being implemented in this project:
- Basic Scoring Recommender
- Content Based Recommender

## Basic Scoring Recommender
This recommender is fairly straight forward; this is a rather general approach as we recommend movies to all users based on a scoring mechanism. The code for this could be found in Basic_Ranking.ipynb.

### How to Implement:
For this project, we'll score all the movies based on IMDb's rating formula: 

<img src="https://render.githubusercontent.com/render/math?math=\large WeightedRating = \frac{v\times{R}%2Bm\times{C}}{v%2Bm}">

- v: Number of Votes
- m: Minumum of Votes to be considered
- R: Average Rating of Movie
- C: Mean Rating across all Movies

Using this formula we could calculate scores for our movies, and recommend them to the user from highest to lowest score.
### Limitation:
This way of recommending Movie, we're over generalizing our user's by assuming they'll all like the same type of Movies based on this scoring mechanism. 

## Content Based Recommender
In this recommender we'll analyze the content of each Movies, and recommend the user with Movies that have similiar content with what they've previously liked.

### How to Implement:
In our dataset, there's a column with a brief overviews of each Movie. These text aren't necessary useful because the computer wont understand the semantics of behind them. So what we could do is turn them into vectors that're useful for computation.

### Vectorization:
The simplest way to vectorize words is using Count Vectorization, where we simply turn words into vector based on the frequency of each word.
```
ex. John has a dog named John. -> [John, has, a, dog, named] -> [2, 1, 1, 1, 1]
```
However, there's a problem with this method; as you can imagine words like 'a', 'an', 'is' or etc. would have a relatively high frequency but doesn't really contribute much on the context. Thus, a better method we could use is TF-IDF (term frequency–inverse document frequency).

Where TF is the same as Count Vectorization which counts the frequency of each word, and IDF is the log of the total number of documents divided by the number of documents that contain that word. 

![alt text](https://github.com/jwCheng28/Movie-Recommender/blob/master/pics/tfidf.png)

This basically ensures rarer words have a higher weight than those with high frequency; which prevents the problem in Count Vectorization of overweighting meaningless words.

### Similarity:
After vectorizing all the movie overviews, we could simply compute the cosine similarity of between each movies via these vectors to find out how similar are these different movies. 

Then, we could sort all the movies based on their similarity score to the user's favorite movie; and the those with high similarity score would be recommended to the user.
