import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample dataset to work with
data = {
    'title': [
        'The Godfather', 'The Dark Knight', 'Pulp Fiction', 'The Shawshank Redemption', 
        'The Godfather Part II', 'The Dark Knight Rises', 'Django Unchained',
        'Inception', 'Fight Club', 'Forrest Gump'
    ],
    'overview': [
        'The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.',
        'When the menace known as the Joker emerges from his mysterious past, he wreaks havoc and chaos on the people of Gotham.',
        'The lives of two mob hitmen, a boxer, a gangster, and his wife intertwine in four tales of violence and redemption.',
        'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.',
        'The early life and career of Vito Corleone in 1920s New York is portrayed, while his son, Michael, expands and tightens his grip on the family crime syndicate.',
        'Eight years after the Joker\'s reign of anarchy, Batman is compelled to return to defend Gotham City from the brutal guerrilla terrorist Bane.',
        'With the help of a German bounty hunter, a freed slave sets out to rescue his wife from a brutal Mississippi plantation owner.',
        'A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O.',
        'An insomniac office worker and a devil-may-care soapmaker form an underground fight club that evolves into something much more.',
        'The presidencies of Kennedy and Johnson, Vietnam, Watergate, and other historical events unfold through the perspective of an Alabama man with an IQ of 75.'
    ]
}

# Convert the data into a DataFrame
movies = pd.DataFrame(data)

# Preprocessing - Fill NaN values with empty strings (not needed here as no NaNs in the sample)
movies['overview'] = movies['overview'].fillna('')

# Define a TF-IDF Vectorizer Object. Remove all English stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

# Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(movies['overview'])

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create a reverse mapping of movie titles to indices
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    if title not in indices:
        print(f"Title '{title}' not found in dataset. Please check the title and try again.")
        return []
    
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 5 most similar movies
    sim_scores = sim_scores[1:6]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 5 most similar movies
    return movies['title'].iloc[movie_indices]

# Example usage
print("Recommended movies for 'The Godfather':")
print(get_recommendations('The Godfather'))