import spacy

nlp = spacy.load('en_core_web_md')


# Read movies.txt file, separate the titles and descriptions, store in a dictionary and return the dictionary.
# strip() and lower() used on the descriptions to avoid whitespace and case issues.
def read_movies():
    movie_dict = {}
    with open("movies.txt", "r") as f:
        for line in f:
            movie_dict[line.split(":")[0].strip()] = line.split(":")[1].strip().lower()
    return movie_dict


# Function to remove tokens that would not be relevant to determining a similar film, such as "the", "or" etc. Also
# removes any punctuation tokens as well as any tokens that only consist of whitespace.
# https://spacy.io/api/token - used for reference for is_stop, is_punct and is_space
# Builds list of non-filtered tokens, converts to string to allow it to be turned back into tokens and returned.
def filter_tokens(tokens):
    results = []
    for t in tokens:
        if not t.is_stop and not t.is_punct and not t.is_space:
            results.append(t.lemma_)
    filtered_string = " ".join(results)
    filtered = nlp(filtered_string)
    return filtered


# Gets tokens for description (and uses lower() to avoid issues with case) and passes to filter_tokens() to return
# keywords within the description.
def compare_movies(description):
    movie_dict = read_movies()
    similarity_dict = {}
    desc_nlp = nlp(description.lower())
    desc_tokens = filter_tokens(desc_nlp)

    # Iterates through dictionary of movies, getting tokens and passing to filter_tokens() to remove stop
    # words/punctuation/whitespace. Compares similarity of the movie description to the input description and stores
    # the similarity score in a dictionary with the movie name as the key.
    for suggest_name, suggest_desc in movie_dict.items():
        suggest_nlp = nlp(suggest_desc)
        suggest_token = filter_tokens(suggest_nlp)
        similarity_dict[suggest_name] = desc_tokens.similarity(suggest_token)

    # https://docs.python.org/3/howto/sorting.html used as reference for sorted()
    # Uses sorted() on similarity_dict to sort by the similarity values. As shown in the documentation, when used on
    # a dictionary it usually sorts the keys of the dict and returns a list of the keys. As we want to sort by the
    # similarity rating, the key parameter was set to similarity_dict.get
    # The documentation states "The value of the key parameter should be a function (or other callable) that takes a
    # single argument and returns a key to use for sorting purposes.", meaning that using similarity_dict.get will
    # fetch the similarity rating for each key and use this for sorting instead of the key, but still sort the keys in
    # order of the similarity rating.
    # reverse=True used as by default sorted() sorts by descending, and [0] used to return only the highest value.
    most_similar = sorted(similarity_dict, key=similarity_dict.get, reverse=True)[0]
    return most_similar


# Used for testing
# in_description = """Will he save
# their world or destroy it? When the Hulk becomes too dangerous for the
# Earth, the Illuminati trick Hulk into a shuttle and launch him into space to a
# planet where the Hulk can live in peace. Unfortunately, Hulk land on the
# planet Sakaar where he is sold into slavery and trained as a gladiator."""

in_description = input("Enter a movie description: ")
movie = compare_movies(in_description)
print(f"\n\nThe most similar movie is {movie}!")
