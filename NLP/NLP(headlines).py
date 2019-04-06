#Predicting Hacker News upvotes using headlines

from collections import Counter
import pandas as pd

headlines = [
    "PretzelBros, airbnb for people who like pretzels, raises $2 million",
    "Top 10 reasons why Go is better than whatever language you use.",
    "Why working at apple stole my soul (I still love it though)",
    "80 things I think you should do immediately if you use python.",
    "Show HN: carjack.me -- Uber meets GTA"
]

# Find all the unique words in the headlines.
unique_words = list(set(" ".join(headlines).split(" ")))

def make_matrix(headlines, vocab):
    matrix = []
    for headline in headlines:
        # Count each word in the headline, and make a dictionary.
        counter = Counter(headline)
        # Turn the dictionary into a matrix row using the vocab.
        row = [counter.get(w, 0) for w in vocab]
        matrix.append(row)
    df = pd.DataFrame(matrix)
    df.columns = unique_words
    return df

print(make_matrix(headlines, unique_words))

#Removing punctuation
#The matrix we just made is very sparse — that means that a lot of the values are zero. 
#This is unavoidable to some extent, because the headlines don't have much shared vocabulary.
# We can take some steps to make the problem better, though.
# Right now Why and why, and use and use. are treated as different entities, but we know they refer
# to the same word.
#
#We can help the parser recognize that these are in fact the same by lowercasing every word and removing all punctuation.

import re

# Lowercase, then replace any non-letter, space, or digit character in the headlines.
new_headlines = [re.sub(r'[^\w\s\d]','',h.lower()) for h in headlines]

# Replace sequences of whitespace with a space character.
new_headlines = [re.sub("\s+", " ", h) for h in new_headlines]

unique_words = list(set(" ".join(new_headlines).split(" ")))

# We've reduced the number of columns in the matrix a bit.
print(make_matrix(new_headlines, unique_words))

# Read in and split the stopwords file.
with open("stop_words.txt", 'r') as f:
    stopwords = f.read().split("\n")
    
# Do the same punctuation replacement that we did for the headlines, 
# so we're comparing the right things.
stopwords = [re.sub(r'[^\w\s\d]','',s.lower()) for s in stopwords]

unique_words = list(set(" ".join(new_headlines).split(" ")))
# Remove stopwords from the vocabulary.
unique_words = [w for w in unique_words if w not in stopwords]

# We're down to 34 columns, which is way better!
print(make_matrix(new_headlines, unique_words))



from sklearn.feature_extraction.text import CountVectorizer

# Construct a bag of words matrix.
# This will lowercase everything, and ignore all punctuation by default.
# It will also remove stop words.
vectorizer = CountVectorizer(lowercase=True, stop_words="english")

matrix = vectorizer.fit_transform(headlines)
# We created our bag of words matrix with far fewer commands.
print(matrix.todense())