import nltk    # We are using NLTK for basic text operations.
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer     #Scikit-learn gives us the TF-IDF calculation.

# Download required NLTK data files
# These lines download English data (only once needed) — like stopwords list, dictionaries for lemmatizer, etc.
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

# Sample document
document = "Cats are running faster than dogs. The little cat chased the big dog."

# 1. Tokenization
# Break the text into individual words and punctuation.
# Example: ['Cats', 'are', 'running', 'faster', 'than', 'dogs', '.', ...]
tokens = word_tokenize(document)
print("\nTokens:", tokens)

# 2. POS Tagging
# Finds whether the word is noun, verb, adjective, etc.
# Example: ('Cats', 'NNS') → NNS = plural noun
pos_tags = pos_tag(tokens)
print("\nPOS Tags:", pos_tags)

# 3. Stop Words Removal
# Removes common words like is, are, the, than, etc.
# Because they don't add much meaning.
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
print("\nAfter Stop Words Removal:", filtered_tokens)

# 4. Stemming
# Reduce words to their root form roughly.
# Example: running → run, chased → chase
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
print("\nAfter Stemming:", stemmed_tokens)

# 5. Lemmatization
# Better than stemming — it uses grammar rules.
# Example: running → run, cats → cat
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
print("\nAfter Lemmatization:", lemmatized_tokens)

# 6. TF-IDF Representation
# It finds:
# TF (Term Frequency): How many times a word appears.
# IDF (Inverse Document Frequency): How rare/important a word is across documents.
# It finally shows numbers that tell how important each word is.
documents = [document]
vectorizer = TfidfVectorizer()
TFIDFF_matrix = vectorizer.fit_transform(documents)

print("\nTF-IDF Matrix:")
print(TFIDFF_matrix.toarray())

print("\nTF-IDF Feature Names:")
print(vectorizer.get_feature_names_out())