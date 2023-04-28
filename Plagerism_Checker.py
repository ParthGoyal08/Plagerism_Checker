import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def preprocess_text(text):
    # tokenize text
    tokens = word_tokenize(text)

    # remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [
        token for token in tokens if token.lower() not in stop_words]

    # lemmmatize_tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(
        token) for token in filtered_tokens]

    # join tokens back into text string
    preprocessed_text = ''.join(lemmatized_tokens)
    return preprocessed_text


def calculate_similarity(text1, text2):

    # preprocess both texts
    preprocessed_text1 = preprocess_text(text1)
    preprocessed_text2 = preprocess_text(text2)

    # create NLTK Text Objects
    nltk_text1 = nltk.Text(word_tokenize(preprocessed_text1))
    nltk_text2 = nltk.Text(word_tokenize(preprocessed_text2))

    # calculate Jaccard Similarity
    jaccard_similarity = nltk.jaccard_distance(
        set(nltk_text1), set(nltk_text2))
    return 1-jaccard_similarity


text1 = "Parth is Bald. The quick brown fox jumps over the lazy dog."
text2 = "Parth is not Bald. The quick brown fox jumps through the hoops of life."
similarity = calculate_similarity(text1, text2) * 100
print(f"Similarity: {similarity}% similar")

# I added two things to change it to percentage. The * 100 and the % similar in the last two lines. lol