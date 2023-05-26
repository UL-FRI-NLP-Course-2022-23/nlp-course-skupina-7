from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score


def calculate_cosine_similarity(text1, text2):
    # Create a TfidfVectorizer object
    vectorizer = TfidfVectorizer()

    # Fit and transform the vectorizer on the given texts
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    # Calculate the cosine similarity between the vectors
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

    return similarity

def calculate_bleu_similarity(text1, text2):
    bleu_score = sentence_bleu([text1.split()], text2.split())
    return bleu_score

def calculate_meteor_score(text1, text2):
    meteor_score_value = meteor_score([text1.split()], text2.split())

    return meteor_score_value
