import os
import numpy as np
import csv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

texts_folder = "texts"


def term_frequency(word, document):
    """
    Calculate the term frequency of a word in a document.

    Args:
        word (str): The word to calculate the frequency for.
        document (str): The document to search within.

    Returns:
        float: The frequency of the word in the document, defined as
               the count of the word divided by the total number of words in the document.
    """

    return document.count(word) / len(document)


def inverse_document_frequency(word, corpus):
    """
    Calculate the inverse document frequency of a word in a corpus.

    Args:
        word (str): The word to calculate the IDF for.
        corpus (list): The list of documents to consider.

    Returns:
        float: The IDF of the word in the corpus, defined as
               the logarithm of the total number of documents divided by the number of documents containing the word.
    """
    count_of_documents = len(corpus) + 1
    count_of_documents_with_word = sum([1 for doc in corpus if word in doc]) + 1
    idf = np.log10(count_of_documents / count_of_documents_with_word) + 1
    return idf


def tf_idf(word, document, corpus):
    """
    Calculate the TF-IDF score of a word in a document.

    Args:
        word (str): The word to calculate the TF-IDF score for.
        document (str): The document to search within.
        corpus (list): The list of documents to consider.

    Returns:
        float: The TF-IDF score of the word in the document, defined as the product of the term frequency and inverse document frequency.
    """

    return term_frequency(word, document) * inverse_document_frequency(word, corpus)


def markov_matrix(document, vocab):
    """
    Calculate a Markov matrix representing the probability of transitioning from one word to another in a document.

    Args:
        document (str): The document to generate the matrix for.
        vocab (list): The list of words to consider.

    Returns:
        The Markov transition matrix, represented as a flattened numpy array.
    """
    words = document.lower().split()
    m_matrix = np.zeros((len(vocab), len(vocab)), dtype=np.float64)

    word_to_index = {word: idx for idx, word in enumerate(vocab)}

    for i in range(len(words) - 1):
        word1 = words[i]
        word2 = words[i + 1]

        if word1 in word_to_index and word2 in word_to_index:
            m_matrix[word_to_index[word1]][word_to_index[word2]] += 1

    row_sums = m_matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        m_matrix = np.divide(m_matrix, row_sums)
        m_matrix[np.isnan(m_matrix)] = 0
    return m_matrix.flatten()


def classify(score):
    """
    Classify a given score as either "Low", "Moderate", or "High" based on the following ranges:
        Low: score < 0.4
        Moderate: 0.4 <= score < 0.7
        High: score >= 0.7

    Args:
        score (float): The score to be classified.

    Returns:
        str: The classification of the score.
    """
    if score < 0.4:
        return "Low"
    elif score < 0.7:
        return "Moderate"
    else:
        return "High"


# Leer texto original
with open("texts/original.txt", "r") as original_file:
    original_content = original_file.read()

# Crear un archivo CSV para almacenar los resultados
output_csv = "comparison_results.csv"
with open(output_csv, "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)

    csv_writer.writerow(
        [
            "Nombre original",
            "Nombre similar",
            "Coseno BOW",
            "Acertó BOW",
            "Coseno TFIDF",
            "Acertó TFIDF",
            "Coseno Markov",
            "Acertó Markov",
        ]
    )

    # Leer textos de comparación
    for filename in os.listdir(texts_folder):
        if filename != "original.txt" and filename.endswith(".txt"):
            with open(os.path.join(texts_folder, filename), "r") as file:
                file_content = file.read()

                texts = [original_content, file_content]
                vocab = list(
                    set(original_content.lower().split() + file_content.lower().split())
                )

                # ===== BoW (Bag of Words) =====
                bow_vectorizer = CountVectorizer()
                bow_matrix = bow_vectorizer.fit_transform(texts)
                bow_array = bow_matrix.toarray()

                original_bow_vec = bow_array[0].tolist()
                comparison_bow_vec = bow_array[1].tolist()

                cos_bow = cosine_similarity(bow_array[0:1], bow_array[1:2])[0][0]

                # ===== TF-IDF =====
                tfidf_vectorizer = TfidfVectorizer()
                tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
                tfidf_array = tfidf_matrix.toarray()

                original_tfidf_vec = tfidf_array[0].tolist()
                comparison_tfidf_vec = tfidf_array[1].tolist()

                cos_tfidf = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][
                    0
                ]

                # ===== Cadenas de Markov =====
                original_mark_vec = markov_matrix(original_content, vocab)
                comparison_mark_vec = markov_matrix(file_content, vocab)

                cos_mark = cosine_similarity(
                    [original_mark_vec], [comparison_mark_vec]
                )[0][0]

                print("\n")
                print(f"Comparando original.txt con {filename}...")
                print("=" * 65)
                print("Cosine Similarity (BoW):", cos_bow, "->", classify(cos_bow))
                print(
                    "Cosine Similarity (TF-IDF):", cos_tfidf, "->", classify(cos_tfidf)
                )
                print(
                    "Cosine Similarity (Cadenas de Markov):",
                    cos_mark,
                    "->",
                    classify(cos_mark),
                )

                acert_bow = cos_bow > 0.8
                acert_tfidf = cos_tfidf > 0.8
                acert_markov = cos_mark > 0.8

                csv_writer.writerow(
                    [
                        "original.txt",
                        filename,
                        cos_bow,
                        acert_bow,
                        cos_tfidf,
                        acert_tfidf,
                        cos_mark,
                        acert_markov,
                    ]
                )
