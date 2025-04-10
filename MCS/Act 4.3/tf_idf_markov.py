import numpy as np
import csv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def term_frequency(word, document):
    return document.count(word) / len(document)


def inverse_document_frequency(word, corpus):
    count_of_documents = len(corpus) + 1
    count_of_documents_with_word = sum([1 for doc in corpus if word in doc]) + 1
    idf = np.log10(count_of_documents / count_of_documents_with_word) + 1
    return idf


def tf_idf(word, document, corpus):
    return term_frequency(word, document) * inverse_document_frequency(word, corpus)


def markov_matrix(document, vocab):
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


# Leer las preguntas desde el archivo
with open("Act 4.3/similarity.csv", "r") as file:
    reader = csv.DictReader(file)
    questions = [(row["question1"], row["question2"]) for row in reader]


# Crear el archivo de salida con encabezados
with open("Act 4.3/tf_idf_markov.csv", "w", newline="") as output_file:
    writer = csv.writer(output_file)
    writer.writerow(
        [
            "question1",
            "question2",
            "cos_BOW",
            "cos_TFID",
            "cos_MARK",
            "q1_vecBoW",
            "q2_vecBoW",
            "q1_vecTFIDF",
            "q2_vecTFIDF",
            "q1_vecMark",
            "q2_vecMark",
        ]
    )

    # for q1, q2 in questions:
    q1, q2 = questions[0]
    texts = [q1, q2]
    vocab = list(set(q1.lower().split() + q2.lower().split()))

    # ===== BoW (Bag of Words) =====
    bow_vectorizer = CountVectorizer()
    bow_matrix = bow_vectorizer.fit_transform(texts)
    bow_array = bow_matrix.toarray()

    q1_bow_vec = bow_array[0].tolist()
    q2_bow_vec = bow_array[1].tolist()

    cos_bow = cosine_similarity(bow_array[0:1], bow_array[1:2])[0][0]

    # ===== TF-IDF =====
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    tfidf_array = tfidf_matrix.toarray()

    q1_tfidf_vec = tfidf_array[0].tolist()
    q2_tfidf_vec = tfidf_array[1].tolist()

    cos_tfidf = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    # ===== Cadenas de Markov =====
    q1_mark_vec = markov_matrix(q1, vocab)
    q2_mark_vec = markov_matrix(q2, vocab)

    cos_mark = cosine_similarity([q1_mark_vec], [q2_mark_vec])[0][0]

    # Escribir resultados
    writer.writerow(
        [
            q1,
            q2,
            cos_bow,
            cos_tfidf,
            cos_mark,
            q1_bow_vec,
            q2_bow_vec,
            q1_tfidf_vec,
            q2_tfidf_vec,
            q1_mark_vec,
            q2_mark_vec,
        ]
    )
