from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv

# Leer las preguntas desde el archivo
with open("MCS/similarity.csv", "r") as file:
    reader = csv.DictReader(file)
    questions = [(row["question1"], row["question2"]) for row in reader]

# Crear el archivo de salida con encabezados
with open("MCS/Act 4.3/similarity_bow_tf_idf.csv", "w", newline="") as output_file:
    writer = csv.writer(output_file)
    writer.writerow(
        [
            "question1",
            "question2",
            "cosine_distance_bow",
            "cosine_distance_tfidf",
            "tfidf_vector_q1",
            "tfidf_vector_q2",
        ]
    )

    # Procesar cada par de preguntas
    for q1, q2 in questions:
        # Vectorización BoW
        bow_vectorizer = CountVectorizer()
        bow_vectors = bow_vectorizer.fit_transform([q1, q2]).toarray()
        bow_similarity = cosine_similarity([bow_vectors[0]], [bow_vectors[1]])[0][0]

        # Vectorización TF-IDF
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_vectors = tfidf_vectorizer.fit_transform([q1, q2]).toarray()
        tfidf_similarity = cosine_similarity([tfidf_vectors[0]], [tfidf_vectors[1]])[0][
            0
        ]

        writer.writerow(
            [
                q1,
                q2,
                bow_similarity,
                tfidf_similarity,
                tfidf_vectors[0].tolist(),
                tfidf_vectors[1].tolist(),
            ]
        )
