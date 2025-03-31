from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv

with open("questions.csv", "r") as file:
    reader = csv.DictReader(file)
    questions = [(row["question1"], row["question2"]) for row in reader]

with open("similarity.csv", "w", newline="") as output_file:
    writer = csv.writer(output_file)
    writer.writerow(
        ["question1", "question2", "cosine_distance", "q1_vector", "q2_vector"]
    )

for q1, q2 in questions:
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform([q1, q2])
    vectors_array = vectors.toarray()

    similarity = cosine_similarity(vectors_array[0:1], vectors_array[1:2])

    with open("similarity.csv", "a", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                q1,
                q2,
                similarity[0][0],
                vectors_array[0].tolist(),
                vectors_array[1].tolist(),
            ]
        )
