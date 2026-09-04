from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model():

    return SentenceTransformer(MODEL_NAME)


def calculate_similarity(
    model,
    question_a,
    question_b
):

    embeddings = model.encode(
        [
            question_a,
            question_b
        ]
    )

    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    return float(similarity)