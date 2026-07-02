from google import genai
import os
from dotenv import load_dotenv
from google.genai import types
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import math

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))

    magnitude_a = math.sqrt(sum(a * a for a in vec1))
    magnitude_b = math.sqrt(sum(b * b for b in vec2))

    return dot_product / (magnitude_a * magnitude_b)

query = "What time does the gym open?"

document = [
        "The gym membership costs 300 MAD.",
        "The gym opens at 9am and closes at 11pm",
        "If the members payment status is not paid he should not enter the gym",
        "Equipment is changed each 5 years unless an equipemnt is broken",
        "Coach classes start at 5pm to 8pm",
]

result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=[query] + document,
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
)
query_vector = result.embeddings[0].values
print(result.embeddings[2].values)

scores = []

for i, doc_embedding in enumerate(result.embeddings[1:]):
    score = cosine_similarity(
        query_vector,
        doc_embedding.values
    )
    scores.append((document[i], score))

for i in range(5):
    print(f"score{i}: {scores[i]}")