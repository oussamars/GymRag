from google import genai
import os
from dotenv import load_dotenv
from google.genai import types
import chromadb



load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


query = "What time does the gym open?"
# the lines that are commented are needed in the first run but when the embedings are stored in the chroma we dont need this informations
# document = [
#         "The gym membership costs 300 MAD.",
#         "The gym opens at 9am and closes at 11pm",
#         "If the members payment status is not paid he should not enter the gym",
#         "Equipment is changed each 5 years unless an equipemnt is broken",
#         "Coach classes start at 5pm to 8pm",
# ]


result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=[query],
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
)

# vectors = [
#     result.embeddings[1].values,
#     result.embeddings[2].values,
#     result.embeddings[3].values,
#     result.embeddings[4].values,
#     result.embeddings[5].values,
# ]
# my_ids = ["id_1", "id_2", "id_3", "id_4", "id_5"]
# metadatas = [
#     {"category": "pricing", "source": "gym_faq"},
#     {"category": "schedule", "source": "gym_faq"},
#     {"category": "policy", "source": "payment_policy"},
#     {"category": "equipment", "source": "equipment_policy"},
#     {"category": "schedule", "source": "coach_policy"},
# ]

chroma_client = chromadb.PersistentClient(path="./chroma_data")

collection = chroma_client.get_or_create_collection(
    name="gym_knowledge",
    embedding_function=None,
    )



# collection.upsert(
#     ids=my_ids,
#     embeddings=vectors,
#     documents=document,
#     metadatas=metadatas,
# )

results = collection.query(
    query_embeddings=[result.embeddings[0].values],
    n_results=2,
    where={"category": "schedule"}
)

print(results)