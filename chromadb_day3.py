from google import genai
import os
from dotenv import load_dotenv
from google.genai import types
import chromadb

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

chroma_client = chromadb.PersistentClient(path="./chroma_data")


def embed_chromadb_collections():

    query = "What equipment does the gym have?"

    fixed_chunk = chunk_fixed("gym_faq.txt", 600, 150)
    paragraph_chunk = chunk_by_paragraph("gym_faq.txt")

    fixed_results = client.models.embed_content(
            model="gemini-embedding-001",
            contents=[query] + fixed_chunk,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
    )

    vector_fixed_collection = get_vectors(fixed_results)
    ids_fixed_collection = create_ids("fixed", len(fixed_chunk))
    fixed_metadata = []
    for i in range(len(fixed_chunk)):
        fixed_metadata.append({"source": "gym_faq", "chunk_index": i, "chunk_type": "fixed"})
    

    paragraph_results = client.models.embed_content(
            model="gemini-embedding-001",
            contents=[query] + paragraph_chunk,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
    )
    vector_paragraph_collection = get_vectors(paragraph_results, len(paragraph_chunk))
    ids_paragraph_collection = create_ids("paragraph", len(paragraph_chunk))
    paragraph_metadata = []
    for i in range(len(paragraph_chunk)):
        paragraph_metadata.append({"source": "gym_faq", "chunk_index": i, "chunk_type": "paragraph"})


    fixed_collection = chroma_client.get_or_create_collection(
        name="gym_fixed_chunks",
        embedding_function=None,
        )
    
    fixed_collection.upsert(
    ids=ids_fixed_collection,
    embeddings=vector_fixed_collection,
    documents=fixed_chunk,
    metadatas=fixed_metadata,
)
    
    paragraph_collection = chroma_client.get_or_create_collection(
        name = "gym_paragraph_chunks",
        embedding_function=None,
    )

    paragraph_collection.upsert(
        ids=ids_paragraph_collection,
        embeddings=vector_paragraph_collection,
        documents=paragraph_chunk,
        metadatas=paragraph_metadata
    )



    results_fixed = fixed_collection.query(
        query_embeddings=[fixed_results.embeddings[0].values],
        n_results=2,
        where={"source": "gym_faq"},
    )

    results_paragraph = paragraph_collection.query(
        query_embeddings=[paragraph_results.embeddings[0].values],
        n_results=2,
        where={"source": "gym_faq"},
    )

    # print(f"fixed results:\n{results_fixed}\n\nparagraph results:\n{results_paragraph}")
    print(f"fixed results:\n{results_fixed}")





    

def get_vectors(result):
    vectors = []
    for embedding in result.embeddings[1:]:
        vectors.append(embedding.values)
    return vectors

def create_ids(name, num):
    ids = []
    for i in range(num):
        ids.append(f"{name}_{i}")
    return ids

def chunk_fixed(file_path, chunk_size, overlap):
    with open(file_path, "r") as file:
        text = file.read()

    chunks = []
    step = chunk_size - overlap

    if step <= 0:
        return "step should be a positive number"

    for i in range(0, len(text), step):
        chunks.append(text[i:i + chunk_size])

        if i + chunk_size >= len(text):
            break

    return chunks

def chunk_by_paragraph(file_path):
    with open(file_path, "r") as file:
        text = file.read()

    chunks = []
    i = 0
    chunk = ""

    while i + 1 < len(text):
        chunk += text[i]

        if text[i] == "\n" and text[i + 1] == "\n":
            chunks.append(chunk.strip())
            chunk = ""
            i += 1

        i += 1

    if chunk.strip():
        chunks.append(chunk.strip())

    return chunks


def main():
    embed_chromadb_collections()

main()