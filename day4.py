from google import genai
import os
from dotenv import load_dotenv
from google.genai import types
import chromadb

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


chroma_client = chromadb.PersistentClient(path="./chroma_data")


def chunk_and_embed(text):

    chunks = []
    for chunk in text.split("\n\n"):
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)

    paragraph_results = client.models.embed_content(
            model="gemini-embedding-001",
            contents=chunks,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
    )
    
    vectors = get_vectors(paragraph_results)
    ids = create_ids("paragraph", len(chunks))

    list_of_dicts = []
    for i in range(len(chunks)):
        chunk_dict = {"id": ids[i], "text": chunks[i], "embedding": vectors[i], "metadata": {"source": "gym_faq", "chunk_index": i, "chunk_type": "paragraph"}}
        list_of_dicts.append(chunk_dict)

    return(list_of_dicts)

def store_in_chroma(list_of_dicts):
    collection = chroma_client.get_or_create_collection(
        name="gym_chunks",
        embedding_function=None,
    )

    collection.upsert(
        ids=[chunk["id"] for chunk in list_of_dicts],
        embeddings=[chunk["embedding"] for chunk in list_of_dicts],
        documents=[chunk["text"] for chunk in list_of_dicts],
        metadatas=[chunk["metadata"] for chunk in list_of_dicts],
    )

    return collection

def retrieve(question, collection, n_results=3):
    query_embeding = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question,
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
    )

    results = collection.query(
        query_embeddings=[query_embeding.embeddings[0].values],
        n_results=n_results,
        where={"source": "gym_faq"}
    )

    return results["documents"][0]


def answer(result, question):
    context = "\n\n".join(result)
    content = f"""
    <context>
    {context}
    </context>

    <question>
    {question}
    </question>

"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
        system_instruction="""You are GymAssistant, an AI assistant for a gym management platform.
        Answer ONLY using the information inside the <context> section.
        If the answer is not present in the context, say you don't have enough information.
        Keep answers concise and under 150 words.""",
        temperature=0.3,
        ),
        contents=content
    )
    return response.text


def get_vectors(result):
    vectors = []
    for embedding in result.embeddings:
        vectors.append(embedding.values)
    return vectors

def create_ids(name, num):
    ids = []
    for i in range(num):
        ids.append(f"{name}_{i}")
    return ids

def load_document(file_path):
    
    try:
        with open(file_path, "r") as file:
            text = file.read()
            if not text:
                return (None, "Error: File is empty.")
            return (text, None)
    except FileNotFoundError:
        return (None, f"Error: The file at '{file_path}' does not exist.")


def main():
    question = "where is your gym located"
    collection = chroma_client.get_collection("gym_chunks")
    if (collection.count() == 0):
        text, error = load_document("gym_faq.txt")
        if error:
            print(error)
            return
        collection = store_in_chroma(chunk_and_embed(text))
    result = retrieve(question, collection, 3)
    final_answer = answer(result, question)

    print(final_answer)
    


main()