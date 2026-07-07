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


def retrieve(question, collection, n_results=5):
    query_embeding = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question,
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
    )

    results = collection.query(
        query_embeddings=[query_embeding.embeddings[0].values],
        n_results=n_results,
        where={"source": "gym_faq"},
        include=["documents", "distances"]
    )

    distance_threshold = 0.33
    filtered_results = []
    for doc, distance in zip(results["documents"][0], results["distances"][0]):
        if distance <= distance_threshold:
            filtered_results.append(doc)
    print(f"retrival:\n{filtered_results}")
    return filtered_results


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


def answer(history, result, prompt, session_facts_list):
    if len(result) == 0:
        context = ""
    else:  
        context = "\n\n".join(result)
    session_facts = "\n".join(session_facts_list)
    content = f"""
    <session_facts>
    {session_facts}
    </session_facts>

    <context>
    {context}
    </context>

    <question>
    {prompt}
    </question>

"""
    temp_history = history.copy()
    temp_history.append(
        types.Content(
            role="user",
            parts=[
                types.Part(text=content)
            ]
        )
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
        system_instruction="""You are GymAssistant, an AI assistant for a gym management platform.

        Use information from the <context> section to answer questions about the gym.

        Use information from <session_facts> only for facts that the user has shared during this conversation, such as their name, preferences, or membership type. When using these facts, make it clear that they were provided by the user.

        If the answer cannot be found in either <context> or <session_facts>, say that you don't have enough information.

        Keep answers concise and under 150 words.""",
        temperature=0.3,
        ),
        contents=temp_history
    )
    print("flash model api called")

    history.append(
        types.Content(
            role="user",
            parts=[
                types.Part(text=prompt)
            ]
        )
    )

    history.append(
        types.Content(
            role="model",
            parts=[
                types.Part(text=response.text)
            ]
        )
    )

    return response.text

def rewrite_query(question, history):
    if not history:
        return question
    else:
        temp_history = history.copy()
        temp_history.append(
        types.Content(
            role="user",
            parts=[
                types.Part(text=f"The current user question is:\n{question}")
            ]
        )
    )
        response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        config=types.GenerateContentConfig(
        system_instruction="""You are a search query optimizer. Given a conversation history and a 
                follow-up question, rewrite the question into a single standalone search 
                query that captures the user's full intent without requiring conversation 
                context. Return only the rewritten query, nothing else.""",
        temperature=0.3,
        ),
        contents=temp_history
    )
    print("flash-lite model api called")
    print(f"user question: {question}\ncontext generated question: {response.text.strip()}\n")
    return response.text.strip()

def session_facts(prompt):
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        config=types.GenerateContentConfig(
        system_instruction="""Extract any user-specific facts that could be useful later in the conversation,
        such as their name, preferences, membership type, goals, or other persistent information.

        If there are no useful facts, return an empty string.
        Return only the extracted facts.""",
        temperature=0.3,
        ),
        contents=prompt
    )
    print("flash-lite model api called")

    return response.text



def main():
    history = []
    session_facts_list = []
    while(True):
        prompt = input("You: ")
        if (prompt == "quit"):
            break
        collection = chroma_client.get_collection("gym_chunks")
        if (collection.count() == 0):
            text, error = load_document("gym_faq.txt")
            if error:
                print(error)
                return
            collection = store_in_chroma(chunk_and_embed(text))
        fact = session_facts(prompt)
        if fact:
            session_facts_list.append(fact)
        
        search_query = rewrite_query(prompt, history)
        result = retrieve(search_query, collection, 5)
        print(f"Gemini: {answer(history, result, prompt, session_facts_list)}")



main()
