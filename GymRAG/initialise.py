from google import genai
import os
from dotenv import load_dotenv
from google.genai import types
import chromadb
from pathlib import Path

class GymRAG:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)

        self.chroma_client = chromadb.PersistentClient(path="./chroma_data")
        self.history = []
        self.session_facts_list = []
        self.collection = self.chroma_client.get_or_create_collection(
            name="gym_chunks",
            embedding_function=None,
        )
        self.distance_threshold = 0.33


    def index_document(self, file_path):
        document_name = Path(file_path).stem
        existing = self.collection.get(where={"source": document_name})
        if existing["ids"]:#we check if the id list is not empty (we suppose that the documents wont change)/ This only checks whether the document has been indexed before/It does NOT check whether the file has changed since it was indexed./A production system would delete the old chunks and re-index the document.
            print(f"Document '{document_name}' already indexed, skipping.")
            return True, None
        text, error = self._load_document(file_path)
        if error:
            return None, error
        
        self._store_in_chroma(self._prepare_documents_for_indexing(document_name, text))
        
        return True, None
        
    def retrieve(self, query, n_results=3):
        try:
            search_query = self._rewrite_query(query)
            return self._similarity_search(search_query, n_results)
        except Exception:
            return []
    

    def chat(self, user_message):
        try:
            new_fact = self._extract_session_facts(user_message)
        
            if new_fact:
                self.session_facts_list.append(new_fact)
            
            retrieved_chunks = self.retrieve(user_message)
            
            return self._generate_answer(retrieved_chunks, user_message)
        
        except Exception as e:
            print(f"Error: {e}")#we can see the real error and the user will se friendly message
            return "Sorry, something went wrong while processing your request. Please try again."
    


    def _build_prompt(self,retrieved_chunks, user_message):
        
        if not retrieved_chunks:
            context = ""
        else:  
            context = "\n\n".join(retrieved_chunks)

        session_facts = "\n".join(self.session_facts_list)

        prompt = f"""
        <session_facts>
        {session_facts}
        </session_facts>

        <context>
        {context}
        </context>

        <question>
        {user_message}
        </question>

        """
        return prompt

    def _generate_model_response(self, retrieved_chunks, user_message):
        
        prompt = self._build_prompt(retrieved_chunks, user_message)
        
        temp_history = self.history.copy()
        temp_history.append(
            types.Content(
                role="user",
                parts=[
                    types.Part(text=prompt)
                ]
            )
        )

        model_response = self.client.models.generate_content(
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
        
        return model_response
    
    def _update_history(self, model_response, user_message):
        
        self.history.append(
            types.Content(
                role="user",
                parts=[
                    types.Part(text=user_message)
                ]
            )
        )

        self.history.append(
            types.Content(
                role="model",
                parts=[
                    types.Part(text=model_response.text)
                ]
            )
        )

    def _generate_answer(self, retrieved_chunks, user_message):
                
        model_response = self._generate_model_response(retrieved_chunks, user_message)

        self._update_history(model_response, user_message)

        return model_response.text
    
    def _similarity_search(self, question, n_results):
        
        search_results = self._query_collection(question, n_results)

        retrieved_chunks = self._filter_search_results(search_results)
        
        return retrieved_chunks
    
    def _filter_search_results(self, search_results):
        filtered_results = []
        for doc, distance in zip(search_results["documents"][0], search_results["distances"][0]):
            if distance <= self.distance_threshold:
                filtered_results.append(doc)
        return filtered_results
    
    def _query_collection(self, question, n_results):
        query_embedding = self.client.models.embed_content(
            model="gemini-embedding-001",
            contents=question,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
        )

        search_results = self.collection.query(
            query_embeddings=[query_embedding.embeddings[0].values],
            n_results=n_results,
            include=["documents", "distances"]
        )
        return search_results

    def _extract_session_facts(self, prompt):
        model_response = self.client.models.generate_content(
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

        return model_response.text

    
    def _rewrite_query(self, question):
        if not self.history:
            return question
        else:
            temp_history = self.history.copy()
            temp_history.append(
            types.Content(
                role="user",
                parts=[
                    types.Part(text=f"The current user question is:\n{question}")
                ]
            )
        )
            rewrite_response = self.client.models.generate_content(
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
            return rewrite_response.text.strip()
    
    def _load_document(self, file_path):
    
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                if not text:
                    return (None, "Error: File is empty.")
                return (text, None)
        except FileNotFoundError:
            return (None, f"Error: The file at '{file_path}' does not exist.")

    def _prepare_documents_for_indexing(self,document_name, text):

        chunks = self._chunk_document(text)

        embedding_response = self._create_embeddings(chunks)
        
        embeddings = self._get_embeddings_vectors(embedding_response)
        ids = self._generate_chunk_ids(document_name, len(chunks))

        chroma_records = self._build_chroma_records(document_name, chunks, embeddings, ids)

        return chroma_records

    def _build_chroma_records(self, document_name, chunks, embeddings, ids):
        chroma_records = []
        for i in range(len(chunks)):
            record = {"id": ids[i], "text": chunks[i], "embedding": embeddings[i], "metadata": {"source": document_name, "chunk_index": i, "chunk_type": "paragraph"}}
            chroma_records.append(record)
        return chroma_records

    def _chunk_document(self, text):
        chunks = []
        for chunk in text.split("\n\n"):
            chunk = chunk.strip()
            if chunk:
                chunks.append(chunk)
        return chunks
    
    def _create_embeddings(self, chunks):
        return self.client.models.embed_content(
                model="gemini-embedding-001",
                contents=chunks,
                config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
        )

    def _store_in_chroma(self, records):

        self.collection.upsert(
            ids=[chunk["id"] for chunk in records],
            embeddings=[chunk["embedding"] for chunk in records],
            documents=[chunk["text"] for chunk in records],
            metadatas=[chunk["metadata"] for chunk in records],
        )

    def _get_embeddings_vectors(self, result):
        embeddings = []
        for embedding in result.embeddings:
            embeddings.append(embedding.values)
        return embeddings

    def _generate_chunk_ids(self, document_name, num):
        ids = []
        for i in range(num):
            ids.append(f"{document_name}_{i}")
        return ids
    

def main():
    rag = GymRAG()

    _, error = rag.index_document("gym_pricing.txt")
    if error:
        print(error)
        return
    _, error = rag.index_document("gym_faq.txt")
    if error:
        print(error)
        return
    
    while (True):
        question = input("You: ")
        if (question == "quit"):
            break
        print(f"Assistant: {rag.chat(question)}")


main()