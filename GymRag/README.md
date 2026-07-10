# GymRAG

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about a gym using its own documents instead of relying on the AI model's memory.

## What this does

GymRAG is a simple AI assistant designed for a gym. It can answer questions about membership plans, pricing, schedules, equipment, classes, and gym policies by searching through the gym's documents before generating an answer. It also remembers useful information that a user shares during the conversation, such as their name or membership type, while avoiding answers that are not supported by the available documents.

---

## How it works

When a user asks a question, the system first searches through the indexed gym documents to find the most relevant pieces of information. Those retrieved sections are then provided to the AI model together with the user's question and any useful facts shared earlier in the conversation. The AI generates its answer using only that information. If the answer cannot be found in the indexed documents or the conversation, the assistant clearly says that it does not have enough information instead of making something up.

---

## Installation

```bash
git clone <your-repository-url>
cd <your-project-folder>

python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
# venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env

# Add your Gemini API key to .env
# GEMINI_API_KEY=your_api_key_here

python rag_system.py
```

---

## Example conversations

### 1. Answering from the gym documents

```text
You: What equipment does the gym have?

Assistant: The gym includes treadmills, rowing machines, stationary bikes, squat racks, cable machines, free weights, resistance machines, benches, and stretching areas.
```

---

### 2. Using session facts

```text
You: My name is Oussama and I'm a student.

Assistant: Nice to meet you, Oussama. I've noted that you're a student.

You: What membership should I get?

Assistant: Since you told me you're a student, you may qualify for the Student Membership, which costs 255 MAD per month with a valid student ID and is available to members under 26 years old.
```

---

### 3. Refusing to invent information

```text
You: Which personal trainer specializes in bodybuilding?

Assistant: I don't have enough information to answer that because the indexed gym documents do not contain information about individual trainers or their specialties.
```

---

## Technical details

* **Embeddings:** Gemini `embedding-001`
* **Vector database:** ChromaDB (local persistent storage)
* **Language model:** Gemini 2.5 Flash

---

## Limitations

This assistant only knows what exists inside the indexed gym documents and the facts the user shares during the current conversation. If a topic is not covered by those documents, it will respond that it does not have enough information rather than inventing an answer. Retrieval quality also depends on the similarity threshold: lowering it may retrieve more relevant information but can also introduce unrelated chunks, while increasing it may miss useful results.
