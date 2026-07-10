import streamlit as st
from pathlib import Path

from GymRag.rag_system import GymRAG

st.title("GymAssistant")

with st.sidebar:
    st.title("GymAssistant")
    st.markdown(
        """
        **About this assistant:**
        
        GymAssistant answers questions about:
        - Gym membership plans and pricing
        - Gym FAQs and policies
        - Facility information
        
        The assistant remembers facts you share during the current conversation, such as your name or membership type. Clearing the conversation resets this memory.
        """
    )
    
    if st.button("Clear Conversation", key="clear_button"):
        st.session_state.messages = []

        if "rag" in st.session_state:
            del st.session_state["rag"]

        st.rerun()

base_dir = Path(__file__).resolve().parent / "GymRag"

# Create the conversation history the first time the app runs
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag" not in st.session_state:
    st.session_state.rag = GymRAG()
    st.session_state.rag.index_document(str(base_dir / "gym_faq.txt"))
    st.session_state.rag.index_document(str(base_dir / "gym_pricing.txt"))

# Display all previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Wait for the user to send a new message
prompt = st.chat_input("Ask a question")

if prompt:
    # Save the user's message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    # Display it immediately
    with st.chat_message("user"):
        st.write(prompt)

    with st.spinner("Thinking..."):
        try:
            response = st.session_state.rag.chat(prompt)
        except Exception:
            st.error("Something went wrong. Please try again.")
            response = None

    if response is not None:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
            }
        )

        with st.chat_message("assistant"):
            st.write(response)
