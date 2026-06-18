from google import genai
import os
from dotenv import load_dotenv
from google.genai import types
# from PIL import Image

def main():
    response = ask_gemini(
        system_instruction = "You are a helpful gym assistant that gives fitness advice.",
        # This fixed persona is important for RAG/chatbots because it ensures the AI stays consistent and focused on fitness advice regardless of user input.
        content = "give me 5 push exercises i can do for this chest, shoulders and triceps workout"
        )
    print(response)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


def ask_gemini(system_instruction, content):


    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction),
        contents=content
    )

    return (response.text)



main()
