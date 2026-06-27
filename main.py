from google import genai
import os
from dotenv import load_dotenv
from google.genai import types

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

def send_message(history, text):
    history.append(
        types.Content(
            role="user",
            parts=[
                types.Part(text=text)
            ]
        )
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
        system_instruction="You are GymAssistant, an AI assistant for a gym management platform, for now you dont have data just give examples, Use short bullet points for lists of items. Keep answers under 150 words ",
        temperature=2.0,
        ),
        contents=history
    )

    history.append(
        types.Content(
            role="model",
            parts=[
                types.Part(text=response.text)
            ]
        )
    )
    return(response.text)

def main():
    history = []
    while(True):
        prompt = input("You: ")
        if (prompt == "quit"):
            break
        print(f"Gemini: {send_message(history, prompt)}")

main()
