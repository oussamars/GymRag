from google import genai
import os
from dotenv import load_dotenv
from google.genai import types


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


def ask_gemini(content, generate_config):

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=generate_config,
        contents=content
    )

    return response.text



def main():
    content = "Explain how the internet works in two lines."
    config=types.GenerateContentConfig(
        system_instruction="You're an expert in IT",
        temperature=1.0,
        max_output_tokens=1000
    )
    print(ask_gemini(content, config))




main()
