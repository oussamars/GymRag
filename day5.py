from google import genai
import os
from dotenv import load_dotenv
from google.genai import types
import json


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


gym_data = {
    "members": [
        {"name": "Ahmed", "plan": "monthly", "sessions_this_month": 1, 
         "payments_on_time": False, "months_as_member": 2},
        {"name": "Sara", "plan": "annual", "sessions_this_month": 12, 
         "payments_on_time": True, "months_as_member": 14},
        {"name": "Youssef", "plan": "monthly", "sessions_this_month": 0, 
         "payments_on_time": False, "months_as_member": 1},
    ]
}

members = [
    {"name": "Ahmed", "plan": "monthly", "sessions": 0, "paid": False},
    {"name": "Sara", "plan": "annual", "sessions": 14, "paid": True},
    {"name": "Youssef", "plan": "monthly", "sessions": 2, "paid": False},
]

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "plan": {"type": "string"},
        "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
        "recommended_action": {"type": "string"}
    },
    "required": ["name", "plan", "risk_level", "recommended_action"]
}


def ask_gemini(content, generate_config):

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        config=generate_config,
        contents=content
    )
    return response.text



def main():
    content = """
            what is the weather in agadir in celicious
            """
    config=types.GenerateContentConfig(
        # system_instruction="""First write your reasoning inside <thinking> tags, Then write your final answer inside <answer> tags.""",
        temperature=0.7,
        # response_mime_type="application/json",
        # response_schema=schema
    )
    print((ask_gemini(content, config)))


main()
