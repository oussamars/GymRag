from google import genai
import os
from dotenv import load_dotenv
from google.genai import types
import json

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

fake_database = {
    "M001": {
        "name": "Ahmed Benali",
        "plan": "monthly",
        "sessions_this_month": 12,
        "paid_this_month": True,
        "months_as_member": 8,
    },
    "M002": {
        "name": "Sara El Idrissi",
        "plan": "quarterly",
        "sessions_this_month": 7,
        "paid_this_month": False,
        "months_as_member": 14,
    },
    "M003": {
        "name": "Youssef Amrani",
        "plan": "annual",
        "sessions_this_month": 18,
        "paid_this_month": True,
        "months_as_member": 26,
    },
}

get_member_stats_declaration = {
    "name": "get_member_stats",
    "description": "Retrieves membership information and monthly statistics for a specific gym member, like attendance, payment status, subscription plan, months as member and how many sessions he attended.",
    "parameters": {
        "type": "object",
         "properties": {
             "member_id": {
                 "type": "string",
                 "description": "The unique ID of the member whose statistics should be retrieved, for example M001.",
             },
         },
         "required": ["member_id"]
    },
}

def get_member_stats(member_id: str) -> dict:
    """This function returns all the information a gym owner needs about a specific member using their member ID"""
    if (member_id not in fake_database):
        return {"error": "member not found"}
    return fake_database[member_id]

tools = [
    types.Tool(function_declarations=[get_member_stats_declaration])
]
config = types.GenerateContentConfig(
    tools=tools,
    automatic_function_calling=types.AutomaticFunctionCallingConfig(
        disable=True
    )
)

contents = [
    types.Content(
        role="user", parts=[types.Part(text="What are the stats for member M999?")]
    )
]

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=contents,
    config=config,
)

function_call = response.candidates[0].content.parts[0].function_call
if function_call:
    result = get_member_stats(**function_call.args)
else:
    print(response.text)
    exit()

contents.append(response.candidates[0].content)

function_response_part = types.Part.from_function_response(
    name=function_call.name,
    response={"result": result},
)

contents.append(
    types.Content(
        role="user",
        parts=[function_response_part]
    )
)

final_response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=contents,
    config=config,
)


print(final_response.text)