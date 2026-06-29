from google import genai
import os
from dotenv import load_dotenv
from google.genai import types
import json

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

mock_database = {
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

get_unpaid_members_declaration = {
        "name": "get_unpaid_members",
        "description": "Returns members who have unpaid subscriptions for a specific month and year. Use this when the user asks about unpaid payments, debts, or missing payments.",
            "parameters": {
        "type": "object",
        "properties": {
            "month": {
                "type": "string",
                "description": "The month to check, for example June."
            },
            "year": {
                "type": "integer",
                "description": "The year to check, for example 2026."
            }
        },
        "required": []
            }
}

update_member_plan_declaration = {
    "name": "update_member_plan",
    "description": "Updates the subscription plan of a gym member.",
    "parameters": {
        "type": "object",
        "properties": {
            "member_id": {
                "type": "string",
                "description": "The unique ID of the member whose plan should be changed.",
            },
            "new_plan": {
                "type": "string",
                "description": "The new subscription plan. Possible values: monthly, quarterly, annual.",
            },
        },
        "required": ["member_id", "new_plan"],
    },
}


def get_member_stats(member_id: str) -> dict:
    """This function returns all the information a gym owner needs about a specific member using their member ID"""
    if (member_id not in mock_database):
        return {"error": "member not found"}
    return mock_database[member_id]

def get_unpaid_members(month: str, year: int) -> list:
    """Returns members who have not paid their subscription for a specific month."""
    return [
        {
            "name": "Sara El Idrissi",
            "member_id": "M002",
            "days_overdue": 12,
        },
        {
            "name": "Youssef Amrani",
            "member_id": "M003",
            "days_overdue": 5,
        },
    ]


def update_member_plan(member_id: str, new_plan: str) -> dict:
    """This function updates the subscription plan of a specific gym member using their member ID."""

    if member_id not in mock_database:
        return {"error": "member not found"}
    
    allowed_plans = ["monthly", "quarterly", "annual"]

    if new_plan not in allowed_plans:
        return {"error": "invalid plan"}

    old_plan = mock_database[member_id]["plan"]

    mock_database[member_id]["plan"] = new_plan

    return {
        "success": True,
        "member_id": member_id,
        "old_plan": old_plan,
        "new_plan": new_plan
    }    





def ask_with_tools(question, contents=None):
    tools = [
        types.Tool(function_declarations=[get_member_stats_declaration, get_unpaid_members_declaration, update_member_plan_declaration])
    ]
    config = types.GenerateContentConfig(
        tools=tools,
        system_instruction="""
You are GymAssistant, an AI assistant for a gym management platform.

Answer questions using only the data provided to you in each conversation. Do not use any knowledge from your training about specific gyms, prices, or schedules.

You can answer questions about:
- clients
- subscriptions
- payments
- attendance
- gym statistics
- business insights

Rules:
- Never invent information.
- Only use provided data.
- If information is missing, say you don't have access to it.
- Ask for clarification if the request is unclear.
- Give concise, practical answers, respond in plain language. Use short bullet points for lists of items. Keep answers under 150 words unless the question requires more detail.
- If asked about topics unrelated to gym management, politely explain that you're specialized for gym operations and redirect the conversation.

Your goal:
Help the gym owner understand their gym operations and make better decisions.
        """,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True
        )
    )

    if contents is None:
        contents = []

    contents.append(
        types.Content(
            role="user",
            parts=[types.Part(text=question)]
        )
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=config,
    )

    part = response.candidates[0].content.parts[0]

    if not part.function_call:
        return response.text
    function_call = part.function_call

    if function_call.name == "get_member_stats":
        result = get_member_stats(**function_call.args)

    elif function_call.name == "get_unpaid_members":
        result = get_unpaid_members(**function_call.args)

    elif function_call.name == "update_member_plan":
        result = update_member_plan(**function_call.args)
    else:
        return({"error": "unknown function"})

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


    return final_response.text

def main():
    print(ask_with_tools("Which members haven't paid this month"))

main()