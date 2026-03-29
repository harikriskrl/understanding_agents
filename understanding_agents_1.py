import requests
import json

API_KEY = "" 
#add api key
MODEL_URL = "https://api.openai.com/v1/chat/completions"


def call_llm(messages):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4.1-mini",
        "messages": messages,
        "temperature": 0
    }

    response = requests.post(MODEL_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def search_web(query):
    # mock tool for now
    return f"Mock search result for: {query}"


def decide_tool(query):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI agent decision-maker. "
                "Reply with only one word: YES or NO. "
                "Say YES if the user query needs external or current information. "
                "Say NO if it can be answered directly."
            )
        },
        {
            "role": "user",
            "content": query
        }
    ]

    decision = call_llm(messages).strip().upper()
    return decision == "YES"


def generate_final_answer(query, observation=None):
    if observation:
        messages = [
            {
                "role": "system",
                "content": """You are a helpful assistant.
                            You MUST follow these rules strictly:

                            1. Start your response with:
                            'Observation usefulness: <useful / not useful>'

                            2. Break the query into parts if needed.

                            3. For static knowledge:
                            - Answer confidently.

                            4. For questions about latest, recent, or current information:
                            - ONLY use the observation.
                            - If observation is not useful:
                                - You MUST say you cannot confirm recent updates.
                                - You MUST NOT provide any specific timelines, dates, or claims about recent changes.
                                - You MUST NOT say 'as of 2024' or any similar phrasing.

                            5. Structure your answer clearly in sections.

                            Follow these rules strictly.""",
            },
            {
                "role": "user",
                "content": f"Query: {query}\nObservation: {observation}"
            }
        ]
    else:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": query
            }
        ]

    return call_llm(messages)


def agent(user_query):
    print(f"\nUser Query: {user_query}")

    tool_needed = decide_tool(user_query)

    if tool_needed:
        print("Agent Decision: Tool needed")
        tool_name = "search_web"
        tool_result = search_web(user_query)
        print(f"Tool Output: {tool_result}")
        observation = tool_result
        final_answer = generate_final_answer(user_query, observation)
    else:
        print("Agent Decision: No tool needed")
        tool_name = None
        final_answer = generate_final_answer(user_query)

    return {
        "query": user_query,
        "tool_used": tool_name,
        "final_answer": final_answer
    }


if __name__ == "__main__":
    queries = [
        "What is 5G?",
        "What is the latest telecom policy update?",
        "Explain spectrum allocation. Is there any recent change in spectrum allocation policy?"
    ]

    for q in queries:
        result = agent(q)
        print("Final Output:")
        print(json.dumps(result, indent=2))
        print("-" * 60)