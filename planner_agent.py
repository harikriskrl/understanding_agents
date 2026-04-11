import requests
import json
import os

API_KEY = os.getenv("OPENAI_API_KEY")
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


def generate_plan(query):
    messages = [
        {
            "role": "system",
            "content": """You are a planning engine.

Break the query into sub-queries.

For each sub-query return:
- text
- type: static / dynamic / synthesis


Return output as JSON list like:
[
  {"text": "...", "type": "static"},
  {"text": "...", "type": "dynamic"}
]

Return ONLY valid JSON.
Do NOT add markdown fences.
Do NOT add explanation text.

Example:
[
  {"text": "Explain 5G", "type": "static"},
  {"text": "Latest developments in 5G", "type": "dynamic"}
]

"""
        },
        {
            "role": "user",
            "content": query
        }
    ]

    return call_llm(messages)

def generate_thought(part, observation=None):
    if observation:
        messages = [
            {
                "role": "system",
                "content": "You are an AI reasoning engine. Think step-by-step. Use the observation to refine your understanding before answering."
            },
            {
                "role": "user",
                "content": f"Query: {part}\nObservation: {observation}"
            }
        ]
    else:
        messages = [
            {
                "role": "system",
                "content": "You are an AI reasoning engine. Think step-by-step about how to answer the query."
            },
            {
                "role": "user",
                "content": part
            }
        ]

    return call_llm(messages)

def generate_final_response(query, results):
    combined = "\n\n".join(results)

    messages = [
        {
            "role": "system",
            "content": "Combine the following parts into a clear final answer. Structure your answer clearly in sections."
        },
        {
            "role": "user",
            "content": f"Query: {query}\n\nParts:\n{combined}"
        }
    ]

    return call_llm(messages)

def generate_final_answer(query, thought, observation=None):
    if observation is not None:
        user_content = f"Query: {query}\nReasoning: {thought}\nObservation: {observation}"
    else:
        user_content = f"Query: {query}\nReasoning: {thought}"

    messages = [
        {
            "role": "system",
            "content": """You are a helpful assistant. Use the reasoning and observation to produce a clear final answer.
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
            6. Follow these rules strictly."""
        },
        {
            "role": "user",
            "content": user_content
        }
    ]
    return call_llm(messages)


def agent(user_query):
    print(f"\nUser Query: {user_query}")
    raw_plan = generate_plan(user_query).strip()
    plan = json.loads(raw_plan)
    print("\nGenerated Plan:")
    print (plan)

    results = []

    for item in plan:
        sub_query = item["text"]
        q_type = item["type"].strip().lower()

        print(f"\nSub-query: {sub_query}")
        print(f"Type: {q_type}")

        if q_type == "static":
            thought = generate_thought(sub_query)
            answer = generate_final_answer(sub_query, thought)
            results.append(answer)

        elif q_type == "dynamic":
            observation = search_web(sub_query)
            thought = generate_thought(sub_query, observation)
            answer = generate_final_answer(sub_query, thought, observation)
            results.append(answer)

        elif q_type == "synthesis":
            # temporary: treat like dynamic
            observation = search_web(sub_query)
            thought = generate_thought(sub_query, observation)
            answer = generate_final_answer(sub_query, thought, observation)
            results.append(answer)

        else:
            # fallback: treat unknown type as static
            thought = generate_thought(sub_query)
            answer = generate_final_answer(sub_query, thought)
            results.append(answer)

    final_answer = generate_final_response(user_query, results)

    return {
        "query": user_query,
        "plan": plan,
        "final_answer": final_answer
    }



if __name__ == "__main__":
    queries = [
        "What is 5G?",
        "What is the latest telecom policy update?",
        "Explain spectrum allocation and recent change in spectrum allocation policy",
        "Explain AI agents and use cases in telecom"
    ]

    for q in queries:
        result = agent(q)
        print("\nFinal Output:")
        print(json.dumps(result, indent=2))
        print("-" * 60)