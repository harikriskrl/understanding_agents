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


# -----------------------------
# Week-2 reasoning functions
# -----------------------------

def split_query(query):
    # simple version for now
    parts = query.split(" and ")
    return [p.strip() for p in parts if p.strip()]


def classify_part(part):
    part_lower = part.lower()

    dynamic_keywords = ["latest", "recent", "current", "today", "price", "news"]
    hybrid_keywords = ["common", "typical", "use cases", "uses"]

    if any(word in part_lower for word in dynamic_keywords):
        return "dynamic"
    elif any(word in part_lower for word in hybrid_keywords):
        return "hybrid"
    else:
        return "static"


def decide_tool(part, classification):
    if classification == "dynamic":
        return True
    elif classification == "hybrid":
        return False   # Option 1: simple for now
    else:
        return False

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

def generate_final_answer(query, thought, observation=None):
    if observation is not None:
        user_content = f"Query: {query}\nReasoning: {thought}\nObservation: {observation}"
    else:
        user_content = f"Query: {query}\nReasoning: {thought}"

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use the reasoning and observation to produce a clear final answer."
        },
        {
            "role": "user",
            "content": user_content
        }
    ]
    return call_llm(messages)


def agent(user_query):
    print(f"\nUser Query: {user_query}")

    parts = split_query(user_query)
    part_outputs = []
    tools_used = []

    for idx, part in enumerate(parts, start=1):
        classification = classify_part(part)
        use_tool = decide_tool(part, classification)

        print(f"\nPart {idx}: {part}")
        print(f"Classification: {classification}")
        print(f"Tool needed: {use_tool}")

        if use_tool:
            tool_name = "search_web"
            tool_result = search_web(part)
            print(f"Tool Output: {tool_result}")
            thought = generate_thought(part, tool_result)
            answer = generate_final_answer(part, thought, tool_result)
            tools_used.append(tool_name)
        else:
            thought = generate_thought(part)
            answer = generate_final_answer(part, thought)
            tools_used.append(None)

        part_outputs.append(f"Part {idx}:\n{answer}")

    final_answer = "\n\n".join(part_outputs)

    return {
        "query": user_query,
        "parts": parts,
        "tools_used": tools_used,
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