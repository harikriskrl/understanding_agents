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


def retrieve_context(query):
    return [
        f"Relevant context chunk 1 for: {query}",
        f"Relevant context chunk 2 for: {query}"
    ]


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

    
def llm_generate(query, context_list=None):
    prompt = f"Query: {query}\n\n"

    if context_list:
        prompt += "Context:\n"
        for item in context_list:
            prompt += f"- {item}\n"

    prompt += "\nAnswer:"

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use the provided context if available. If no context is provided, answer from model knowledge. Structure the answer clearly."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    return call_llm(messages)


def agent(user_query):
    print(f"\nUser Query: {user_query}")
    raw_plan = generate_plan(user_query).strip()
    raw_plan = raw_plan.replace("```json", "").replace("```", "").strip()
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
            context_list = []
            answer = llm_generate(sub_query, context_list)
            results.append(answer)

        elif q_type == "dynamic":
            context_list = retrieve_context(sub_query)
            answer = llm_generate(sub_query, context_list)
            results.append(answer)

        elif q_type == "synthesis":
            context_list = retrieve_context(sub_query)
            answer = llm_generate(sub_query, context_list)
            results.append(answer)

        else:
            # fallback: treat unknown type as static
            context_list = []
            answer = llm_generate(sub_query, context_list)
            results.append(answer)


    final_answer = "\n\n".join(results)

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