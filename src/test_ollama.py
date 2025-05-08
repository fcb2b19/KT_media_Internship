import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "gemma3",
        "prompt": "What is the capital of France?",
        "stream": False
    }
)

data = response.json()
print(data['response'])