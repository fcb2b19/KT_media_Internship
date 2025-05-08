import requests
import json
import re

class OllamaClient:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url

    def query(self, model="gemma3", prompt="", stream=False):
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": stream
            }
        )
        return response.json()['response']

    def preprocess_query(self, query, history, book_list):
        # Preprocess the query using the model
        with open("query_preprocess_prompt.txt", "r") as f:
            PROMPT_TEMPLATE = f.read()
        response = self.query(prompt=PROMPT_TEMPLATE.format(query=query, history=history, book_list=book_list))
        preprocessed_query = json.loads(re.search(r'\{.*\}', response, re.DOTALL).group())
        return preprocessed_query
    
    def rag_query(self, query, book_titles):
        # Generate a RAG query for the books title
        with open("rag_prompt.txt", "r") as f:
            PROMPT_TEMPLATE = f.read()
        response = self.query(prompt=PROMPT_TEMPLATE.format(query=query, book_titles=book_titles))
        rag_query = json.loads(re.search(r'\{.*\}', response, re.DOTALL).group())
        return rag_query