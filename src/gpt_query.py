import openai

class gpt_api:
    def __init__(self, openai_key):
        self.openai_client = openai.OpenAI(api_key=openai_key)

    def get_embedding(self, text):
        response = self.openai_client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    
    def get_response(self, query, history, retreived_chunks, language,mode="D"):
        # get the response from the gpt-4o model
        if mode == "D":
            with open("main_prompt_discuss.txt", "r", encoding="utf-8") as f:
                SYSTEM_PROMPT = f.read()
        elif mode == "N":
            with open("main_prompt_normal.txt", "r", encoding='utf-8') as f:
                SYSTEM_PROMPT = f.read()
        else:
            return "Invalid mode. Use 'D' for discuss or 'N' for normal."
        # format the prompt with the query and history
        with open("main_prompt.txt", "r") as f:
            PROMPT_TEMPLATE = f.read()
        book_titles = retreived_chunks.keys()
        formatted_prompt = PROMPT_TEMPLATE.format(query=query, book_titles=book_titles, history=history, retreived_chunks=retreived_chunks, language=language, mode=mode)
        # get the response from the model
        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role":"system", "content":SYSTEM_PROMPT},{"role": "user", "content": formatted_prompt}],
            temperature=0.7,
            max_tokens=2000,
            n=1,
            stop=None
        )
        return response.choices[0].message.content.strip()
    
    def get_summary(self, text):
        # get the summary of the text using the gpt-4o model
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"""You are an expert summarizer. summarize the following text into one sentence. preserve the language.: {text}"""}],
            temperature=0.1,
            max_tokens=200,
            n=1,
            stop=None
        )
        return response.choices[0].message.content.strip()