import ollama_query
import chroma_query
import gpt_query
from langdetect import detect

# things to do in this class:
# keep track of the conversation history
# get query from the user
# toss to ollama local running on http://localhost:11434/api/
# get result from the titles collection
# generate rag query for each book returned
# get result from the books db
# using all the above, generate a response to the user using gpt-4o
# summarize query and response to save to conversation history
# save the conversation history to a file

class chatbot:
    def __init__(self, openai_key):
        self.gpt_client = gpt_query.gpt_api(openai_key)
        self.history = []# conversation history: list of tuples (query, response)
        self.actual_history = [] # actual history: list of tuples (query, response) for the model
        self.ollama_client = ollama_query.OllamaClient()
        self.chroma_client = chroma_query.db()
        self.book_list = self.chroma_client.get_book_list()
        self.main_book = None
        self.mode = "D" # default mode is discuss

    def get_user_input(self):
        return input("You: ")

    def process_query(self, query):
        # Preprocess query using Ollama client
        language = detect(query)
        preprocessed = self.ollama_client.preprocess_query(query, self.history, self.book_list)
        print(f"Preprocessed: {preprocessed}")
        preprocessed_query = preprocessed['translated_query']
        multi_book = preprocessed['multi_book']
        if multi_book in ["true", "True", "TRUE"]:
            multi_book = 2
        else:
            multi_book = 1
        search_query = self.gpt_client.get_embedding(preprocessed['search_query'])

        # Search titles from Chroma client
        infos = self.chroma_client.search_titles(search_query, multi_book)
        book_titles = [infos[i]['title'] for i in range(len(infos))]
        if self.main_book and (not book_titles or self.main_book not in book_titles) :
            book_titles.append(self.main_book)
        print(f"Book Titles: {book_titles}")

        # Generate RAG query for each book returned
        rag_queries = self.ollama_client.rag_query(preprocessed_query, book_titles=book_titles)
        print(f"RAG Queries: {rag_queries}")

        # Get relevant chunks from Chroma client
        relevant_chunks = {
            book.replace(" ", "_"): self.chroma_client.search_book(
                book.replace(" ", "_"),
                self.gpt_client.get_embedding(query),
                3 - multi_book
            )
            for book, query in rag_queries.items()
        }
        # Get mode from toggle UI (assuming a method exists to fetch this)
        mode = self.mode # Placeholder for mode selection, e.g., "D" for discuss, "N" for normal

        # Input all into GPT client to get response
        response = self.gpt_client.get_response(preprocessed_query, self.history, relevant_chunks, language ,mode)

        return response

    def set_mode(self, mode):
        if mode in ["D", "N"]:
            self.mode = mode
    def set_main_book(self, book):
        self.main_book = book
    
    def update_history(self, query, response):
        # Summarize query and response
        self.actual_history.append((query, response))
        q = self.gpt_client.get_summary(query)
        r = self.gpt_client.get_summary(response)
        self.history.append((q, r))

        # Ensure history length does not exceed 10
        if len(self.history) > 10:
            self.history.pop(0)
            self.actual_history.pop(0)

    def run(self):
        while True:
            query = self.get_user_input()
            if query.lower() in ["exit", "quit"]:
                print("Exiting chatbot. Goodbye!")
                break

            response = self.process_query(query)
            print(f"Chatbot: {response}")

            self.update_history(query, response)

if __name__ == "__main__":
    # Example usage
    with open("openai_key.txt", "r") as f:
        openai_key = f.read().strip()
    
    bot = chatbot(openai_key)
    bot.run()