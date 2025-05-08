import chromadb

class db:
    def __init__(self, db_path="./chroma_db"):
        # set up environment for chroma
        self.client = chromadb.PersistentClient(path=db_path)
        self.titles = self.client.get_collection("titles")

    def search_titles(self, embeddings, n_results=3):
        # search for the most relevant titles in the database
        results = self.titles.query(query_embeddings=[embeddings], n_results=n_results)
        return results['metadatas'][0]
    
    def search_book(self, title, embeddings, n_results=3):
        # search for the book in the database
        collection = self.client.get_collection(title)
        results = collection.query(query_embeddings=[embeddings], n_results=n_results)
        return results['documents'][0]
    
    def get_book_list(self):
        # get the list of books in the database
        return self.client.list_collections()
