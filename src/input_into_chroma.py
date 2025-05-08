# feed preprocessed data into chroma
# expect text file to be in the format 'book_title_en-book_title_kr-author.txt'

import os
import re
from collections import deque
import time
import tiktoken
import openai
import chromadb
from chromadb.config import Settings
from langdetect import detect

# set up environment for chroma
with open('openai_key.txt', 'r') as f:
    key = f.read().strip()
client = openai.OpenAI(api_key=key)
encoding = tiktoken.encoding_for_model("gpt-4o-mini")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
# titles collection is used to store book titles and their summaries
titles = chroma_client.get_or_create_collection("titles")
# need the last id to avoid overwriting existing titles
last_id = len(titles.get()['documents'])
list_of_titles = chroma_client.list_collections()

# use openai to get the embedding of the text
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

for filename in os.listdir('preprocessed_books'):
    if filename.endswith('.txt'):
        # expect text file to be in the format 'book_title_en-book_title_kr-author.txt'
        t = filename.split('.')[0]
        print("started book: ", t)
        BOOK_TITLE_EN = '_'.join(t.split('-')[0].split())
        BOOK_TITLE_KO = '_'.join(t.split('-')[1].split())
        AUTHOR = t.split('-')[2]
        if BOOK_TITLE_EN in list_of_titles:
            print(f"{BOOK_TITLE_EN} already exists in the database.")
            continue
        last_id += 1
        BOOK_ID = f"{last_id:03d}"
        with open(os.path.join('preprocessed_books', filename), 'r', encoding='utf-8') as f:
            text = f.read()
        # detect language of the text
        LANGUAGE = detect(text)
        # chunk the text into smaller pieces
        sentences = deque(re.split(r'(?<=[.?!…])\s+', text))
        chunks = []
        chunk = []
        toks = 0
        # split the text into chunks of around 500 tokens
        while sentences:
            sentence = sentences.popleft()
            toks += len(encoding.encode(sentence))
            chunk.append(sentence)
            # if the chunk is too long, add it to the list of chunks
            if toks >= 500:
                chunks.append(' '.join(chunk))
                chunk = [sentence]
                toks = len(encoding.encode(sentence))
        # add the last chunk to the list of chunks
        if chunk:
            chunks.append(' '.join(chunk))
        # get summary of the text: the text is too long for the model to process in one go, so we need to split it into chunks
        summary = ""
        for i, chunk in enumerate(chunks):
            print(f"Summarizing chunk {i+1}/{len(chunks)}")
            summary += client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": f"""You are an expert summarizer. summarize the following text into one sentence. return summary in english.: {chunk}"""}],
                    temperature=0.1,
                ).choices[0].message.content.strip()
            summary += " "
            time.sleep(1)  # to avoid hitting the rate limit
        summary = summary[:25000] # limit the summary to 25000 characters
        summary = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"""You are an expert summarizer. summarize the following text into around 500 token length. return summary in english: {summary}"""}],
                temperature=0.1,
            ).choices[0].message.content.strip()
        titles.add(
            documents=[summary],
            ids=[BOOK_ID],
            embeddings=[get_embedding(BOOK_TITLE_EN+BOOK_TITLE_KO+summary)],
            metadatas=[{
                "title": BOOK_TITLE_EN,
                "title_ko": BOOK_TITLE_KO,
                "author": AUTHOR,
                "language": LANGUAGE}]
        )
        print(f"Added {BOOK_TITLE_EN} to the titles.")
        # add the text to the database
        collection = chroma_client.get_or_create_collection(BOOK_TITLE_EN)
        for i, chunk in enumerate(chunks):
            print(f"Adding chunk {i+1}/{len(chunks)} to the database")
            # get the embedding of the text
            embedding = get_embedding(chunk)
            # add the text to the database
            collection.add(
                documents=[chunk],
                ids=[f"{BOOK_ID}_{i:04d}"],
                embeddings=[embedding],
                metadatas=[{
                    "title": BOOK_TITLE_EN,
                    "title_ko": BOOK_TITLE_KO,
                    "author": AUTHOR,
                    "book_id": BOOK_ID,
                    "chunk_id": f"{BOOK_ID}_{i:04d}",
                    "language": LANGUAGE}]
            )
        print(f"Added {BOOK_TITLE_EN} to the database.")