# preprocess books
# expect text file to be txt file
# expect text file to be in the 'raw' directory as this script
# expect text file name to be in the format 'book_title.txt'

import os
import re
import time
from collections import deque
import tiktoken
import openai

with open('openai_key.txt', 'r') as f:
    key = f.read().strip()
client = openai.OpenAI(api_key=key)
encoding = tiktoken.encoding_for_model("gpt-4o")

def gpt_preprocess(text_chunk, model="gpt-4o"):
    PROMPT_TEMPLATE = """
Preprocess Below Text according to following instructions:

1. Resolve Coreferences
2. Normalize text (예: ‘ﬁ’를 ‘fi’로, ‘e-mail’을 ‘email’로).
3. Remove Repetitive Sentences
4. Remove Header, Footer, any other Metadata that includes but not limited to Copyright info, publication info, etc.

! Do NOT Alter Sentence Structure apart from above instructions!
! DO preserve original sentence structure apart from above instructions!

[Begin text]
{text}
[End text]
"""
    prompt = PROMPT_TEMPLATE.format(text=text_chunk)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("Error:", e)
        return ""

for filename in os.listdir('raw'):
    if filename.endswith('.txt') and not filename in os.listdir('preprocessed_books'):
        print(f"Processing {filename}...")
        with open(os.path.join('raw', filename), 'r', encoding='utf-8') as f:
            text = f.read()
        # remove all new lines
        text = re.sub(r'\n+', ' ', text)
        # remove all leading and trailing spaces
        text = text.strip()
        # 종결 기호(., ?, !, … 등) 기준 분할
        split_text = deque(re.split(r'(?<=[.?!…])\s+', text))
        chunks = []
        chunk = []
        toks = 0
        while split_text:
            s = split_text.popleft()
            toks += len(encoding.encode(s))
            chunk.append(s)
            if toks >= 5000:
                chunks.append(' '.join(chunk))
                chunk = chunk[-3:]
                toks = 600
        results = []
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            processed = gpt_preprocess(chunk)
            results.append(processed)
            time.sleep(1.0)
        text = ' '.join(results)
        # save the preprocessed text to a new file in the 'preprocessed' directory
        # need to format the filename to be in the format 'book_title_en-book_title_kr-author.txt'
        # but only demo, so just hand-write the filename
        with open(os.path.join('preprocessed_books', filename), 'w', encoding='utf-8') as f:
            f.write(text)