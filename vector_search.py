import os

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama.embeddings import OllamaEmbeddings

df = pd.read_csv("vaman_dataset1.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
db_location = "./chroma_langchain_db"
rebuild_db = os.getenv("VAMAN_REBUILD_DB", "0") == "1"

documents = []
ids = []
for index, row in df.iterrows():
    instruction = str(row["Instruction"]).strip()
    response = str(row["Response"]).strip()
    content = f"{instruction}. {response}"
    documents.append(Document(page_content=content, metadata={}))
    ids.append(str(index))

vector = Chroma(
    collection_name="PSIT",
    persist_directory=db_location,
    embedding_function=embeddings,
)

if rebuild_db or not os.path.exists(db_location):
    vector.add_documents(documents=documents, ids=ids)
    print(f"Added {len(documents)} documents to vector DB.")

retriever = vector.as_retriever(search_kwargs={"k": 50})
