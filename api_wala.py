from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from vector_search import retriever

app = FastAPI()
model = OllamaLLM(model="llama3.2")
template = """
You are a helpful assistant developed by Mr. Anshuman Pathak to answer questions related to PSIT College of Engineering.
Your responses should be concise, factual, and respectful. You are running completely on a local machine.
Always generate short and precise answers.

Use the following relevant information retrieved from the database to form your answer only.
Avoid repeating any information already mentioned in previous answers or redundant details.

Below  the user's question:
{Instruction}

Relevant information:
{Response}

Now generate your answer without any redundancy:
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


class Query(BaseModel):
    question: str


@app.post("/ask")
def ask_question(query: Query):
    question = query.question
    retrieved_docs = retriever.invoke(question)

    seen = set()
    filtered_docs = []
    for doc in retrieved_docs:
        content = doc.page_content.strip()
        if content not in seen:
            seen.add(content)
            filtered_docs.append(content)

    result = chain.invoke({"Instruction": question, "Response": "\n".join(filtered_docs)})
    return {"response_text": result}
