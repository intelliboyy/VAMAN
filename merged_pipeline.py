import os

import requests
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from playsound import playsound

from vector_search import retriever

load_dotenv()

API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")


def text_to_speech(text):
    if not API_KEY or not VOICE_ID:
        print("Skipping TTS: set ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID in .env")
        return

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    output_filename = "output.mp3"
    headers = {
        "xi-api-key": API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.75,
            "similarity_boost": 0.75,
        },
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        with open(output_filename, "wb") as f:
            f.write(response.content)
        print(f"Audio saved to {output_filename}")
        playsound(output_filename)
    else:
        print("TTS error:", response.status_code, response.text)


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
print("Assistant is ready. Waiting for your question...\n")

while True:
    question = input("Ask the question (or 'q' to quit): ")
    if question.lower() == "q":
        break

    retrieved_docs = retriever.invoke(question)
    seen = set()
    filtered_docs = []
    for doc in retrieved_docs:
        content = doc.page_content.strip()
        if content not in seen:
            seen.add(content)
            filtered_docs.append(content)

    result = chain.invoke({"Instruction": question, "Response": "\n".join(filtered_docs)})
    print("\n--- Answer ---")
    print(result)
    text_to_speech(result)
