# merged_voice_assistant.py

import os
import time
import threading
import requests
import numpy as np
import pyaudio
import webrtcvad
import torch
import torchaudio
import noisereduce as nr
import pyttsx3 as pt
from transformers import (
    AutoModelForCTC,
    Wav2Vec2Processor,
    WhisperProcessor,
    WhisperTokenizer,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
)

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector_search import retriever

# ========== Audio + Speech Model Config ==========
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 320  # 20ms at 16kHz
VAD_MODE = 3
SILENCE_THRESHOLD = 0.5
MIN_BUFFER_DURATION = 1
MAX_BUFFER_DURATION = 5

# ========== Text-to-Speech Engine ==========
engine = pt.init()

# ========== LangChain + LLM Config ==========
llm = OllamaLLM(model="llama3.2")

prompt_template = """
You are a helpful assistant developed by Mr. Anshuman Pathak to answer questions related to PSIT College of Engineering. 
Your responses should be concise, factual, and respectful. You are running completely on a local machine.
Always generate short and precise answers.

Use the following relevant information retrieved from the database to form your answer only.
Avoid repeating any information already mentioned in previous answers or redundant details.

Below the user's question:
{Instruction}

Relevant information:
{Response}

Now generate your answer without any redundancy:
"""

prompt = ChatPromptTemplate.from_template(prompt_template)
llm_chain = prompt | llm

# ========== Speech Processor ==========
class SpeechProcessor:
    def __init__(self, language="en"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        self.processors = {}
        self.language = language
        self.vad = webrtcvad.Vad(VAD_MODE)
        self.audio_buffer = []
        self.is_speaking = False
        self.silence_duration = 0
        self.running = True
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.load_models()

    def load_models(self):
        print("🔁 Loading models...")
        # Hindi
        self.models["hi"] = AutoModelForCTC.from_pretrained("ai4bharat/indicwav2vec-hindi").to(self.device)
        self.processors["hi"] = Wav2Vec2Processor.from_pretrained("ai4bharat/indicwav2vec-hindi")

        # English
        english_model_id = "Tejveer12/Indian-Accent-English-Whisper-Finetuned"
        tokenizer = WhisperTokenizer.from_pretrained(english_model_id)
        feature_extractor = WhisperFeatureExtractor.from_pretrained(english_model_id)
        self.processors["en"] = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        self.models["en"] = WhisperForConditionalGeneration.from_pretrained(english_model_id).to(self.device)
        print("✅ Models loaded.")

    def reduce_noise(self, waveform):
        return nr.reduce_noise(y=waveform, sr=RATE, prop_decrease=0.95)

    def normalize_audio(self, waveform):
        return waveform / np.max(np.abs(waveform)) if np.max(np.abs(waveform)) > 0 else waveform

    def start_stream(self):
        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        print("🎙️ Listening... (Press Ctrl+C to stop)")
        while self.running:
            audio_data = self.stream.read(CHUNK, exception_on_overflow=False)
            is_speech = self.vad.is_speech(audio_data, RATE)
            audio_int16 = np.frombuffer(audio_data, dtype=np.int16)

            if is_speech:
                self.is_speaking = True
                self.silence_duration = 0
                self.audio_buffer.append(audio_int16)
                if len(self.audio_buffer) * CHUNK / RATE >= MAX_BUFFER_DURATION:
                    self.process_audio_chunk()
            else:
                if self.is_speaking:
                    self.silence_duration += CHUNK / RATE
                    if self.silence_duration >= SILENCE_THRESHOLD:
                        if len(self.audio_buffer) * CHUNK / RATE >= MIN_BUFFER_DURATION:
                            self.process_audio_chunk()
                        self.audio_buffer = []
                        self.is_speaking = False
                else:
                    self.audio_buffer.append(audio_int16)

    def process_audio_chunk(self):
        audio_chunk = np.concatenate(self.audio_buffer).tobytes()
        transcription = self.transcribe(audio_chunk)
        if transcription:
            print(f"\n📝 Transcription: {transcription}")
            self.send_to_llm(transcription)
        self.audio_buffer = []

    def transcribe(self, audio_data):
        waveform = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        waveform = self.normalize_audio(waveform)
        waveform = self.reduce_noise(waveform)
        waveform = torch.tensor(waveform).float().to(self.device)

        if RATE != 16000:
            waveform = torchaudio.functional.resample(waveform, RATE, 16000)

        if self.language == "hi":
            inputs = self.processors["hi"](waveform.cpu().numpy(), return_tensors="pt", sampling_rate=16000).input_values.to(self.device)
            with torch.no_grad():
                logits = self.models["hi"](inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            return self.processors["hi"].batch_decode(predicted_ids)[0]
        else:
            input_features = self.processors["en"](
                waveform.cpu().numpy(),
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features.to(self.device)
            forced_decoder_ids = self.processors["en"].get_decoder_prompt_ids(language="en", task="transcribe")
            with torch.no_grad():
                outputs = self.models["en"].generate(
                    input_features,
                    forced_decoder_ids=forced_decoder_ids,
                    num_beams=1,
                    max_length=30
                )
            return self.processors["en"].batch_decode(outputs.sequences, skip_special_tokens=True)[0]

    def send_to_llm(self, transcription):
        try:
            retrieved_docs = retriever.invoke(transcription)
            seen = set()
            filtered_docs = [doc.page_content.strip() for doc in retrieved_docs if not (doc.page_content.strip() in seen or seen.add(doc.page_content.strip()))]
            context = "\n".join(filtered_docs)
            result = llm_chain.invoke({"Instruction": transcription, "Response": context})

            print("💬 Assistant:", result)
            engine.say(result)
            engine.runAndWait()

        except Exception as e:
            print("❌ LLM processing error:", e)

    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()


# ========== Main Runner ==========
def main():
    language = input("🌐 Choose language (hi for Hindi, en for English): ").strip().lower()
    if language not in ["hi", "en"]:
        print("❌ Invalid input. Please choose 'hi' or 'en'.")
        return

    processor = SpeechProcessor(language=language)
    try:
        processor.start_stream()
    except KeyboardInterrupt:
        print("\n🛑 Stopping assistant...")
        processor.stop()

if __name__ == "__main__":
    print("✅ Voice Assistant starting...")
    main()
