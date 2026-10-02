import warnings
import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector_search import retriever
from langchain import LLMChain
import torch
import torchaudio
import pyaudio
import numpy as np
import re
import noisereduce as nr
from transformers import (
    AutoModelForCTC,
    Wav2Vec2Processor,
    WhisperProcessor,
    WhisperTokenizer,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
)
import webrtcvad
from queue import Queue
import threading
import time
import queue
from concurrent.futures import ThreadPoolExecutor
import speech_recognition as sr
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.spatial.distance import cosine
import wave


# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", message=".*weight.*", module="transformers")
warnings.filterwarnings("ignore", message=".*special.*", module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Hugging Face model cache
os.environ["TRANSFORMERS_CACHE"] = os.path.expanduser("~/.cache/huggingface/hub")

# Audio config for STT
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 320  # 20 ms at 16 kHz (valid for webrtcvad)
VAD_MODE = 3  # Aggressive VAD
SILENCE_THRESHOLD = 0.5  # 500 ms silence
MIN_BUFFER_DURATION = 1  # Minimum 1 second audio to process
MAX_BUFFER_DURATION = 5  # Maximum 5 seconds audio to process

class SpeechProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        self.processors = {}
        self.vad = webrtcvad.Vad(VAD_MODE)
        self.audio_queue = Queue()
        self.audio_buffer = []
        self.is_speaking = False
        self.silence_duration = 0
        self.running = True
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.stream = None
        self.p = None
        self.audio_thread = None
        self.load_models()

    def load_models(self):
        cache_dir = os.environ["TRANSFORMERS_CACHE"]
        hindi_model_id = "ai4bharat/indicwav2vec-hindi"
        self.models["hi"] = AutoModelForCTC.from_pretrained(
            hindi_model_id, cache_dir=cache_dir, local_files_only=True
        ).to(self.device)
        self.processors["hi"] = Wav2Vec2Processor.from_pretrained(
            hindi_model_id, cache_dir=cache_dir, local_files_only=True
        )
        english_model_id = "Tejveer12/Indian-Accent-English-Whisper-Finetuned"
        tokenizer = WhisperTokenizer.from_pretrained(
            english_model_id, cache_dir=cache_dir, local_files_only=True
        )
        feature_extractor = WhisperFeatureExtractor.from_pretrained(
            english_model_id, cache_dir=cache_dir, local_files_only=True
        )
        self.processors["en"] = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        self.models["en"] = WhisperForConditionalGeneration.from_pretrained(
            english_model_id, cache_dir=cache_dir, local_files_only=True
        ).to(self.device)

    def reduce_noise(self, waveform, sample_rate):
        return nr.reduce_noise(y=waveform, sr=sample_rate, prop_decrease=0.95)

    def normalize_audio(self, waveform):
        return waveform / np.max(np.abs(waveform)) if np.max(np.abs(waveform)) > 0 else waveform

    def start_audio_stream(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            stream_callback=self.audio_callback,
        )
        self.stream.start_stream()

    def stop_audio_stream(self):
        self.running = False
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        if self.p is not None:
            self.p.terminate()
            self.p = None
        self.executor.shutdown(wait=True)
        if self.audio_thread:
            self.audio_thread.join(timeout=1.0)
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def audio_callback(self, in_data, frame_count, time_info, status):
        if self.running:
            self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)

    def process_audio_chunk(self, audio_data, language):
        waveform = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        waveform = self.normalize_audio(waveform)
        waveform = self.reduce_noise(waveform, RATE)
        waveform = torch.tensor(waveform).float().to(self.device)

        if RATE != 16000:
            waveform = torchaudio.functional.resample(waveform, RATE, 16000)

        if language == "hi":
            inputs = self.processors["hi"](waveform.cpu().numpy(), return_tensors="pt", sampling_rate=16000).input_values.to(self.device)
            with torch.no_grad():
                logits = self.models["hi"](inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processors["hi"].batch_decode(predicted_ids)[0]
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
                    return_dict_in_generate=True,
                    output_scores=True,
                    num_beams=1,
                    max_length=30
                )
                scores = outputs.scores
                if scores and len(scores) > 0:
                    avg_score = torch.mean(torch.stack([torch.max(s, dim=-1)[0] for s in scores])).item()
                    if avg_score < 0.8:
                        return ""
                transcription = self.processors["en"].batch_decode(outputs.sequences, skip_special_tokens=True)[0]

        return transcription

    def process_audio_loop(self, language):
        while self.running:
            if not self.audio_queue.empty():
                audio_data = self.audio_queue.get()
                audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
                is_speech = self.vad.is_speech(audio_data, RATE)

                if is_speech:
                    self.is_speaking = True
                    self.silence_duration = 0
                    self.audio_buffer.append(audio_int16)
                    buffer_duration = len(self.audio_buffer) * CHUNK / RATE
                    if buffer_duration >= MAX_BUFFER_DURATION:
                        audio_chunk = np.concatenate(self.audio_buffer).tobytes()
                        transcription = self.process_audio_chunk(audio_chunk, language)
                        if transcription:
                            print(transcription)  #### main thing for us to consider
                        self.audio_buffer = []
                        self.is_speaking = False
                else:
                    if self.is_speaking:
                        self.silence_duration += CHUNK / RATE
                        if self.silence_duration >= SILENCE_THRESHOLD:
                            if self.audio_buffer:
                                buffer_duration = len(self.audio_buffer) * CHUNK / RATE
                                if buffer_duration >= MIN_BUFFER_DURATION:
                                    audio_chunk = np.concatenate(self.audio_buffer).tobytes()
                                    transcription = self.process_audio_chunk(audio_chunk, language)
                                    if transcription:
                                        print(transcription)
                            self.audio_buffer = []
                            self.is_speaking = False
                        else:
                            self.audio_buffer.append(audio_int16)
            time.sleep(0.001)

    def run(self, language):
        self.start_audio_stream()
        self.audio_thread = threading.Thread(
            target=self.process_audio_loop,
            args=(language,),
            daemon=True
        )
        self.audio_thread.start()
        while self.running:
            time.sleep(0.1)

# Voice Authentication Functions
encoder = VoiceEncoder()

def record_audio(duration=5, rate=16000, channels=1):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=1024)
    print("Recording for authentication...")
    frames = []
    for _ in range(0, int(rate / 1024 * duration)):
        data = stream.read(1024, exception_on_overflow=False)
        frames.append(data)
    print("Recording finished.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    audio_data = b''.join(frames)
    with wave.open("temp_audio.wav", 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(audio_data)
    wf.close()

def get_embeddings(wav_file_path):
    wav = preprocess_wav(wav_file_path)
    embedding = encoder.embed_utterance(wav)
    os.remove(wav_file_path)
    return embedding

def compare_embeddings(embedding1, embedding2):
    similarity = 1 - cosine(embedding1, embedding2)
    return similarity

def load_user_embedding(user_id):
    return np.load(f"user_{user_id}_embedding.npy")

def generate_user_embedding(user_id):
    print(f"No user embedding found for user_{user_id}. Let's create one.")
    print("Please speak for 5 seconds to record your voice sample.")
    record_audio(duration=5)
    embedding = get_embeddings("temp_audio.wav")
    np.save(f"user_{user_id}_embedding.npy", embedding)
    print(f"User embedding saved as user_{user_id}_embedding.npy")
    return embedding

def recognize_user(user_id):
    try:
        stored_embedding = load_user_embedding(user_id)
    except FileNotFoundError:
        generate_user_embedding(user_id)
        stored_embedding = load_user_embedding(user_id)

    print("Please speak to authenticate.")
    record_audio(duration=5)
    embeddings = get_embeddings("temp_audio.wav")
    similarity = compare_embeddings(stored_embedding, embeddings)
    print(f"Similarity: {similarity}")
    threshold = 0.70
    if similarity >= threshold:
        print("User authenticated successfully!")
        return True
    else:
        print("Authentication failed!")
        return False

# Wake Word Detection
def listen_for_wakeword():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        print("Listening for wake word: 'hello'...")
        recognizer.adjust_for_ambient_noise(source)
        while True:
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                text = recognizer.recognize_google(audio).lower()
                print(f"Heard: {text}")
                if "hello" in text:
                    print("Wake word detected")
                    return True
            except (sr.WaitTimeoutError, sr.UnknownValueError):
                continue
            except sr.RequestError:
                continue
    mic.__exit__(None, None, None)

# Main Function
def main():
    user_id = "1"
    processor = SpeechProcessor()
    while True:
        try:
            if listen_for_wakeword():
                if recognize_user(user_id):
                    language = input("Enter language (hi for Hindi, en for English): ").strip().lower()
                    if language not in ["hi", "en"]:
                        print("Invalid language. Use 'hi' or 'en'.")
                        continue
                    try:
                        processor.run(language)
                    except KeyboardInterrupt:
                        processor.stop_audio_stream()
                print("Returning to wake word detection...")
            else:
                break
        except KeyboardInterrupt:
            break
    processor.stop_audio_stream()

if __name__ == "__main__":
    main()