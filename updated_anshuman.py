# stt_service.py
import pyttsx3 as pt

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
from transformers import (
    AutoModelForCTC,
    Wav2Vec2Processor,
    WhisperProcessor,
    WhisperTokenizer,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
)
engine=pt.init()
# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 320  # 20 ms at 16 kHz
VAD_MODE = 3  # Aggressive VAD
SILENCE_THRESHOLD = 0.5  # 500 ms silence
MIN_BUFFER_DURATION = 1  # Minimum 1 second audio
MAX_BUFFER_DURATION = 5  # Maximum 5 seconds audio

class SpeechProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        self.processors = {}
        self.vad = webrtcvad.Vad(VAD_MODE)
        self.audio_buffer = []
        self.is_speaking = False
        self.silence_duration = 0
        self.running = True
        self.load_models()
        self.p = pyaudio.PyAudio()
        self.stream = None

    def load_models(self):
        # Load Hindi model
        hindi_model_id = "ai4bharat/indicwav2vec-hindi"
        self.models["hi"] = AutoModelForCTC.from_pretrained(hindi_model_id).to(self.device)
        self.processors["hi"] = Wav2Vec2Processor.from_pretrained(hindi_model_id)

        # Load English model
        english_model_id = "Tejveer12/Indian-Accent-English-Whisper-Finetuned"
        tokenizer = WhisperTokenizer.from_pretrained(english_model_id)
        feature_extractor = WhisperFeatureExtractor.from_pretrained(english_model_id)
        self.processors["en"] = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        self.models["en"] = WhisperForConditionalGeneration.from_pretrained(english_model_id).to(self.device)

    def reduce_noise(self, waveform, sample_rate):
        return nr.reduce_noise(y=waveform, sr=sample_rate, prop_decrease=0.95)

    def normalize_audio(self, waveform):
        return waveform / np.max(np.abs(waveform)) if np.max(np.abs(waveform)) > 0 else waveform

    def start_stream(self, language):
        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        print("Listening...")
        while self.running:
            audio_data = self.stream.read(CHUNK, exception_on_overflow=False)
            is_speech = self.vad.is_speech(audio_data, RATE)
            audio_int16 = np.frombuffer(audio_data, dtype=np.int16)

            if is_speech:
                self.is_speaking = True
                self.silence_duration = 0
                self.audio_buffer.append(audio_int16)
                buffer_duration = len(self.audio_buffer) * CHUNK / RATE
                if buffer_duration >= MAX_BUFFER_DURATION:
                    self.process_audio_chunk(language)
            else:
                if self.is_speaking:
                    self.silence_duration += CHUNK / RATE
                    if self.silence_duration >= SILENCE_THRESHOLD:
                        if self.audio_buffer:
                            buffer_duration = len(self.audio_buffer) * CHUNK / RATE
                            if buffer_duration >= MIN_BUFFER_DURATION:
                                self.process_audio_chunk(language)
                        self.audio_buffer = []
                        self.is_speaking = False
                else:
                    self.audio_buffer.append(audio_int16)

    def process_audio_chunk(self, language):
        audio_chunk = np.concatenate(self.audio_buffer).tobytes()
        transcription = self.transcribe(audio_chunk, language)
        if transcription:
            print(f"Transcription: {transcription}")
            self.send_to_llm(transcription)
        self.audio_buffer = []

    def transcribe(self, audio_chunk):
        input_features = self.processor(audio_chunk, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)

        if self.language == "hi":
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="hi", task="transcribe")
            output_sequences = self.model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_length=448
            )
            return self.processor.batch_decode(output_sequences, skip_special_tokens=True)[0]

        elif self.language == "en":
            forced_decoder_ids = self.processors["en"].get_decoder_prompt_ids(language="en", task="transcribe")
            output_sequences = self.models["en"].generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_length=30
            )
            return self.processors["en"].batch_decode(output_sequences, skip_special_tokens=True)[0]

    def send_to_llm(self, transcription):
        try:
            
            print(transcription)
            engine.say(transcription)
            engine.runAndWait()
                # engine.say(transcription)
                # engine.runAndWait()
            
        except requests.exceptions.RequestException as e:
            print("Error connecting to LLM service:", e)

    def stop(self):
        self.running = False
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

def main():
    language = input("Enter language (hi for Hindi, en for English): ").strip().lower()
    if language not in ["hi", "en"]:
        print("Invalid language. Use 'hi' or 'en'.")
        return
    processor = SpeechProcessor()
    try:
        processor.start_stream(language)
    except KeyboardInterrupt:
        processor.stop()

if __name__ == "__main__":
    main()
