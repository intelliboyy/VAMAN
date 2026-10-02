# VAMAN

Voice assistant for PSIT College of Engineering. It transcribes Hindi or Indian-accent English speech, retrieves campus Q&A from a local Chroma index, and answers with a local Ollama model.

## What this repo contains

| File | Role |
| --- | --- |
| `un.py` | Full local pipeline: STT → retrieval → Ollama → pyttsx3 |
| `Anshuman.py` | Wake word (`hello`) plus speaker check, then streaming STT |
| `api_wala.py` | FastAPI `/ask` endpoint (text in, text out) |
| `api_wala_client_side.py` | Sample client for `/ask` |
| `merged_pipeline.py` | Text RAG loop with optional ElevenLabs TTS |
| `dia_tts.py` | Nari Dia 1.6B text-to-speech helper |
| `vector_search.py` | Builds or loads the Chroma retriever from `vaman_dataset1.csv` |

Local virtualenvs (`24_ENV/`, `naari/`), the cloned [nari-labs/dia](https://github.com/nari-labs/dia) tree, Chroma data, and voice embeddings stay off GitHub.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) with `llama3.2` and `mxbai-embed-large`
- Microphone for the STT scripts
- Optional GPU for Whisper / wav2vec / Dia

```bash
pip install -r requirements.txt
copy .env.example .env
```

On first retrieval run, keep `VAMAN_REBUILD_DB=0` after the index exists. Set it to `1` only when you need to rebuild from the CSV.

## Run

Text API:

```bash
uvicorn api_wala:app --reload --port 8000
python api_wala_client_side.py
```

Local voice assistant:

```bash
python un.py
```

Optional ElevenLabs speech (needs keys in `.env`):

```bash
python merged_pipeline.py
```

Dia TTS (install Dia separately, then run from a Dia-capable environment):

```bash
pip install git+https://github.com/nari-labs/dia.git
python dia_tts.py
```

Speech models used at runtime:

- Hindi: `ai4bharat/indicwav2vec-hindi`
- English: `Tejveer12/Indian-Accent-English-Whisper-Finetuned`

## Safety notes

Do not commit `.env`, voice embeddings, or API keys. If an ElevenLabs key was ever pasted into a local script, rotate it in the ElevenLabs dashboard.
