# Omnilingual Voice AI Bot (OmniASR + Sherpa-ONNX)

This **multilingual voice conversation bot** records microphone audio, transcribes the speech using **Meta’s OmniASR (`omnilingual_1b`)** via **Sherpa-ONNX**, sends the text to an LLM for reasoning, then converts the reply back into speech and plays it automatically. You can speak in any of 1600+ languages, and the bot will understand and reply in that same language.

---

## Features

- Real-time microphone recording  
- OmniASR omnilingual_1b offline speech-to-text  
- Automatic language detection  
- LLM response generation  
- Text-to-speech output  
- Automatic local audio playback  
---

## Requirements

- macOS Apple Silicon  
- Python 3.10+  
- Homebrew  
- Xcode Command Line Tools (`xcode-select --install`)

---

## Setup Instructions

### 1. Clone the repository

### 2. Create a virtual environment
python3 -m venv audio_llm
source audio_llm/bin/activate

### 3. Install Python dependencies
pip install -r requirements.txt

### 4. Install system dependencies
brew install cmake ffmpeg libsndfile

### 5. Build Sherpa-ONNX from source
cd ~
git clone https://github.com/k2-fsa/sherpa-onnx.git
mkdir -p sherpa-build/sherpa-onnx/build
cd sherpa-build/sherpa-onnx/build
cmake ../../sherpa-onnx -DCMAKE_BUILD_TYPE=Release
make -j8

### 6. Verify OmniASR model works
/Users/<you>/sherpa-build/sherpa-onnx/build/bin/sherpa-onnx-offline \
  --tokens="asr_models/omnilingual_1b/tokens.txt" \
  --omnilingual-asr-model="asr_models/omnilingual_1b/model.onnx" \
  "asr_models/omnilingual_1b/test_wavs/en.wav"

## Run the Omnilingual Voice Conversation Bot
python -m streamlit run app.py

## Architecture

```mermaid
flowchart LR
    A[User Speech] --> B[ASR: OmniASR]
    B --> C[Transcript]
    C --> D[LLM: LangChain + GPT]
    D --> E[System Prompt Mode]
    E --> F[AI Response]
    F --> G[TTS Output]
    G --> H[Spoken Reply]
```

