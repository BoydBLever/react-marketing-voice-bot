import os
import requests

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL_ID = "csukuangfj/sherpa-onnx-omnilingual-asr-1600-languages-1B-ctc-2025-11-12"
API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL_ID}"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

def transcribe_with_omni_asr(audio_path: str) -> str:
    """
    Transcribe audio using Meta's OmniASR model via HuggingFace Inference API.
    Supports 1600+ languages as of November 10, 2025.
    """
    print("Using OmniASR with HuggingFace Router.")
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    response = requests.post(
        API_URL,
        headers=headers,
        data=audio_bytes
    )

    response.raise_for_status()
    data = response.json()

    # HF returns a list of predictions; take top result
    if isinstance(data, list) and "text" in data[0]:
        return data[0]["text"]
    else:
        return str(data)
